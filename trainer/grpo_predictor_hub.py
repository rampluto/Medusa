"""GRPO action predictor backed by a Hub-hosted LoRA adapter.

This module is the **production** entrypoint that satisfies the contract
enforced by `server.agent_policies.GrpoTrainedPolicy`. Set on the HF
Space:

    MEDUSA_GRPO_PREDICTOR=trainer.grpo_predictor_hub:predict

and these supporting variables (all read from `os.environ`):

    BASE_MODEL_ID    Qwen/Qwen2.5-3B-Instruct       (default)
    GRPO_MODEL_ID    your_user/medusa-qwen-grpo     (REQUIRED)
    HF_TOKEN         hf_…                           (optional; for gated repos)

    MEDUSA_GRPO_LOAD_IN_4BIT   "1" → 4-bit nf4 quantization (saves VRAM)
    MEDUSA_GRPO_NO_MERGE       "1" → keep the LoRA adapter wrapper
                               (saves a few seconds at startup, costs
                                a small per-token slowdown)
    MEDUSA_GRPO_MAX_NEW_TOKENS  default 192
    MEDUSA_GRPO_TEMPERATURE     default 0.1   (0 disables sampling)

The predictor mirrors `eval_grpo_olist.py` end-to-end:
  - same prompt format (SYSTEM_PROMPT + state-rendered user turn)
  - same action parser (fenced JSON or <action> XML)
  - same fallback to `PROFILE_TABLE bronze` on parse failure

The model load is **lazy** and **thread-safe**: the first call into
`predict()` triggers a one-time download + load (held under a lock so
two simultaneous requests don't race two GPU loads), and every
subsequent call reuses the cached bundle for the lifetime of the worker.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Tuple

# Repo root on sys.path so we can import top-level modules
# (`models`, `medusa_prompts`) regardless of how the worker was launched.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medusa_prompts import SYSTEM_PROMPT, VALID_ACTIONS  # noqa: E402
from models import MedusaAction  # noqa: E402

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy model bundle  (process-lifetime cache, thread-safe)
# ---------------------------------------------------------------------------

_BUNDLE_LOCK = Lock()
_BUNDLE: Optional[Tuple[Any, Any]] = None  # (model, tokenizer)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _load_bundle() -> Tuple[Any, Any]:
    """Load (or return cached) (model, tokenizer) from the Hub.

    Heavy imports (`torch`, `transformers`, `peft`) are deferred to this
    function so a worker that never serves a `grpo_trained` request never
    pays the import cost.
    """
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    with _BUNDLE_LOCK:
        if _BUNDLE is not None:
            return _BUNDLE

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_id = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct").strip()
        adapter_id = os.environ.get("GRPO_MODEL_ID", "").strip()
        if not adapter_id:
            raise RuntimeError(
                "trainer.grpo_predictor_hub: GRPO_MODEL_ID is not set — "
                "point it at your Hub LoRA adapter "
                "(e.g. 'myuser/medusa-qwen-grpo')."
            )

        token = os.environ.get("HF_TOKEN", "").strip() or None
        load_in_4bit = _env_flag("MEDUSA_GRPO_LOAD_IN_4BIT")
        no_merge = _env_flag("MEDUSA_GRPO_NO_MERGE")

        bnb_cfg = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        LOGGER.info(
            "grpo_hub_predictor_loading base=%s adapter=%s 4bit=%s no_merge=%s",
            base_id,
            adapter_id,
            load_in_4bit,
            no_merge,
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            device_map="auto",
            token=token,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_id, token=token)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = PeftModel.from_pretrained(base, adapter_id, token=token)
        if not no_merge:
            LOGGER.info("grpo_hub_predictor_merging_adapter")
            model = model.merge_and_unload()
        model.eval()

        _BUNDLE = (model, tokenizer)
        LOGGER.info("grpo_hub_predictor_ready")
        return _BUNDLE


def warmup() -> None:
    """Trigger the lazy load explicitly (e.g. from server startup hooks)."""
    _load_bundle()


# ---------------------------------------------------------------------------
# Prompt rendering  (port of MedusaEnv.generate_llm_prompt, decoupled from env)
# ---------------------------------------------------------------------------


def _render_prompt_from_state(state: Any) -> str:
    """Reconstruct the env's user-prompt from a `MedusaState`-shaped object.

    This is a 1:1 port of `MedusaEnv.generate_llm_prompt()` — every field
    accessed there exists on `MedusaState`. We use `getattr(..., default)`
    so the dataframe-cleaner path's `SimpleNamespace` (which only carries
    a subset of fields) degrades gracefully instead of raising.
    """
    current_day = getattr(state, "current_day", 1)
    pk_col = getattr(state, "pk_col", "") or ""
    contract_cols = list(getattr(state, "current_contract_columns", []) or [])
    silver_rows = int(getattr(state, "silver_row_count", 0) or 0)
    total_raw = int(getattr(state, "total_raw_rows", 0) or 0)
    did_dedup = bool(getattr(state, "did_dedup_today", False))
    did_evolve = bool(getattr(state, "did_evolve_schema", False))
    new_schema_cols = list(getattr(state, "new_schema_cols", []) or [])
    last_action_result = getattr(state, "last_action_result", "") or ""
    trap_type = getattr(state, "trap_type", "") or ""

    unhandled = getattr(state, "unhandled_anomalies_today", {}) or {}
    if not isinstance(unhandled, dict):
        unhandled = {}

    lines = [
        f"=== DAY {current_day} PIPELINE STATE ===",
        f"Primary Key: '{pk_col}'",
        f"Current Target Schema: {contract_cols}",
        f"Total Silver Rows Committed (All Days): {silver_rows}",
        f"Today's Incoming Batch Rows: {total_raw}",
        f"Is deduplication complete for today? {'Yes' if did_dedup else 'No'}",
        "",
        "=== ACTIVE ANOMALIES TODAY ===",
    ]

    if not unhandled:
        lines.append("None. The pipeline is clean. Ready to DEDUPLICATE and EXECUTE_MERGE.")
    else:
        lines.append(f"Trap detected: {trap_type}" if trap_type else "Normal corruption observed.")
        for col, ops in unhandled.items():
            lines.append(f"- Column '{col}': requires ({', '.join(ops)})")

    if new_schema_cols and not did_evolve:
        lines.append(
            f"- Schema Drift Detected: {new_schema_cols} appear in the source but not the contract. "
            "Requires EVOLVE_SILVER_SCHEMA."
        )

    lines.append("")
    lines.append("=== PREVIOUS ACTION FEEDBACK ===")
    lines.append(f"{last_action_result}\n" if last_action_result else "No previous action taken today.\n")
    lines.append(
        "Based on the data engineering required, output a valid JSON `MedusaAction` "
        "with keys 'action' and 'params'."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing  (mirrors eval_grpo_olist.parse_action)
# ---------------------------------------------------------------------------


def _parse_action(text: str) -> Tuple[Optional[MedusaAction], Optional[str]]:
    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
    if fenced:
        clean = fenced.group(1).strip()

    # XML-style <action>...</action>
    action_match = re.search(r"<action>(.*?)</action>", clean, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        params: dict[str, Any] = {}
        args_match = re.search(r"<args>(.*?)</args>", clean, re.DOTALL)
        if args_match:
            try:
                params = json.loads(args_match.group(1).strip())
            except json.JSONDecodeError:
                return None, "invalid_xml_args_json"
        if action not in VALID_ACTIONS:
            return None, f"invalid_action:{action}"
        return MedusaAction(action=action, params=params), None

    # JSON object (greedy across newlines)
    json_match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not json_match:
        return None, "missing_json"
    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"

    action = payload.get("action")
    params = payload.get("params", {})
    if not isinstance(action, str) or action not in VALID_ACTIONS:
        return None, f"invalid_action:{action}"
    if not isinstance(params, dict):
        return None, "invalid_params"
    return MedusaAction(action=action, params=params), None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _generate(model, tokenizer, prompt_text: str) -> str:
    import torch

    max_new_tokens = int(os.environ.get("MEDUSA_GRPO_MAX_NEW_TOKENS", "192"))
    temperature = float(os.environ.get("MEDUSA_GRPO_TEMPERATURE", "0.1"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    chat_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def predict(*, task: Any, state: Any, observation: Any) -> MedusaAction:
    """Predictor entrypoint matching `GrpoTrainedPolicy.select_action`.

    On any failure (load error, generation error, parse error) we return
    `PROFILE_TABLE bronze`, which is the same safe fallback used by
    `eval_grpo_olist.py`. This keeps the UI alive even when the model is
    misconfigured — the action trace will show the fallbacks and you can
    diagnose from logs.
    """
    try:
        model, tokenizer = _load_bundle()
    except Exception:
        LOGGER.exception("grpo_hub_predictor_load_failed")
        # Re-raise so build_agent surfaces the misconfig as a 422; this
        # only happens once per worker, on the *first* call.
        raise

    prompt_text = _render_prompt_from_state(state)
    try:
        raw = _generate(model, tokenizer, prompt_text)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("grpo_hub_predictor_generate_failed err=%s", exc)
        return MedusaAction(action="PROFILE_TABLE", params={"table": "bronze"})

    LOGGER.debug("grpo_hub_predictor_raw_output text=%s", raw[:200])
    action, parse_err = _parse_action(raw)
    if action is None:
        LOGGER.warning(
            "grpo_hub_predictor_unparseable err=%s text=%s",
            parse_err,
            raw[:200],
        )
        return MedusaAction(action="PROFILE_TABLE", params={"table": "bronze"})
    return action
