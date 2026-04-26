#!/usr/bin/env python3
"""Evaluate the trained GRPO LoRA adapter on the real Olist 30-day dataset.

Loads the merged model (base + GRPO LoRA adapter from HuggingFace), then runs
the full 30-day MedusaEnv episode using OlistDayGenerator and measures:
  - Per-day reward and commit success
  - JSON format validity rate
  - Valid action rate
  - Total cumulative reward vs a rule-based expert baseline
  - Per-day breakdown table

Usage (local, GPU recommended):
    export HF_TOKEN=hf_...
    python eval_grpo_olist.py

    # Override model:
    python eval_grpo_olist.py --model anubhavkamal/medusa-qwen-grpo --base Qwen/Qwen2.5-3B-Instruct

Environment variables:
    HF_TOKEN       - HuggingFace token for downloading gated models
    GRPO_MODEL_ID  - Hub model repo (default: anubhavkamal/medusa-qwen-grpo)
    BASE_MODEL_ID  - Base model (default: Qwen/Qwen2.5-3B-Instruct)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: allow running from repo root or any subdirectory
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medusa_prompts import OLIST_SYSTEM_PROMPT, SYSTEM_PROMPT, VALID_ACTIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GRPO_MODEL_ID = os.environ.get("GRPO_MODEL_ID", "anubhavkamal/medusa-qwen-grpo")
DEFAULT_BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO model on the Olist 30-day Medusa gauntlet."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GRPO_MODEL_ID,
        help="HuggingFace repo ID of the GRPO LoRA adapter (default: %(default)s)",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE_MODEL_ID,
        help="Base model ID used during GRPO training (default: %(default)s)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per action (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit quantization (saves VRAM but slower)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        default=False,
        help="Skip merging the LoRA adapter (use PEFT model.generate() directly)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Episode seed passed to the environment reset (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/olist",
        help="Relative path to Olist CSV data dir (default: %(default)s)",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        default=False,
        help="Also run the rule-based expert baseline for comparison",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save full results as JSON",
    )

    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--use-olist-prompt",
        action="store_true",
        default=False,
        help=(
            "Replace the generic SYSTEM_PROMPT with OLIST_SYSTEM_PROMPT: a shorter, "
            "Olist-specific prompt with concrete column examples and CoT guidance. "
            "Recommended when the GRPO model struggles on day-1 (no retraining needed)."
        ),
    )
    prompt_group.add_argument(
        "--system-prompt-file",
        default="",
        metavar="PATH",
        help="Load a custom system prompt from a plain-text file (overrides both defaults).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the base model + GRPO LoRA adapter and return (model, tokenizer)."""
    print(f"[load] Base model : {args.base}")
    print(f"[load] GRPO adapter: {args.model}")
    print(f"[load] 4-bit quant : {args.load_in_4bit}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    from peft import PeftModel

    token = HF_TOKEN or None

    bnb_cfg = None
    if args.load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("[load] loading base model weights …")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
        device_map="auto",
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] attaching GRPO LoRA adapter …")
    model = PeftModel.from_pretrained(base_model, args.model, token=token)

    if not args.no_merge:
        print("[load] merging adapter into base weights …")
        model = model.merge_and_unload()

    model.eval()
    print("[load] model ready ✓")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------


def resolve_system_prompt(args: argparse.Namespace) -> str:
    """Return the system prompt to use for this eval run."""
    if getattr(args, "system_prompt_file", ""):
        path = Path(args.system_prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"--system-prompt-file not found: {path}")
        prompt = path.read_text(encoding="utf-8").strip()
        print(f"[prompt] loaded custom system prompt from {path}  ({len(prompt)} chars)")
        return prompt
    if getattr(args, "use_olist_prompt", False):
        print(f"[prompt] using OLIST_SYSTEM_PROMPT  ({len(OLIST_SYSTEM_PROMPT)} chars)")
        return OLIST_SYSTEM_PROMPT
    print(f"[prompt] using default SYSTEM_PROMPT  ({len(SYSTEM_PROMPT)} chars)")
    return SYSTEM_PROMPT


def generate_action(
    model,
    tokenizer,
    prompt_text: str,
    args: argparse.Namespace,
    system_prompt: str = "",
) -> str:
    """Run one forward pass and return the raw generated text."""
    import torch

    active_prompt = system_prompt or SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": active_prompt},
        {"role": "user", "content": prompt_text},
    ]

    # apply_chat_template → plain string, then tokenize separately.
    # (Some transformers versions return BatchEncoding instead of a plain
    # tensor when return_tensors="pt", which breaks model.generate().)
    chat_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature > 0 else None,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Action parsing  (mirrors train_medusa_grpo.py)
# ---------------------------------------------------------------------------


def parse_action(text: str):
    """Return (MedusaAction | None, error_str | None)."""
    from models import MedusaAction

    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
    if fenced:
        clean = fenced.group(1).strip()

    # XML-style <action>...</action>
    action_match = re.search(r"<action>(.*?)</action>", clean, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        params: Dict[str, Any] = {}
        args_match = re.search(r"<args>(.*?)</args>", clean, re.DOTALL)
        if args_match:
            try:
                params = json.loads(args_match.group(1).strip())
            except json.JSONDecodeError:
                return None, "invalid_xml_args_json"
        return MedusaAction(action=action, params=params), None

    # JSON object
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

    from models import MedusaAction
    return MedusaAction(action=action, params=params), None


# ---------------------------------------------------------------------------
# Rule-based expert baseline (mirrors train_medusa_grpo.py)
# ---------------------------------------------------------------------------


def get_expert_action(env):
    """Rule-based policy for baseline comparison."""
    from models import MedusaAction

    state = env.state
    if state.new_schema_cols and not state.did_evolve_schema:
        return MedusaAction(
            action="EVOLVE_SILVER_SCHEMA",
            params={"column": state.new_schema_cols[0]},
        )
    if state.unhandled_anomalies_today:
        col = next(iter(state.unhandled_anomalies_today))
        op = state.unhandled_anomalies_today[col][0]
        if op == "quarantine":
            return MedusaAction(
                action="QUARANTINE_ROWS",
                params={"table": "bronze", "condition": f"{col} IS NULL"},
            )
        if op == "evolve":
            return MedusaAction(action="EVOLVE_SILVER_SCHEMA", params={"column": col})
        if op == "deduplicate":
            return MedusaAction(action="DEDUPLICATE", params={"key": col})
        return MedusaAction(action="CLEAN_COLUMN", params={"col": col, "op": op})

    if not state.did_dedup_today:
        return MedusaAction(action="DEDUPLICATE", params={})
    if not state.did_merge_today:
        return MedusaAction(action="EXECUTE_MERGE", params={})
    return MedusaAction(action="COMMIT_DAY", params={})


# ---------------------------------------------------------------------------
# Single-day runner
# ---------------------------------------------------------------------------


def run_one_day(
    env,
    day: int,
    model,
    tokenizer,
    args: argparse.Namespace,
    use_expert: bool = False,
    system_prompt: str = "",
) -> Dict[str, Any]:
    """Run one day of the 30-day gauntlet and return per-day stats.

    If the env has already terminated (stage != "running") because a previous
    day failed grader / hit a terminal crash, we record the day as SKIPPED
    rather than letting the env's `if stage != "running"` short-circuit
    inflate `total_steps` with zombie 1-step rows.
    """
    if env.state.stage != "running":
        return {
            "day": day,
            "steps": 0,
            "committed": False,
            "skipped": True,
            "skip_reason": f"env_terminal:{env.state.stage}",
            "sum_reward": 0.0,
            "rewards": [],
            "valid_json_pct": 0.0,
            "valid_action_pct": 0.0,
            "parse_errors": {},
            "elapsed_s": 0.0,
        }

    day_rewards: List[float] = []
    day_steps = 0
    day_commits = 0
    valid_json_count = 0
    valid_action_count = 0
    total_actions = 0
    parse_errors: Dict[str, int] = {}
    committed = False
    t0 = time.time()

    # Run up to 10 steps (env enforces the same limit)
    for _ in range(12):
        prompt_text = env.generate_llm_prompt()
        day_steps += 1
        total_actions += 1

        if use_expert:
            action = get_expert_action(env)
            raw_text = json.dumps({"action": action.action, "params": action.params})
            parse_err = None
        else:
            raw_text = generate_action(model, tokenizer, prompt_text, args, system_prompt=system_prompt)
            action, parse_err = parse_action(raw_text)

        if parse_err:
            parse_errors[parse_err] = parse_errors.get(parse_err, 0) + 1
        else:
            valid_json_count += 1
            if action is not None and action.action in VALID_ACTIONS:
                valid_action_count += 1

        # Fallback: if we can't parse, send a safe no-op to avoid crash
        if action is None:
            from models import MedusaAction
            action = MedusaAction(action="PROFILE_TABLE", params={"table": "bronze"})

        obs = env.step(action)
        reward = float(obs.reward or 0.0)
        day_rewards.append(reward)

        if action.action == "COMMIT_DAY":
            committed = True
            day_commits += 1

        if obs.done:
            break

    elapsed = time.time() - t0
    return {
        "day": day,
        "steps": day_steps,
        "committed": committed,
        "skipped": False,
        "skip_reason": None,
        "sum_reward": sum(day_rewards),
        "rewards": day_rewards,
        "valid_json_pct": valid_json_count / max(total_actions, 1),
        "valid_action_pct": valid_action_count / max(total_actions, 1),
        "parse_errors": parse_errors,
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Full 30-day episode runner
# ---------------------------------------------------------------------------


def run_episode(
    model,
    tokenizer,
    args: argparse.Namespace,
    use_expert: bool = False,
    label: str = "GRPO",
    system_prompt: str = "",
) -> Dict[str, Any]:
    from scenarios import OlistDayGenerator
    from server.medusa_env import MedusaEnv

    gen = OlistDayGenerator(
        episode_seed=args.seed,
        n_rows=100,
        data_dir=args.data_dir,
    )
    env = MedusaEnv(day_generator=gen, max_steps=10 * 30)  # global step cap
    env.reset(seed=args.seed)

    print(f"\n{'='*60}")
    print(f"  Running 30-day episode  [{label}]")
    print(f"{'='*60}")

    per_day: List[Dict[str, Any]] = []
    total_reward = 0.0
    total_commits = 0
    total_valid_json = 0.0
    total_valid_action = 0.0
    total_steps = 0
    days_attempted = 0
    days_skipped = 0

    for day in range(1, 31):
        day_stats = run_one_day(
            env=env,
            day=day,
            model=model,
            tokenizer=tokenizer,
            args=args,
            use_expert=use_expert,
            system_prompt=system_prompt,
        )
        per_day.append(day_stats)
        total_reward += day_stats["sum_reward"]
        total_commits += int(day_stats["committed"])
        total_steps += day_stats["steps"]

        if day_stats.get("skipped"):
            days_skipped += 1
        else:
            days_attempted += 1
            total_valid_json += day_stats["valid_json_pct"]
            total_valid_action += day_stats["valid_action_pct"]

        trap_marker = ""
        trap_days = {8: "type_trap", 14: "oom_trap", 21: "schema_drift", 28: "null_nuke"}
        if day in trap_days:
            trap_marker = f"  ⚠ TRAP({trap_days[day]})"

        if day_stats.get("skipped"):
            print(
                f"  Day {day:02d}{trap_marker:<22}  SKIPPED "
                f"(env terminal: {day_stats['skip_reason']})"
            )
        else:
            commit_icon = "✓" if day_stats["committed"] else "✗"
            print(
                f"  Day {day:02d}{trap_marker:<22}  "
                f"steps={day_stats['steps']:2d}  "
                f"reward={day_stats['sum_reward']:+.2f}  "
                f"commit={commit_icon}  "
                f"json_ok={day_stats['valid_json_pct']:.0%}  "
                f"act_ok={day_stats['valid_action_pct']:.0%}  "
                f"({day_stats['elapsed_s']:.1f}s)"
            )

    denom = max(days_attempted, 1)
    mean_valid_json = total_valid_json / denom
    mean_valid_action = total_valid_action / denom

    summary = {
        "label": label,
        "total_reward": round(total_reward, 4),
        "env_cumulative_reward": round(env.state.cumulative_reward, 4),
        "days_committed": total_commits,
        "days_attempted": days_attempted,
        "days_skipped": days_skipped,
        "total_steps": total_steps,
        "mean_valid_json_pct": round(mean_valid_json, 4),
        "mean_valid_action_pct": round(mean_valid_action, 4),
        "per_day": per_day,
    }

    print(f"\n  {'─'*50}")
    print(f"  TOTAL reward         : {total_reward:+.4f}")
    print(f"  Env cumulative reward: {env.state.cumulative_reward:+.4f}")
    print(f"  Days committed       : {total_commits}/30")
    print(f"  Days attempted       : {days_attempted}/30  (skipped: {days_skipped})")
    print(f"  Total steps          : {total_steps}")
    print(f"  Mean JSON valid rate : {mean_valid_json:.1%}  (over attempted days)")
    print(f"  Mean action valid    : {mean_valid_action:.1%}  (over attempted days)")
    return summary


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------


def print_comparison(grpo_res: Dict, expert_res: Optional[Dict]) -> None:
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    headers = ["Metric", "GRPO Model"]
    rows = [
        ["Total reward", f"{grpo_res['total_reward']:+.4f}"],
        ["Env cumulative", f"{grpo_res['env_cumulative_reward']:+.4f}"],
        ["Days committed", f"{grpo_res['days_committed']}/30"],
        [
            "Days attempted",
            f"{grpo_res.get('days_attempted', '?')}/30 "
            f"(skipped {grpo_res.get('days_skipped', 0)})",
        ],
        ["Total steps", str(grpo_res["total_steps"])],
        ["Mean JSON valid", f"{grpo_res['mean_valid_json_pct']:.1%}"],
        ["Mean action valid", f"{grpo_res['mean_valid_action_pct']:.1%}"],
    ]

    if expert_res:
        headers.append("Expert Baseline")
        rows[0].append(f"{expert_res['total_reward']:+.4f}")
        rows[1].append(f"{expert_res['env_cumulative_reward']:+.4f}")
        rows[2].append(f"{expert_res['days_committed']}/30")
        rows[3].append(
            f"{expert_res.get('days_attempted', '?')}/30 "
            f"(skipped {expert_res.get('days_skipped', 0)})"
        )
        rows[4].append(str(expert_res["total_steps"]))
        rows[5].append(f"{expert_res['mean_valid_json_pct']:.1%}")
        rows[6].append(f"{expert_res['mean_valid_action_pct']:.1%}")
        # Compute delta
        delta_reward = grpo_res["total_reward"] - expert_res["total_reward"]
        delta_commits = grpo_res["days_committed"] - expert_res["days_committed"]
        print(f"\n  GRPO vs Expert  Δreward={delta_reward:+.4f}  Δcommits={delta_commits:+d}")

        # Trap-day analysis
        trap_days_set = {8, 14, 21, 28}
        for res in [grpo_res, expert_res]:
            trap_rewards = sum(
                d["sum_reward"] for d in res["per_day"] if d["day"] in trap_days_set
            )
            normal_rewards = sum(
                d["sum_reward"] for d in res["per_day"] if d["day"] not in trap_days_set
            )
            print(
                f"  [{res['label']}]  trap-day reward={trap_rewards:+.2f}  "
                f"normal-day reward={normal_rewards:+.2f}"
            )

    col_w = 22
    print("\n  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "  ".join("-" * col_w for _ in headers))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(col_w) for v in row))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        # Log into HF so gated models download correctly
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)
            print(f"[auth] logged in to HuggingFace ✓")
        except Exception as e:
            print(f"[auth] HF login skipped: {e}")

    # ------------------------------------------------------------------
    # Resolve system prompt (once, shared across all runs)
    # ------------------------------------------------------------------
    active_system_prompt = resolve_system_prompt(args)

    # ------------------------------------------------------------------
    # Load GRPO model
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args)

    # ------------------------------------------------------------------
    # Run GRPO agent
    # ------------------------------------------------------------------
    grpo_result = run_episode(
        model=model,
        tokenizer=tokenizer,
        args=args,
        use_expert=False,
        label="GRPO",
        system_prompt=active_system_prompt,
    )

    # ------------------------------------------------------------------
    # Optionally run expert baseline
    # ------------------------------------------------------------------
    expert_result = None
    if args.run_baseline:
        expert_result = run_episode(
            model=None,      # type: ignore[arg-type]
            tokenizer=None,  # type: ignore[arg-type]
            args=args,
            use_expert=True,
            label="Expert",
            system_prompt=active_system_prompt,
        )

    # ------------------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------------------
    print_comparison(grpo_result, expert_result)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    if args.output_json:
        out = {"grpo": grpo_result}
        if expert_result:
            out["expert"] = expert_result
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"[output] results saved to {args.output_json}")


if __name__ == "__main__":
    main()
