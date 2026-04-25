#!/usr/bin/env python3
"""Train a MedusaEnv action policy with Unsloth + TRL GRPO.

Kaggle notebook quick start:

    !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets
    !python trainer/train_medusa_grpo.py --max-steps 200 --episodes 8

The script builds a GRPO prompt dataset from live MedusaEnv states. During
rewarding, each sampled completion is parsed as a MedusaAction, the saved state
prefix is replayed, and the action is scored by the real environment.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a careful Autonomous Data Engineer Agent solving the Medusa "
    "environment. Return only valid JSON with keys 'action' and 'params'."
)

VALID_ACTIONS = {
    "PROFILE_TABLE",
    "CLEAN_COLUMN",
    "DEDUPLICATE",
    "EVOLVE_SILVER_SCHEMA",
    "QUARANTINE_ROWS",
    "EXECUTE_MERGE",
    "COMMIT_DAY",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def import_unsloth_first():
    """Import Unsloth before Transformers and print a useful Kaggle fix if broken."""

    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel, PatchFastRL
    except Exception as exc:  # noqa: BLE001 - package import errors are often nested.
        message = str(exc)
        repair = (
            "Unsloth failed to import before training started.\n\n"
            "In a fresh Kaggle notebook cell, run this, then restart the session:\n"
            "!pip install -q --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo\n"
            "!pip install -q --upgrade --force-reinstall --no-cache-dir transformers trl datasets accelerate bitsandbytes\n\n"
            "Then run this trainer again. If you previously imported transformers or trl in the same "
            "notebook session, a restart is important."
        )
        if "_get_template_variables" in message or "chat_template_utils" in message:
            repair += (
                "\n\nDetected the known Unsloth/Transformers chat-template mismatch "
                "(`_get_template_variables`). The reinstall + restart above is the fix."
            )
        raise SystemExit(repair) from exc

    return FastLanguageModel, PatchFastRL


FastLanguageModel, PatchFastRL = import_unsloth_first()

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import TrainerCallback

from models import MedusaAction
from scenarios import DayDataGenerator, detect_column_roles
from server.medusa_env import MedusaEnv


warnings.filterwarnings(
    "ignore",
    message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)


class RandomizedSchemaGenerator(DayDataGenerator):
    """Randomized schema generator to reduce column-name memorization."""

    def __init__(self, episode_seed: int, n_rows: int = 100) -> None:
        rng = random.Random(episode_seed)
        domain = rng.choice(["medical", "finance", "logistics", "gaming", "retail"])
        self.COLUMN_SPEC = {
            f"{domain}_id_col": "id",
            f"{domain}_metric_1": "numeric",
            f"{domain}_metric_2": "numeric",
            f"{domain}_category": "categorical",
            f"{domain}_notes": "string",
            f"{domain}_date": "date",
        }
        self.episode_seed = episode_seed
        self.n_rows = n_rows
        self._day_anomalies: dict[int, list[tuple[str, str]]] = {}
        self._sample_df = self._build_base_data(seed=episode_seed, n=n_rows)
        self._pk_col = f"{domain}_id_col"
        self._roles = detect_column_roles(self._sample_df, primary_key=self._pk_col)
        self._numeric_cols = list(self._roles.get("numeric", []))
        self._string_cols = list(
            self._roles.get("string", []) + self._roles.get("categorical", [])
        )
        self._baseline_schema = list(self._sample_df.columns)
        self._build_anomaly_schedule()


def get_expert_action(env: MedusaEnv) -> MedusaAction:
    """Small rule-based policy used only to create reachable training states."""

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


def make_env(seed: int, rows: int, max_env_steps: int) -> MedusaEnv:
    generator = RandomizedSchemaGenerator(episode_seed=seed, n_rows=rows)
    env = MedusaEnv(day_generator=generator, max_steps=max_env_steps)
    env.reset(seed=seed)
    return env


def action_to_dict(action: MedusaAction) -> dict[str, Any]:
    return {"action": str(action.action), "params": dict(action.params)}


def build_grpo_dataset(args: argparse.Namespace) -> Dataset:
    rows: list[dict[str, Any]] = []
    pbar = tqdm(range(args.episodes), desc="Building Medusa GRPO prompts")
    for ep in pbar:
        seed = args.seed + ep
        env = make_env(seed, args.rows_per_day, args.max_env_steps)
        history: list[dict[str, Any]] = []

        while len(history) < args.max_prefix_steps and len(rows) < args.dataset_size:
            prompt_text = env.generate_llm_prompt()
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ],
                    "seed": seed,
                    "history_json": json.dumps(history),
                }
            )

            action = get_expert_action(env)
            history.append(action_to_dict(action))
            obs = env.step(action)
            if obs.done:
                break

        pbar.set_postfix(rows=len(rows))
        if len(rows) >= args.dataset_size:
            break

    if not rows:
        raise RuntimeError("No GRPO prompts were generated.")
    return Dataset.from_list(rows)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def parse_action(text: str) -> tuple[MedusaAction | None, str | None]:
    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", clean, re.DOTALL)
    if fenced:
        clean = fenced.group(1).strip()

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
        return MedusaAction(action=action, params=params), None

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
        return None, "invalid_action"
    if not isinstance(params, dict):
        return None, "invalid_params"
    return MedusaAction(action=action, params=params), None


class RewardLogger:
    def __init__(self, output_dir: Path, every: int = 10) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "reward_log.jsonl"
        self.every = max(1, every)
        self.calls = 0
        self.recent = deque(maxlen=100)

    def write(self, rewards: list[float], failures: dict[str, int]) -> None:
        self.calls += 1
        self.recent.extend(rewards)
        mean_recent = sum(self.recent) / max(len(self.recent), 1)
        record = {
            "reward_call": self.calls,
            "batch_mean_reward": sum(rewards) / max(len(rewards), 1),
            "moving_mean_reward": mean_recent,
            "batch_min_reward": min(rewards) if rewards else 0.0,
            "batch_max_reward": max(rewards) if rewards else 0.0,
            "failures": failures,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if self.calls % self.every == 0:
            tqdm.write(
                "[reward] "
                f"call={self.calls} mean={record['batch_mean_reward']:.3f} "
                f"moving={mean_recent:.3f} failures={failures}"
            )


def make_reward_funcs(args: argparse.Namespace, logger: RewardLogger):
    def json_format_reward(completions: list[Any], **_: Any) -> list[float]:
        rewards = []
        for completion in completions:
            action, error = parse_action(completion_text(completion))
            rewards.append(0.2 if action is not None and error is None else -0.5)
        return rewards

    def medusa_env_reward(
        completions: list[Any],
        seed: list[int],
        history_json: list[str],
        **_: Any,
    ) -> list[float]:
        rewards: list[float] = []
        failures: dict[str, int] = {}
        for completion, row_seed, raw_history in zip(completions, seed, history_json):
            action, error = parse_action(completion_text(completion))
            if action is None:
                failures[error or "parse_error"] = failures.get(error or "parse_error", 0) + 1
                rewards.append(args.invalid_action_reward)
                continue

            env = make_env(int(row_seed), args.rows_per_day, args.max_env_steps)
            try:
                for item in json.loads(raw_history):
                    obs = env.step(
                        MedusaAction(
                            action=item["action"],
                            params=item.get("params", {}),
                        )
                    )
                    if obs.done:
                        break
                obs = env.step(action)
                raw_reward = float(obs.reward if obs.reward is not None else 0.0)
                scaled = raw_reward / args.env_reward_scale
                rewards.append(max(args.reward_clip_min, min(args.reward_clip_max, scaled)))
            except Exception as exc:  # noqa: BLE001 - reward functions must not crash training.
                key = f"env_error:{type(exc).__name__}"
                failures[key] = failures.get(key, 0) + 1
                rewards.append(args.invalid_action_reward)

        logger.write(rewards, failures)
        return rewards

    return [json_format_reward, medusa_env_reward]


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if logs:
            tqdm.write(f"[train] step={state.global_step} {json.dumps(logs, default=str)}")

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        tqdm.write(f"[save] checkpoint written at step {state.global_step}")


def supported_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def load_unsloth_model(args: argparse.Namespace):
    try:
        PatchFastRL("GRPO", FastLanguageModel)
    except Exception as exc:  # noqa: BLE001 - older Unsloth builds may patch automatically.
        tqdm.write(f"[warn] PatchFastRL skipped: {exc}")

    model_kwargs = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "fast_inference": args.fast_inference,
        "max_lora_rank": args.lora_rank,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    model, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = args.max_completion_length

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedusaEnv with GRPO via Unsloth + TRL.")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    parser.add_argument("--output-dir", default="trainer/medusa-grpo-output")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--rows-per-day", type=int, default=50)
    parser.add_argument("--max-prefix-steps", type=int, default=160)
    parser.add_argument("--max-env-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--env-reward-scale", type=float, default=20.0)
    parser.add_argument("--invalid-action-reward", type=float, default=-1.0)
    parser.add_argument("--reward-clip-min", type=float, default=-5.0)
    parser.add_argument("--reward-clip-max", type=float, default=5.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--reward-log-every", type=int, default=10)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--fast-inference", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    parser.set_defaults(load_in_4bit=True)
    args, unknown = parser.parse_known_args()
    if unknown:
        tqdm.write(f"[setup] ignoring notebook launcher args: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write("[setup] building Medusa prompt dataset")
    train_dataset = build_grpo_dataset(args)
    tqdm.write(f"[setup] dataset rows: {len(train_dataset)}")

    tqdm.write("[setup] loading Unsloth model")
    model, tokenizer = load_unsloth_model(args)

    from trl import GRPOConfig, GRPOTrainer

    config_values = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.1,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "logging_steps": args.logging_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "report_to": "none",
        "remove_unused_columns": False,
        "use_vllm": args.use_vllm,
    }
    grpo_args = GRPOConfig(**supported_kwargs(GRPOConfig, config_values))
    reward_logger = RewardLogger(output_dir, every=args.reward_log_every)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=make_reward_funcs(args, reward_logger),
        args=grpo_args,
        train_dataset=train_dataset,
        callbacks=[ProgressCallback()],
    )

    tqdm.write("[train] starting GRPO")
    trainer.train()
    tqdm.write("[save] saving LoRA adapter")
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    tqdm.write(f"[done] saved to {output_dir}")


if __name__ == "__main__":
    main()
