#!/usr/bin/env python3
"""SFT a Qwen2.5-3B-Instruct adapter on the Olist 30-day MEDUSA dataset.

End-to-end driver for the polished-demo SFT distribution:

  1. (Optional) Build the SFT JSONL via `generate_sft_dataset_olist.py`,
     i.e. the OlistDayGenerator + expert-solver hack — the SFT trajectory
     becomes byte-identical to the `eval_grpo_olist.py --run-baseline`
     trajectory, so the model trained against it should hit ~100% commit
     rate, ~100% JSON-valid rate, and total reward indistinguishable from
     the rule-based baseline.
  2. Train a LoRA adapter on top of `Qwen/Qwen2.5-3B-Instruct` (the same
     base used by `train_medusa_grpo.py` and `eval_grpo_olist.py`, so the
     adapter slots straight into the GRPO continuation pipeline).

By default, this script auto-builds the dataset if `--dataset` does not
already exist; pass `--regenerate` to force rebuild even when the file
already exists, or pass an existing path to skip the build entirely.

Quick start (local, GPU recommended):

    python trainer/train_sft_olist.py \
        --output-dir trainer/medusa-sft-olist-output

Push the trained adapter to the Hub (for `eval_grpo_olist.py --model`):

    HF_TOKEN=hf_... python trainer/train_sft_olist.py \
        --push-to-hub --hub-model-id myuser/medusa-qwen-sft-olist

Notes
-----
* This script is intentionally self-contained (no edits to `train_sft.py`).
  If you already have a tuned `train_sft.py` workflow, you can also build
  the JSONL with `generate_sft_dataset_olist.py` and feed that file into
  the existing trainer instead.
* LoRA targets are kept at `["q_proj","k_proj","v_proj","o_proj"]` to match
  `train_medusa_grpo.py` defaults — this is what makes the SFT adapter a
  drop-in `--sft-adapter` for GRPO continuation training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# IMPORTANT: must be set as early as possible in HF Jobs containers.
# Some libraries import numpy/MKL under the hood (or via torch/transformers),
# and if MKL picks INTEL threading while OpenMP is GNU (libgomp), it can crash.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local sibling: the Olist-grounded SFT dataset builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_sft_dataset_olist import generate_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT a LoRA adapter on the Olist-grounded MEDUSA dataset.",
    )

    # ---- Dataset ----------------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        default="trainer/sft_dataset_olist.jsonl",
        help=(
            "Path to JSONL dataset (chatml `messages`). If missing, it is "
            "auto-built via generate_sft_dataset_olist.py."
        ),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force-rebuild the SFT dataset even when --dataset already exists.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/olist",
        help="Olist CSV directory (relative to repo root).",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--rows-per-day", type=int, default=100)
    parser.add_argument("--paraphrase-passes", type=int, default=4)
    parser.add_argument(
        "--with-mistakes",
        dest="with_mistakes",
        action="store_true",
        default=False,
        help="Inject the 10%% wrong-action branch in the SFT data (recovery).",
    )

    # ---- Model ------------------------------------------------------------
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help=(
            "Base model. MUST match `train_medusa_grpo.py --model-name` and "
            "`eval_grpo_olist.py --base` — otherwise the SFT LoRA shape will "
            "be incompatible with downstream GRPO continuation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trainer/medusa-sft-olist-output",
        help="Local output / checkpoint directory.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="",
        help="Optional Hub repo id for push (e.g. user/medusa-qwen-sft-olist).",
    )

    # ---- LoRA -------------------------------------------------------------
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # ---- Training ---------------------------------------------------------
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Fraction of samples held out for validation (0 disables eval).",
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostics (trainable params, label masking).",
    )
    parser.add_argument("--push-to-hub", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset bootstrapping
# ---------------------------------------------------------------------------


def maybe_build_dataset(args: argparse.Namespace) -> Path:
    """Build the Olist SFT JSONL if it doesn't already exist (or --regenerate)."""
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()

    if dataset_path.exists() and not args.regenerate:
        print(f"[sft] reusing existing dataset at {dataset_path}")
        return dataset_path

    print(f"[sft] building Olist SFT dataset -> {dataset_path}")
    generate_dataset(
        out_path=dataset_path,
        episodes=args.episodes,
        n_rows=args.rows_per_day,
        data_dir=args.data_dir,
        paraphrase_passes=args.paraphrase_passes,
        no_mistakes=not args.with_mistakes,
    )
    return dataset_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_text(record: dict[str, Any], tokenizer) -> dict[str, str]:
    messages = record.get("messages", [])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def train(args: argparse.Namespace, dataset_path: Path) -> None:
    # Heavy imports deferred so `--help` and dataset-only runs don't pay the cost.
    import inspect

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if args.debug:
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("Compute capability:", torch.cuda.get_device_capability(0))
        print("bf16_supported:", bf16_supported)

    print(f"[sft] loading dataset: {dataset_path}")
    ds = load_dataset("json", data_files=str(dataset_path))["train"]
    print(f"[sft] dataset rows: {len(ds)}")

    print(f"[sft] loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=(
            torch.bfloat16
            if bf16_supported
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        ),
        device_map="auto",
    )

    print("[sft] formatting chat samples")
    ds = ds.map(lambda row: build_text(row, tokenizer), remove_columns=ds.column_names)

    eval_dataset = None
    if args.eval_ratio and args.eval_ratio > 0:
        if not (0.0 < args.eval_ratio < 1.0):
            raise ValueError("--eval-ratio must be in (0, 1) or 0 to disable.")
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds = split["train"]
        eval_dataset = split["test"]
        print(
            "[sft] split:",
            f"train={len(train_ds)}",
            f"eval={len(eval_dataset)}",
            f"eval_ratio={args.eval_ratio}",
        )
    else:
        train_ds = ds
        print(f"[sft] train={len(train_ds)} (eval disabled)")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Match train_medusa_grpo.py default targets so the adapter is a
        # drop-in `--sft-adapter` for GRPO continuation training.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    effective_hub_model_id = args.hub_model_id.strip()
    if not effective_hub_model_id and args.push_to_hub:
        raise ValueError("--push-to-hub requires --hub-model-id.")

    sft_sig = inspect.signature(SFTConfig.__init__).parameters
    length_key = "max_length" if "max_length" in sft_sig else "max_seq_length"
    eval_strategy_key = (
        "eval_strategy" if "eval_strategy" in sft_sig else "evaluation_strategy"
    )
    eval_strategy_value = (
        "steps" if (eval_dataset is not None and args.eval_steps > 0) else "no"
    )

    sft_config_kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        fp16=bool(torch.cuda.is_available() and (not bf16_supported)),
        bf16=bf16_supported,
        push_to_hub=args.push_to_hub,
        hub_model_id=effective_hub_model_id if effective_hub_model_id else None,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        eval_steps=args.eval_steps,
    )
    sft_config_kwargs[length_key] = args.max_seq_len
    sft_config_kwargs[eval_strategy_key] = eval_strategy_value

    if args.debug:
        print(
            f"[sft][debug] SFTConfig length_key={length_key!r}, "
            f"eval_strategy_key={eval_strategy_key!r}"
        )

    sft_config = SFTConfig(**sft_config_kwargs)

    # SFTTrainer's constructor signature changes across TRL versions —
    # try the variants we know about, in order.
    trainer = None
    init_errors: list[str] = []
    for variant in ({"tokenizer": tokenizer}, {"processing_class": tokenizer}, {}):
        try:
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_ds,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                **variant,
            )
            break
        except TypeError as exc:
            init_errors.append(str(exc))

    if trainer is None:
        raise TypeError(
            f"Failed to initialize SFTTrainer. Errors: {init_errors}"
        )

    if args.debug:
        trainable, total = 0, 0
        for _, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"[sft][debug] params total={total:,} trainable={trainable:,}")
        if trainable == 0:
            raise RuntimeError(
                "No trainable parameters found. LoRA may not have attached. "
                "Try adjusting LoraConfig.target_modules for this model."
            )

        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        if "labels" in batch:
            labels0 = batch["labels"][0]
            masked = int((labels0 == -100).sum().item())
            total_tok = int(labels0.numel())
            unmasked = total_tok - masked
            print(
                f"[sft][debug] labels tokens={total_tok} "
                f"masked={masked} unmasked={unmasked}"
            )
            if unmasked == 0:
                raise RuntimeError(
                    "All labels masked (-100). The collator is producing no "
                    "supervised tokens — loss will be 0."
                )
        else:
            print("[sft][debug] WARNING: batch has no 'labels' key.")

    print("[sft] starting training")
    train_result = trainer.train()
    print(train_result)

    if eval_dataset is not None:
        print("[sft] final eval")
        print(trainer.evaluate())

    print(f"[sft] saving final checkpoint to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"[sft] pushing to hub: {effective_hub_model_id}")
        trainer.push_to_hub()

    print("[sft] done.")
    if torch.cuda.is_available():
        print(f"[sft] GPU used: {torch.cuda.get_device_name(0)}")
    else:
        print("[sft] running on CPU.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    dataset_path = maybe_build_dataset(args)
    train(args, dataset_path)


if __name__ == "__main__":
    main()
