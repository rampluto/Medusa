#!/usr/bin/env python3
"""Train MEDUSA SFT model locally or on Hugging Face Jobs.

This script is designed for:
1) Local/Colab execution
2) Remote execution inside `hf jobs run` containers

Input JSONL is expected to contain a `messages` field with chat turns.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

# IMPORTANT: must be set as early as possible in HF Jobs containers.
# Some libraries import numpy/MKL under the hood (or via torch/transformers), and
# if MKL picks INTEL threading while OpenMP is GNU (libgomp), it can crash.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA SFT model for MEDUSA.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSONL dataset.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./medusa-sft",
        help="Local output/checkpoint directory.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="",
        help="Optional Hub repo id for push (e.g. user/medusa-qwen-sft).",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.02,
        help="Fraction of samples to hold out for validation (0 disables eval).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Run evaluation every N steps (requires --eval-ratio > 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting / reproducibility.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print extra diagnostics (trainable params, label masking) to help debug "
            "0 loss / NaN grad_norm issues in remote HF Jobs logs."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final model/checkpoints to Hub.",
    )
    return parser.parse_args()


def build_text(record: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    messages = record.get("messages", [])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()

    # ----
    # BF16 safety: only enable if the GPU/runtime supports it.
    # TRL/Accelerate can behave badly when bf16=True on unsupported hardware.
    # ----
    bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if args.debug:
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("Compute capability:", torch.cuda.get_device_capability(0))
        print("bf16_supported:", bf16_supported)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset("json", data_files=args.dataset)["train"]

    print(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        # Keep model dtype aligned with precision flags.
        # If bf16 is supported, prefer bf16; else fp16 on CUDA; else fp32.
        torch_dtype=(
            torch.bfloat16
            if bf16_supported
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        ),
        device_map="auto",
    )

    print("Formatting chat samples...")
    ds = ds.map(lambda row: build_text(row, tokenizer), remove_columns=ds.column_names)

    # ----
    # Train/validation split (optional)
    # ----
    eval_dataset = None
    if args.eval_ratio and args.eval_ratio > 0:
        # Ensure ratio is sane
        if not (0.0 < args.eval_ratio < 1.0):
            raise ValueError("--eval-ratio must be in (0,1) or 0 to disable.")

        # `train_test_split` shuffles by default.
        split = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds = split["train"]
        eval_dataset = split["test"]
        print(
            "Dataset split:",
            f"train={len(train_ds)}",
            f"eval={len(eval_dataset)}",
            f"eval_ratio={args.eval_ratio}",
        )
    else:
        train_ds = ds
        print(f"Dataset: train={len(train_ds)} (eval disabled)")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    effective_hub_model_id = args.hub_model_id.strip()
    if not effective_hub_model_id and args.push_to_hub:
        raise ValueError("--push-to-hub requires --hub-model-id.")

    sft_config = SFTConfig(
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
        max_seq_length=args.max_seq_len,
        packing=False,
        seed=args.seed,
        evaluation_strategy=(
            "steps" if (eval_dataset is not None and args.eval_steps > 0) else "no"
        ),
        eval_steps=args.eval_steps,
    )

    # Keep compatibility across TRL versions.
    trainer = None
    init_errors: list[str] = []
    init_variants = [
        {"tokenizer": tokenizer},
        {"processing_class": tokenizer},
        {},
    ]
    for variant in init_variants:
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
            "Failed to initialize SFTTrainer with available signatures. "
            f"Errors: {init_errors}"
        )

    # ----
    # Debugging helpers for "loss=0" / "grad_norm=nan".
    # These do not change training behavior; they just print state.
    # ----
    if args.debug:
        # 1) Confirm LoRA attached trainable parameters
        trainable_params = 0
        total_params = 0
        for _, p in model.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print(f"Model params total: {total_params:,}")
        print(f"Model params trainable: {trainable_params:,}")
        if trainable_params == 0:
            raise RuntimeError(
                "No trainable parameters found. LoRA may not have attached. "
                "Try adjusting LoraConfig.target_modules for this model."
            )

        # 2) Confirm dataset produces non-masked labels
        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        if "labels" in batch:
            labels0 = batch["labels"][0]
            masked = int((labels0 == -100).sum().item())
            total = int(labels0.numel())
            print(f"labels tokens: {total}, masked: {masked}, unmasked: {total - masked}")
            if total - masked == 0:
                raise RuntimeError(
                    "All labels are masked (-100). This usually causes loss=0. "
                    "Your data/collator is producing no supervised tokens."
                )
        else:
            print("WARNING: batch has no 'labels' key; trainer may be misconfigured.")

    print("Starting SFT training...")
    train_result = trainer.train()
    print(train_result)

    if eval_dataset is not None:
        print("Final evaluation...")
        metrics = trainer.evaluate()
        print(metrics)

    print("Saving final checkpoint...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing to Hub: {effective_hub_model_id}")
        trainer.push_to_hub()

    print("Done.")
    if torch.cuda.is_available():
        print(f"GPU used: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
    print(f"HF_TOKEN present: {'HF_TOKEN' in os.environ}")


if __name__ == "__main__":
    main()
