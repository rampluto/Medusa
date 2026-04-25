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
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset("json", data_files=args.dataset)["train"]

    print(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    print("Formatting chat samples...")
    ds = ds.map(lambda row: build_text(row, tokenizer), remove_columns=ds.column_names)

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
        fp16=torch.cuda.is_available(),
        bf16=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=effective_hub_model_id if effective_hub_model_id else None,
        dataset_text_field="text",
        max_length=args.max_seq_len,
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
                train_dataset=ds,
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

    print("Starting SFT training...")
    train_result = trainer.train()
    print(train_result)

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
