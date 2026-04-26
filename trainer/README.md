# Trainer scripts

End-to-end training entry points for the MEDUSA pipeline.

| Script | Stage | What it does |
| --- | --- | --- |
| `generate_sft_dataset_olist.py` | data | Builds an SFT JSONL on the **real Olist 30-day gauntlet** using `OlistDayGenerator` + the rule-based expert solver. |
| `train_sft_olist.py` | SFT | Trains a Qwen2.5-3B-Instruct LoRA adapter on that JSONL. Auto-builds the dataset if missing. |
| `train_medusa_grpo.py` | GRPO | Continues from the SFT adapter (or base model) with TRL GRPO + Unsloth. |

All three are configured to share the **same base model and LoRA target
modules** (`Qwen/Qwen2.5-3B-Instruct`, `q_proj/k_proj/v_proj/o_proj`), so
adapters produced by one stage drop straight into the next.

## SFT on Olist (the polished-demo recipe)

The Olist generator + the rule-based expert solver collectively produce a
trajectory that is **byte-identical** to `eval_grpo_olist.py --run-baseline`.
SFT'ing on this data therefore makes:

> SFT distribution = eval distribution = expert-perfect

so the trained model should hit ~100% commit rate, ~100% JSON-valid rate,
and total reward indistinguishable from the rule-based baseline.

### One-shot run

```bash
python trainer/train_sft_olist.py \
    --output-dir trainer/medusa-sft-olist-output
```

This will:
1. Build `trainer/sft_dataset_olist.jsonl` if it doesn't exist (1 episode
   × 4 paraphrase passes ≈ 230 rows of expert demonstrations).
2. Train a LoRA adapter on `Qwen/Qwen2.5-3B-Instruct` for 2 epochs.
3. Save the adapter to `trainer/medusa-sft-olist-output/`.

Push to the Hub for `eval_grpo_olist.py --model`:

```bash
HF_TOKEN=hf_... python trainer/train_sft_olist.py \
    --push-to-hub --hub-model-id myuser/medusa-qwen-sft-olist
```

### Step-by-step run

```bash
# Build dataset only (e.g. to inspect rows).
python trainer/generate_sft_dataset_olist.py \
    --episodes 1 --paraphrase-passes 4 \
    --out trainer/sft_dataset_olist.jsonl

# Train against an existing JSONL.
python trainer/train_sft_olist.py \
    --dataset trainer/sft_dataset_olist.jsonl \
    --output-dir trainer/medusa-sft-olist-output
```

### Recipe knobs

* `--paraphrase-passes N` — emit `N` `<think>` phrasings per (prompt,
  action) transition. The action target is unchanged — cheap data
  multiplier without changing the action distribution. Default `4`.
* `--with-mistakes` — re-enable the 10% random "wrong action" branch in
  the SFT data. Use this if you want the model to also learn recovery on
  Olist; leave it off (default) for the polished demo.
* `--episodes N` — Olist episodes are deterministic, so this is a no-op
  unless `--with-mistakes` is on (different episodes get different
  wrong-action RNG).
* `--rows-per-day` — kept at `100` to match `eval_grpo_olist.py`. Some
  role detections (negative-value detection) are sensitive to row count;
  do not lower without re-checking the heuristics.

## SFT → GRPO continuation

Once the SFT adapter is on the Hub (or on disk), feed it into the GRPO
trainer:

```bash
python trainer/train_medusa_grpo.py \
    --sft-adapter myuser/medusa-qwen-sft-olist \
    --hub-model-id  myuser/medusa-qwen-grpo \
    --push-to-hub \
    --max-steps 200
```

GRPO will resume from the SFT adapter weights instead of cold-starting
from the base model — significantly faster convergence and a much
stronger eval score on `eval_grpo_olist.py`.
