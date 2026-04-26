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

## Deploy the Studio UI + GRPO on Hugging Face Spaces

The **repo root `Dockerfile`** is the image HF Spaces build from. It is
configured for:

- **Docker** Space with a **GPU** (T4 or better; T4: set
  `MEDUSA_GRPO_LOAD_IN_4BIT=1` in Space variables if you OOM).
- **Default adapter:** `anubhavkamal/medusa-qwen-grpo` via `GRPO_MODEL_ID`
  (same as `eval_grpo_olist.py` GRPO default).
- **Base model:** `Qwen/Qwen2.5-3B-Instruct` via `BASE_MODEL_ID`.
- **Predictor entrypoint:** `MEDUSA_GRPO_PREDICTOR=trainer.grpo_predictor_hub:predict`.

**HF_TOKEN (when you need it)**

- **Not required** for a public adapter + public base, if the Hub download
  succeeds without auth.
- **Add as a Space *Secret* named `HF_TOKEN`** when:
  - the base model is **gated** (you must accept the license on the model
    card, then a token is needed for the container to download), or
  - **`anubhavkamal/medusa-qwen-grpo` is private**, or
  - downloads return **401 / 403** in the Space **Logs**.

**After the build goes green:** open
`https://<your-space>.hf.space/medusa/studio`, select agent **GRPO
Trained**, and run **Auto-run**. The first GRPO action may take a long
time (model download + load).

**Canonical image:** the repo **root** `Dockerfile` (also visible as
`hf_space/Dockerfile` — a **symlink** to the same file; see
[`hf_space/README.md`](../hf_space/README.md)). The file
`trainer/Dockerfile.hub_predictor` is an older **reference** only.

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

---

## Serving the GRPO agent on an HF Space

The `grpo_trained` UI agent is wired through
`server.agent_policies.GrpoTrainedPolicy`, which loads a callable named by
the env var `MEDUSA_GRPO_PREDICTOR` (format `module:function`). Without
this var the UI returns:

> `grpo_trained agent requires MEDUSA_GRPO_PREDICTOR='module:function' to be configured.`

`trainer/grpo_predictor_hub.py` is a drop-in predictor that loads your
**Hub-hosted base + LoRA adapter** in-process (mirrors `eval_grpo_olist.py`
exactly: same SYSTEM_PROMPT, same prompt rendering, same action parser,
same fallback to `PROFILE_TABLE bronze` on parse failure). Heavy modules
(`torch`, `transformers`, `peft`) are imported lazily and the
`(model, tokenizer)` bundle is cached behind a thread-safe lock so the
first request triggers a single download/load and every subsequent call
reuses it.

### 1. HF Space Variables and Secrets

Open the Space → Settings → **Variables and secrets**. Add:

| Type     | Key                       | Value (example)                  | Notes |
|----------|---------------------------|----------------------------------|-------|
| Variable | `MEDUSA_GRPO_PREDICTOR`   | `trainer.grpo_predictor_hub:predict` | The only one that MUST be set. |
| Variable | `BASE_MODEL_ID`           | `Qwen/Qwen2.5-3B-Instruct`       | Match `train_sft_olist.py` / `train_medusa_grpo.py` / `eval_grpo_olist.py`. |
| Variable | `GRPO_MODEL_ID`           | `myuser/medusa-qwen-grpo`        | Your Hub repo holding the LoRA adapter. |
| Secret   | `HF_TOKEN`                | `hf_…`                           | Required if the base model or your adapter is gated/private. Use **Secret**, not Variable, so it's masked in logs. |

Optional knobs (sensible defaults baked into the Dockerfile reference):

| Type     | Key                            | Default   | Effect |
|----------|--------------------------------|-----------|--------|
| Variable | `MEDUSA_GRPO_LOAD_IN_4BIT`     | `0`       | `1` → nf4 quantization (saves ~5 GB VRAM, slightly slower). |
| Variable | `MEDUSA_GRPO_NO_MERGE`         | `0`       | `1` → keep the LoRA adapter wrapped instead of `merge_and_unload()`. |
| Variable | `MEDUSA_GRPO_MAX_NEW_TOKENS`   | `192`     | Per-step generation budget. |
| Variable | `MEDUSA_GRPO_TEMPERATURE`      | `0.0`     | `0` = greedy (default); raise slightly for diversity. |

### 2. GPU hardware

`Settings → Hardware`:

| SKU       | Works? | Notes |
|-----------|--------|-------|
| CPU basic | No     | `transformers` will load but a single forward pass takes minutes. |
| T4 small  | Yes    | Set `MEDUSA_GRPO_LOAD_IN_4BIT=1` (Qwen2.5-3B in bf16 won't fit). |
| A10G small / L4 | Yes | Comfortable in bf16 with `merge_and_unload`. |

### 3. Image dependencies

The repo-root `Dockerfile` does not install `torch`/`transformers`/`peft`/
`accelerate`/`bitsandbytes`. The Hub predictor needs all five.
`trainer/Dockerfile.hub_predictor` is a **reference image** that layers
those wheels on top of the existing server stack and pins a CUDA-aware
torch build. Two ways to use it:

* **Hand-merge** the relevant `RUN uv pip install …` lines and the
  `ENV` block into the root `Dockerfile`. HF Spaces only reads
  `Dockerfile` at the repo root, so this is the simplest path.
* **Deploy a sibling Space** from a branch where `Dockerfile.hub_predictor`
  is renamed to `Dockerfile`.

### 4. Cold-start warmup (optional)

The first call into `predict()` triggers the model load (~30–60 s for
Qwen2.5-3B from the Hub) — the user clicking "Auto-run" the very first
time will wait that long. To pre-pay the cost at server startup, call
`trainer.grpo_predictor_hub.warmup()` from a FastAPI startup hook, or
ping `/run/auto-run` with the GRPO agent right after deploy.

### 5. Troubleshooting

| Symptom | Cause |
|--------|-------|
| `grpo_trained agent requires MEDUSA_GRPO_PREDICTOR=...` | Variable not set, or set to a non-`module:function` string. |
| `Configured GRPO predictor 'foo:bar' is missing or not callable.` | Module imports OK but the named attribute doesn't exist or isn't callable. Verify the symbol with `python -c "import foo; print(foo.bar)"`. |
| `GRPO_MODEL_ID is not set` (raised by the predictor) | Add the Variable; it's the Hub repo id of your LoRA adapter. |
| `OutOfMemoryError` during first call | Switch to `MEDUSA_GRPO_LOAD_IN_4BIT=1`, or upgrade hardware to A10G. |
| Model loads on every request (slow) | The worker is being recycled. Check Space logs for OOM-killer / health-check failures, and consider warmup at startup. |
| 4xx with `unknown action` | Adapter is producing a non-VALID_ACTIONS name. Predictor falls back to `PROFILE_TABLE bronze`; check Space logs for the raw output and re-train if persistent. |
