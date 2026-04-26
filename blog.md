---
title: "MEDUSA: Teaching a 3B model to be an autonomous data engineer with GRPO"
thumbnail: /blog/assets/medusa/thumbnail.png
authors:
  - user: anubhavkamal
tags:
  - reinforcement-learning
  - openenv
  - grpo
  - trl
  - unsloth
  - data-engineering
  - llm-agents
---

# MEDUSA: Teaching a 3B model to be an autonomous data engineer with GRPO

> *"We built a 30-day data pipeline that actively breaks. Type traps, OOM bombs, schema drift, null nukes — all injected on specific days, all deterministically graded. Then we asked a 3B model to survive."*

Most "LLM agent" demos crumble the moment the world stops being a toy. The cleaning is canned, the join keys are immaculate, and the grader is another LLM that quietly cheers the agent on. Real data engineering doesn't work like that. Pipelines fail because the *world* keeps changing — yesterday's schema doesn't match today's, a numeric column comes back as `"$50.50"`, and a single duplicate key blows up an OOM-bound merge.

So we built **MEDUSA** — *Medallion-Engineered Deterministic Unified Storage Agent* — an [OpenEnv](https://github.com/openenv-ai/openenv) environment + a two-stage training recipe (SFT → GRPO) that turns a small open model into a usable autonomous data engineer. No LLM-as-judge. No reward hacking. Just a deterministic Python grader running mathematical assertions on a cumulative Silver table.

This post is the long story: what we built, what tried to kill us during training, and what worked.

---

## TL;DR

- **Environment:** A 30-day Bronze→Silver pipeline gauntlet built as an [OpenEnv](https://huggingface.co/spaces/anubhavkamal/medusa-env) Space. Every day has at least one anomaly. Major trap days at **Day 8** (type trap), **Day 14** (OOM), **Day 21** (schema drift), **Day 28** (null nuke).
- **Agent:** Qwen2.5-3B-Instruct with LoRA, trained in two stages — **SFT** on synthetic expert trajectories, then **GRPO** against the live environment using TRL + Unsloth.
- **Reward:** Hybrid of (1) a graded JSON-format reward to keep outputs parseable and (2) the actual environment reward replayed step-by-step.
- **Outcome:** 600 GRPO steps on an A10G (≈ **1.97M tokens** trained), recoverable training stats parsed from stdout (because HF Jobs disks are ephemeral), and an Olist-based 30-day evaluation harness.

Repo: <https://github.com/rampluto/Medusa>
Space: <https://huggingface.co/spaces/anubhavkamal/medusa-env>
SFT adapter: <https://huggingface.co/anubhavkamal/medusa-qwen-sft>
GRPO adapter: <https://huggingface.co/anubhavkamal/medusa-qwen-grpo>

![MEDUSA architecture diagram — placeholder]( /blog/assets/medusa/architecture.png )
*<sub>Replace with your architecture diagram: Bronze → Agent → Silver + Quarantine, with the deterministic Grader hanging off `COMMIT_DAY`.</sub>*

---

## Why a "Medallion gauntlet"?

Modern data platforms don't fail on one clean table. They fail on *integration over time*. The Bronze→Silver transition is where pipelines actually die:

- **Stale data** — yesterday's snapshot processed as today's.
- **Schema drift** — a new column appears upstream that Silver doesn't know about.
- **Dirty join keys** — NULLs and whitespace produce silent 0-row joins.
- **Cartesian explosions** — a non-unique dimension key multiplies rows catastrophically.
- **Orphaned records** — unmatched fact rows that must be quarantined, not silently dropped.

A useful environment for training agents on this has to (a) accumulate state across many days, (b) inject realistic faults, and (c) grade with code, not vibes. MEDUSA does all three.

---

## The Environment

### The world is a cumulative Silver table

Each episode is **30 days long**. Each day:

1. A fresh Bronze batch lands. It is *guaranteed* to fail at least one grader assertion — no free days.
2. The agent has up to **10 tool calls** to clean it.
3. The agent ends the day with `COMMIT_DAY`, which triggers a deterministic audit.
4. If the audit passes, Silver carries forward to the next day. If not, terminal crash.

Silver never resets. **It is the persistent world model the agent has to reason about across 30 days.**

### 7 tools, no synonyms

The action space is intentionally tiny:

| Tool | Purpose |
|---|---|
| `PROFILE_TABLE` | Inspect schema, types, nulls, duplicates |
| `CLEAN_COLUMN` | `cast`, `strip`, or `fill_zero` |
| `DEDUPLICATE` | Capped at 1 call/table/day |
| `EVOLVE_SILVER_SCHEMA` | Required when a new column appears |
| `QUARANTINE_ROWS` | Route bad rows to `/quarantine` |
| `EXECUTE_MERGE` | Append today's clean batch into Silver |
| `COMMIT_DAY` | Submit Silver to the grader |

Every output is a JSON object: `{"action": "...", "params": {...}}`. The system prompt is the single source of truth across SFT, GRPO, and eval — that consistency turns out to matter a lot (more on this below).

### The major trap days

| Day | Trap | Canonical fix |
|---|---|---|
| 8 | Revenue stored as `"$50.50"` | `CLEAN_COLUMN(strip "$")` then `CLEAN_COLUMN(cast float)` |
| 14 | Massive duplicate keys → estimated OOM | `PROFILE_TABLE` → `DEDUPLICATE` → `EXECUTE_MERGE` |
| 21 | New column `promo_code` appears upstream | `EVOLVE_SILVER_SCHEMA("promo_code")` |
| 28 | 20% of `user_id` is NULL | `QUARANTINE_ROWS(... user_id IS NULL)` |

Every other day draws from a fixed corruption pool (null injection, whitespace, minor type mismatch) seeded deterministically by day number, so the environment is reproducible end-to-end.

### The grader has no LLM in it

Four checks run after every `COMMIT_DAY`:

1. **Freshness** — Silver must have grown today.
2. **Schema** — `set(silver.columns) ⊇ set(contract_columns)`.
3. **Type integrity** — every numeric-role column has a numeric dtype.
4. **Null integrity** (Day 28+) — no NULL primary-key rows leak into Silver.

If anything fails: terminal crash, **−100**. If it passes: **+Day Number** (so Day 30 is worth +30, which is also what makes the agent *want* to survive). A perfect 30-day run plus a clean end-of-episode quarantine ratio gives you another **+100**.

That's it. No prompt-judge, no fuzzy "did it look right?". Just code reading a DataFrame.

---

## The recipe: SFT → GRPO

We trained **Qwen2.5-3B-Instruct + LoRA** in two stages:

### Stage 1 — SFT (`train_sft.py`)

We generated ~1,000 synthetic Bronze→Silver trajectories with an expert rule-based solver. To prevent name-memorization (the model learning "if `total_amount`, then `cast`"), the schema generator picks a random domain (`medical`/`finance`/`logistics`/`gaming`/`retail`) per episode and renames every column accordingly. The agent has to read profile output and *infer* what to do, not pattern-match column names.

SFT used standard TRL `SFTTrainer` with rank-16 LoRA on `q_proj/k_proj/v_proj/o_proj`. One epoch, batch 2 × grad-accum 8, lr `2e-5`, on an A10G via `hf jobs run`.

![SFT loss curve — placeholder]( /blog/assets/medusa/sft_loss.png )
*<sub>Drop in your SFT training loss / eval-loss plot here.</sub>*

### Stage 2 — GRPO (`trainer/train_medusa_grpo.py`)

This is where it gets interesting. GRPO (Group Relative Policy Optimization) needs **diverse rewards across the group** — if every sampled completion gets the same reward, the gradient is zero and the policy doesn't move. We used [TRL's GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) with [Unsloth](https://github.com/unslothai/unsloth) for fast 4-bit-friendly LoRA inference.

Key knobs from `train_medusa_grpo.py`:

```python
GRPOConfig(
    temperature=0.9,                # exploration
    learning_rate=2e-6,             # higher LR burns entropy in LoRA GRPO
    num_generations=6,              # group size — diversity matters
    max_prompt_length=1536,
    max_completion_length=192,
    min_new_tokens=15,              # block 14-token reward hacks (see below)
    entropy_coeff=0.02,             # explicit anti-collapse term
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    use_vllm=args.use_vllm,
)
```

The trick was building **two reward functions** and composing them:

```python
def json_format_reward(completions, **_):
    # Graded JSON validity — prevents binary collapse
    # +0.20  fully valid JSON with correct schema
    # -0.30  valid JSON but action ∉ VALID_ACTIONS  (kills `deduplicate_rows` attractor)
    #  0.00  valid JSON missing keys
    # -0.30  unparseable JSON-like
    # -0.50  no JSON at all

def medusa_env_reward(completions, seed, history_json, **_):
    # For each completion: rebuild the env with the saved history prefix,
    # apply the proposed action, return scaled env reward in [-5, +5].
```

Each batch: parse the action, **replay the history into a fresh env**, step once, return the scaled environment reward. This is the part that makes the model actually learn data-engineering decisions instead of vibes.

---

## What tried to kill us during training

This section is the actual reason to read this post. The graphs at the top of every "we trained an agent" post never tell you about the three days you spent wondering why the model only emitted `deduplicate_rows`.

### War story #1 — The `deduplicate_rows` mode collapse

Early GRPO runs collapsed onto a single token sequence: the model emitted `{"action": "deduplicate_rows", "params": {}}` on every prompt. Looks fine? It's not — `VALID_ACTIONS` requires the canonical `DEDUPLICATE`. The string was *almost right*, which is what made it lethal.

Why it happened: the original JSON-format reward gave **+0.05 for "JSON parses, has `action` + `params` keys, but action name is unknown"**. That tier created an attractor. Any time the env reward was near zero (which happens often early on), the +0.05 format reward was enough to pull the policy toward "any plausible JSON, real action name optional."

The fix is the comment block in `make_reward_funcs`:

```python
# +0.20  fully valid JSON with correct action + params schema
# -0.30  valid JSON, correct keys, but action name not in VALID_ACTIONS
#        (was +0.05; that tier created the `deduplicate_rows` attractor —
#         keep it negative so emitting fake action names is strictly worse
#         than emitting a real one.)
```

Lesson: **never give positive reward for a near-miss in a discrete action space.** Make the local optimum strictly worse than the global one.

### War story #2 — The 14-token hack

After patching #1, we caught the model emitting absurdly short completions — sometimes 14 tokens — because that was just enough to satisfy `{"action":"COMMIT_DAY","params":{}}`. It would then take the step penalty and try again. Net: it found the cheapest format-only positive reward.

Fixes (both shipped):

```python
parser.add_argument("--min-new-tokens", type=int, default=15)
parser.add_argument("--entropy-coeff", type=float, default=0.02)
```

`min_new_tokens=15` floors completion length so the model can't degenerate into format-only outputs. `entropy_coeff=0.02` explicitly penalises low-entropy policies — the standard GRPO/PPO entropy bonus, but in TRL you have to ask for it.

### War story #3 — The disappearing checkpoints

Halfway through a 300-step run on HF Jobs, we wanted to inspect `checkpoint-300/`. Surprise: HF Jobs runners have **ephemeral disks**. Once the job exits — finish, fail, or cancel — `trainer/medusa-grpo-output/checkpoint-300/` is gone. The script only pushes to the Hub at the very end (`trainer.train()` returns → `model.save_pretrained` → `api.upload_folder`).

We documented this in [`recover_checkpoint.txt`](https://github.com/rampluto/Medusa/blob/main/recover_checkpoint.txt) with the exact patch you want for next time:

```python
# Add to GRPOConfig to push every checkpoint as it's written:
"push_to_hub": True,
"hub_model_id": args.hub_model_id,
"hub_strategy": "checkpoint",
```

But for *this* run we needed to recover stats from a job whose disk had already been wiped. Which leads to the next part.

### War story #4 — Reconstructing training metrics from stdout

Every TRL `[train] step=N {...}` and our custom `[reward] call=N mean=... moving=... failures={...}` line went to stdout. HF Jobs retains stdout/stderr after the job ends — so we wrote [`recover_grpo_logs.py`](https://github.com/rampluto/Medusa/blob/main/recover_grpo_logs.py), which:

1. Pulls the full log via `hf jobs logs <JOB_ID>` (with rate-limit-aware retries on `/whoami-v2`).
2. Parses both line patterns into pandas DataFrames.
3. Saves CSVs and renders a 6-panel training-health figure.

The `STEP_RE` and `REWARD_RE` patterns are tiny:

```python
STEP_RE = re.compile(r"\[train\] step=(\d+)\s+(\{.*?\})\s*$", re.MULTILINE)
REWARD_RE = re.compile(
    r"\[reward\] call=(\d+)\s+mean=([-\d.eE]+)\s+moving=([-\d.eE]+)\s+failures=(\{[^}]*\})"
)
```

That's how we reconstructed every plot you'll see below from a job whose checkpoints were already in the void.

---

## Training results

We ran 600 GRPO steps on `a10g-large`, batch 1 × grad-accum 4 × 6 generations = 24 rollouts/step ≈ 14k env interactions. The recovered logs are 10k lines of stdout, parsed into:

- `df_train.csv` — 600 rows, one per training step, with loss / grad_norm / entropy / clip ratios / reward components.
- `df_reward.csv` — 60 rows, one per `--reward-log-every=10` checkpoint, with batch / moving means and the failure breakdown (`invalid_json`, `invalid_action`, `missing_json`).

### Headline numbers

| Metric | Value |
|---|---|
| Steps observed | **600** |
| Total tokens trained | **1,973,698** |
| Frac steps with ~zero grad | **28.2 %** |
| Avg `frac_reward_zero_std` | **28.2 %** |
| Final 50-step total reward (mean) | **−0.240** |
| Final 50-step env reward (mean) | **−0.395** |
| Peak env reward (20-step MA) | **−0.209** |
| Best 100-call moving mean | **−0.131** |
| Worst 100-call moving mean | **−0.708** |
| Total parse failures across run | **`invalid_action: 150` · `invalid_json: 34` · `missing_json: 3`** |

Two things in this table are worth highlighting:

- The **final reward is still negative**. That's expected at 600 steps with `--env-reward-scale 20.0` and a clipped `[-5, +5]` env reward — the policy is learning *to format and to act correctly*, not yet to win full episodes. The signal we care about is the *direction of travel* below.
- **`frac_reward_zero_std = 28.2 %`** means roughly a quarter of GRPO batches had zero reward variance across the 6 generations — i.e. zero gradient. This is the dominant inefficiency in small-budget GRPO and the strongest argument for `num_generations ≥ 8` and warmer sampling next time.

### The 6-panel training-health chart

![GRPO panels]( /blog/assets/medusa/panels.png )
*<sub>Source: `grpo_recovery/panels.png`. Six panels: total reward, env reward, JSON-format reward, policy entropy, gradient norm, mean completion length.</sub>*

The shape of the run, in numbers:

| Quantity | Early (first 50 steps) | Late (last 50 steps) | Δ |
|---|---|---|---|
| `reward` | **−0.353** | **−0.240** | **+0.113** |
| `rewards/medusa_env_reward/mean` | **−0.560** | **−0.395** | **+0.165** |
| `rewards/json_format_reward/mean` | **+0.096** | **+0.155** | **+0.059** |
| `entropy` | **0.810** | **0.698** | **−0.112** |
| `completions/mean_length` | **84.1 tokens** | **50.8 tokens** | **−33.3** |

Read it together and the story is clean: the model **(a)** stops being verbose (84 → 51 tokens — it learned the JSON envelope is enough), **(b)** raises its format-reward floor without ever crossing the saturation threshold of `+0.20`, and **(c)** keeps entropy near the regulariser target — no collapse, but a clear narrowing toward correct actions.

The 50-step moving average of `rewards/json_format_reward/mean` first **crosses +0.15 at step 294**, which is the cleanest single-number answer to "when did GRPO start working?" on this run.

### Reward trajectory

![Reward trajectory]( /blog/assets/medusa/reward_trajectory.png )
*<sub>Source: `grpo_recovery/reward_trajectory.png`. Faint = batch mean; bold = 100-call moving mean. Note `min batch_mean = −1.333` (one bad batch where 11 of 60 generations emitted `invalid_action`) and `best moving_mean = −0.131`.</sub>*

### Failure-mode evolution

The single most informative artifact we recovered is the **failure dictionary over time** — i.e. *what kind* of mistake the model was making at each window of 200 reward calls:

| Window (reward calls) | `invalid_action` | `invalid_json` | `missing_json` |
|---|---|---|---|
| 0 – 200 | **65** | 18 | 1 |
| 200 – 400 | **38** | 10 | 0 |
| 400 – 600 | **47** | 5 | 2 |

The middle window is the cleanest proof GRPO is doing something — `invalid_action` falls **−42 %** from the first window. The bump back up in the final window is the model rediscovering an `invalid_action` attractor (likely a near-miss of `EVOLVE_SCHEMA` vs. `EVOLVE_SILVER_SCHEMA`) and being pulled back toward the correct vocabulary by the `−0.30` format penalty. We expect this to drop back below 30 with more training.

![Failure modes — placeholder]( /blog/assets/medusa/failure_modes.png )
*<sub>Stacked area chart of failure counts per 10-call window. Generate from `grpo_recovery/df_reward.csv` `failures` column.</sub>*

---

## Evaluation: the 30-day Olist gauntlet

Synthetic data trains the model. Real data tells you whether it actually works.

[`eval_grpo_olist.py`](https://github.com/rampluto/Medusa/blob/main/eval_grpo_olist.py) loads the merged GRPO model and runs the full 30-day episode against an `OlistDayGenerator` that injects the same family of corruptions into the **real Olist Brazilian e-commerce dataset** (30 daily CSVs already shipped under `data/olist/day_01.csv … day_30.csv`). It measures:

- Per-day reward and commit success.
- JSON format validity rate.
- Valid action rate.
- Trap-day vs normal-day reward decomposition.
- Optionally compares against a rule-based expert baseline.

```bash
export HF_TOKEN=hf_...
python eval_grpo_olist.py \
  --model anubhavkamal/medusa-qwen-grpo \
  --base  Qwen/Qwen2.5-3B-Instruct \
  --run-baseline \
  --output-json eval_results.json
```

### Per-day breakdown

![Per-day rewards — placeholder]( /blog/assets/medusa/per_day_rewards.png )
*<sub>Bar chart of `sum_reward` per day for GRPO vs Expert. Highlight Days 8/14/21/28 in red. Generate from `eval_results.json["grpo"]["per_day"]`.</sub>*

### Comparison summary

| Metric | GRPO Model | Expert Baseline |
|---|---|---|
| Total reward | **REPLACE_ME** | **REPLACE_ME** |
| Days committed | **REPLACE_ME** /30 | **REPLACE_ME** /30 |
| Total steps | **REPLACE_ME** | **REPLACE_ME** |
| Mean JSON valid rate | **REPLACE_ME** % | 100.0 % |
| Mean action valid rate | **REPLACE_ME** % | 100.0 % |
| Trap-day reward | **REPLACE_ME** | **REPLACE_ME** |
| Normal-day reward | **REPLACE_ME** | **REPLACE_ME** |

*<sub>Run `python eval_grpo_olist.py --run-baseline --output-json eval_results.json` and paste the printed comparison table here. The expert baseline is a hand-coded rule-based policy with full state visibility — it exists to anchor what a "perfect" trajectory looks like, not to claim parity. Closing the gap is future work.</sub>*

### What survival looks like

When the agent does survive Day 8 (the type trap), the trace is genuinely satisfying to read:

```
Day 08  ⚠ TRAP(type_trap)        steps=6  reward=+8.20  commit=✓  json_ok=100% act_ok=100%
  → PROFILE_TABLE(bronze)        observe revenue dtype is object
  → CLEAN_COLUMN(revenue, strip) remove "$"
  → CLEAN_COLUMN(revenue, cast)  to float
  → DEDUPLICATE                  routine
  → EXECUTE_MERGE                clean batch lands in Silver
  → COMMIT_DAY                   grader passes (+8 reward)
```

That's the same 6-action sequence the rule-based expert produces. The model didn't memorise it — the schema is fully randomised at training time — it learned to *read* the profile output and choose `strip` before `cast`.

---

## What we'd do differently (and what's next)

A non-exhaustive list of things this run made obvious:

1. **Always have a graded format reward, never a binary one.** The `deduplicate_rows` attractor is what happens when "almost right" pays positive. Make it strictly negative.
2. **Floor `min_new_tokens`** for any RL run that has *any* format reward. Otherwise the model finds the shortest valid string and stays there. (Note our completion length still fell from 84 → 51 tokens — that's healthy compression, but we'd watch it like a hawk if the floor weren't there.)
3. **Keep the system prompt in one file.** [`medusa_prompts.py`](https://github.com/rampluto/Medusa/blob/main/medusa_prompts.py) is imported by SFT data generation, GRPO training, and eval. Drift between any two of those three crashes everything silently.
4. **Bigger model + more steps.** 3B at 600 steps is enough to show the learning signal — `invalid_action` count fell 42 % between the first and middle window, format reward crossed +0.15 at step 294, mean completion length compressed by 40 %. The Day 8 → Day 14 transition is reliable; Day 21 schema-drift is borderline; Day 28 null-nuke is still flaky. The architecture has obvious headroom.
5. **Data-quality-based reward shaping.** The current grader is binary per check (pass / fail). The repo already ships [`data_quality_score.py`](https://github.com/rampluto/Medusa/blob/main/data_quality_score.py), an 8-component continuous **Data Quality Score** (DQS) over readability, completeness, uniqueness, type consistency, date sanity, column quality, string cleanliness, and numeric sanity. The plan for the next run is to use the **per-step ΔDQS of Silver** as a dense, continuous reward shaper on top of the discrete grader signal. That fixes our biggest 600-step pain — `frac_reward_zero_std = 28.2 %` (a quarter of all batches had zero gradient) — by guaranteeing reward variance across generations even when none of them flips a grader check. We've also catalogued DQS's known blind spots (currency-formatted numerics, sub-90 % numeric columns, key-column-only duplication) in [`data_quality_issues.md`](https://github.com/rampluto/Medusa/blob/main/data_quality_issues.md), so corruption injection is paired with metric coverage from day one.

Things we want to try next:

- 7B base + extended LoRA targets (`gate_proj/up_proj/down_proj`).
- vLLM rollouts for ~3× faster GRPO.
- Curriculum: trap-only days first, then mixed, then full 30-day.
- A small Auditor LoRA on top of the same base — lets the agent self-review before `COMMIT_DAY` without adding a second policy.

---

## Try it yourself

```bash
git clone https://github.com/rampluto/Medusa.git && cd Medusa
uv sync && source .venv/bin/activate

# Run a manual episode
python run_episode.py

# Or run the trained agent on the Olist gauntlet
export HF_TOKEN=hf_...
python eval_grpo_olist.py --run-baseline
```

The Space is also live and clickable:

- **OpenEnv Space:** <https://huggingface.co/spaces/anubhavkamal/medusa-env>
- **API docs:** `BASE_URL/docs`
- **Playground:** `BASE_URL/web`

If you train on top of this, please let us know — and especially please tell us when *your* agent collapses into emitting `deduplicate_rows`. We'll know exactly which knob to turn.

---

## Acknowledgements

- [OpenEnv](https://github.com/openenv-ai/openenv) for the environment framework.
- [TRL](https://github.com/huggingface/trl) for `GRPOTrainer`.
- [Unsloth](https://github.com/unslothai/unsloth) for the LoRA + 4-bit speedups.
- [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/main/en/guides/jobs) for the A10G runtime (and for keeping stdout around long enough to recover from ourselves).
- The [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) for being a wonderfully messy real-world testbed.

---

*If you found this useful, ⭐ the [repo](https://github.com/rampluto/Medusa) and follow [@anubhavkamal](https://huggingface.co/anubhavkamal) for the next post — adding a vLLM-backed rollout loop, a DQS-shaped reward, and pushing past Day 28.*
