---
title: MEDUSA Environment
emoji: 🦑
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - data-engineering
app_port: 7860
---

# MEDUSA

**Medallion-Engineered Deterministic Unified Storage Agent**

An OpenEnv reinforcement learning environment that trains agents to act as *Relational Controllers* — orchestrating multi-source Bronze→Silver data integration pipelines inside a Medallion Architecture.

### Hugging Face Space (Docker)

The root **`Dockerfile`** is built by Spaces (`sdk: docker`, port **7860**). It includes the **GRPO Hub predictor** with defaults: `GRPO_MODEL_ID=anubhavkamal/medusa-qwen-grpo`, `BASE_MODEL_ID=Qwen/Qwen2.5-3B-Instruct`, `MEDUSA_GRPO_PREDICTOR=trainer.grpo_predictor_hub:predict`. In the Space, enable a **GPU** (T4 or better). After deploy, open **`/medusa/studio`**, select **GRPO Trained**, and run **Auto-run**. If downloads fail (gated model or private adapter), add **`HF_TOKEN`** as a Space *Secret* — see **`trainer/README.md`**. Deploy layout: **`hf_space/README.md`** (`hf_space/Dockerfile` is a symlink to the root `Dockerfile`).

---

## Problem

Modern data platforms fail not because they can't clean a single table, but because they can't reliably integrate **multiple shifting sources**. The Bronze→Silver transition is a minefield of:

- **Stale data** — processing yesterday's snapshot wastes compute and produces wrong results
- **Schema drift** — new columns appear in sources that Silver doesn't know about yet
- **Dirty join keys** — NULLs and whitespace cause 0-row joins and silent data loss
- **Cartesian explosions** — joining on non-unique Dimension keys multiplies rows catastrophically
- **Orphaned records** — unmatched Fact rows must be quarantined, not silently dropped

MEDUSA trains an agent to detect and handle all of these autonomously.

---

## Environment Overview

```
Bronze A (Fact)  ──┐
                   ├──► [Agent] ──► Silver  +  /quarantine
Bronze B (Dim)   ──┘
```

The agent observes data quality signals and selects ETL actions step-by-step. At the end it issues `COMMIT`, triggering a deterministic grader audit.

---

## The MDP

### Observation Space

A **16-element normalised float vector** `[0, 1]`:

| Index | Feature | Description |
|-------|---------|-------------|
| 0–1 | `time_delta_a/b_norm` | Source freshness (hours / 48h ceiling) |
| 2–3 | `is_stale_a/b` | Binary staleness flag |
| 4–5 | `null_ratio_key_a/b` | Fraction of null join keys |
| 6–7 | `uniqueness_a/b` | Key uniqueness ratio (1.0 = fully unique) |
| 8 | `match_rate` | % of Fact keys found in Dimension |
| 9–10 | `new_cols_a/b_norm` | Schema drift columns pending |
| 11 | `schema_compat` | Key type compatibility score |
| 12–14 | `did_prep_a/b`, `did_dedup_b` | Prerequisite action flags |
| 15 | `step_frac` | Episode progress (step / max_steps) |

### Action Space

11 discrete actions:

| Action | Description |
|--------|-------------|
| `SYNC_CHECK` | Verify freshness of both sources |
| `EVOLVE_SCHEMA` | Add new columns from A/B into Silver schema |
| `PREP_KEYS_A` | Cast, strip, null-fill join key in Source A |
| `PREP_KEYS_B` | Cast, strip, null-fill join key in Source B |
| `DEDUPLICATE_B` | Ensure Dimension (B) is unique on the join key |
| `EXECUTE_JOIN_INNER` | Inner join A ⋈ B |
| `EXECUTE_JOIN_LEFT` | Left join A ⋈ B (orphans → quarantine) |
| `EXECUTE_JOIN_ANTI` | Anti-join: extract rows in A with no match in B |
| `APPLY_SCD_1` | Overwrite Silver records (SCD Type 1) |
| `APPLY_SCD_2` | Close old records, insert new with timestamps (SCD Type 2) |
| `COMMIT` | Finalise pipeline; triggers grader audit |

### Reward Model

| Event | Reward | Trigger |
|-------|--------|---------|
| High-Match Join | **+25.0** | `match_rate > 90%` after join |
| Quarantine Precision | **+10.0** | Orphaned rows correctly isolated |
| Correct SCD-2 | **+5.0** | SCD-2 applied on a tracked column |
| Grader All-Pass Bonus | **+15.0** | All 4 post-commit checks pass |
| Row Explosion | **−100.0** | Join output > 105% of Fact row count |
| Join on Dirty Keys | **−30.0** | Join without PREP_KEYS → 0-row result |
| Stale Processing | **−15.0** | Action taken while source is stale, SYNC_CHECK never called |
| Step Penalty | **−0.2** | Applied every step (efficiency incentive) |

---

## Post-Commit Grader

After `COMMIT` the deterministic grader runs 4 checks:

| Check | Pass Condition |
|-------|---------------|
| **Volume** | `Silver rows ≤ Source A rows` (for left joins) |
| **Integrity** | Quarantine holds only true orphans (not keys that could have joined if cleaned) |
| **Schema** | Silver contains the union of all required columns from A and B |
| **History** | SCD-2 `valid_from`/`valid_to` timestamps are non-overlapping |

All 4 pass → **+15.0** bonus. Each failure costs **−5.0**.

---

## Episode Scenarios

Four canonical scenarios (selectable by seed):

| Seed | Scenario | Challenge |
|------|----------|-----------|
| 0 | `clean` | Fresh, unique keys, ~100% match rate. Baseline. |
| 1 | `dirty_keys` | NULLs + whitespace in join keys. Must PREP first. |
| 2 | `stale` | Source A is 8–24h old. Must SYNC_CHECK first. |
| 3 | `schema_drift` | New columns in A and B not yet in Silver. Must EVOLVE first. |

Random seeds produce blended variants.

---

## Setup

```bash
# Clone / navigate to repo
cd Medusa

# Create venv and install all deps (including pandas, numpy)
uv sync

# Activate
source .venv/bin/activate
```

---

## Running

### Start the FastAPI server

```bash
openenv validate
openenv build --tag openenv-medusa
docker run -p 8000:8000 openenv-medusa:latest
```

API docs available at `http://localhost:8000/docs`.
Playground available at `https://localhost:8000/web`
### Run tests

```bash
python -m pytest tests/envs/test_medusa_environment.py -v
# 53 passed in ~4s
```

### Run a manual episode (Python)

```python
from medusa_env.server import MedusaEnv
from medusa_env.models import MedusaActionType, MedusaAction

env = MedusaEnv(n_fact_rows=200, n_dim_rows=150)
obs = env.reset(seed=0)  # seed 0 = clean scenario
print(obs.message)

for action_type in [
    MedusaActionType.SYNC_CHECK,
    MedusaActionType.EVOLVE_SCHEMA,
    MedusaActionType.PREP_KEYS_A,
    MedusaActionType.PREP_KEYS_B,
    MedusaActionType.DEDUPLICATE_B,
    MedusaActionType.EXECUTE_JOIN_LEFT,
    MedusaActionType.APPLY_SCD_2,
    MedusaActionType.COMMIT,
]:
    obs = env.step(MedusaAction(action=action_type))
    print(f"{action_type.value:25s} reward={obs.reward:+.1f}  done={obs.done}")

print(f"\nGrader: {env.state.grader_report}")
```

### steps to push to hugging face
```bash
openenv push --repo-id <hf_username>/<hf_space>
```

Huggingface BASE_URL="https://<hf_username>-<hf_space>.hf.space"

---

## Architecture

```
envs/medusa_env/
├── __init__.py          # Package exports
├── models.py            # MedusaAction, MedusaObservation, MedusaState (Pydantic)
├── scenarios.py         # ScenarioGenerator — procedural Bronze A/B DataFrames
├── operators.py         # Stateless ETL functions (sync_check, prep_keys, execute_join, apply_scd …)
├── rewards.py           # RewardEngine — per-step reward computation
├── grader.py            # Grader — post-commit deterministic audit
├── openenv.yaml         # OpenEnv environment manifest
└── server/
    └── app.py           # FastAPI app via create_app()
    ├── medusa_env.py        # MedusaEnv — reset / step / commit loop

tests/envs/
└── test_medusa_environment.py   # 39 tests across 6 test classes
```

**Stack:** Python 3.10+ · Pandas · Pydantic v2 · FastAPI · OpenEnv

---

## Technical Notes

- **No external data required.** All Bronze tables are generated procedurally per episode.
- **No Spark or Delta Lake required.** All logic uses Pandas — identical semantics, zero cluster setup.
- The grader is fully deterministic: same Silver + quarantine tables always produce the same audit result.
- The governance log (accessible at `env._tables.governance_log`) records every agent decision with its reward and operator metrics.
