"""MEDUSA inference script — OpenEnv Hackathon submission.

Runs an LLM agent (via OpenAI-compatible API) against all three MEDUSA tasks
and reports per-task scores (0.0–1.0).

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM (OpenAI-compatible).
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key (used as the API key).

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="hf-..."
    python inference.py

Output:
    Prints per-task results and a final summary table to stdout.
    Exits with code 0 if all tasks score >= 0.35, else 1.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Validate required environment variables before anything else
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_missing = [k for k, v in {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME": MODEL_NAME,
    "HF_TOKEN": HF_TOKEN,
}.items() if not v]

if _missing:
    print(f"ERROR: Missing required environment variables: {', '.join(_missing)}", file=sys.stderr)
    print("Set them before running:", file=sys.stderr)
    for k in _missing:
        print(f"  export {k}=<value>", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# OpenAI client (uses API_BASE_URL + HF_TOKEN as the key)
# ---------------------------------------------------------------------------

from openai import OpenAI  # noqa: E402

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# MEDUSA environment imports
# ---------------------------------------------------------------------------

from pathlib import Path

# Dynamically add the OpenEnv repo root to sys.path so absolute imports work
# no matter where this script is executed from.
repo_root = str(Path(__file__).resolve().parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    # In-repo
    from envs.medusa_env import MedusaEnv
    from envs.medusa_env.models import MedusaAction, MedusaActionType
    from envs.medusa_env.tasks import TASKS, TaskResult, score_episode
except ImportError:
    # Standalone (running from inside envs/medusa_env/ installation)
    from medusa_env import MedusaEnv  # type: ignore
    from models import MedusaAction, MedusaActionType  # type: ignore
    from tasks import TASKS, TaskResult, score_episode  # type: ignore

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a data integration agent controlling a Bronze→Silver ETL pipeline.

You observe a 16-float feature vector describing data quality signals, and
you must choose one action per step from the list below.

ACTIONS (respond with ONLY the action name — nothing else):
  SYNC_CHECK          — Verify source freshness before processing
  EVOLVE_SCHEMA       — Add new columns from sources into Silver schema
  PREP_KEYS_A         — Clean and normalise join keys in Source A (Fact)
  PREP_KEYS_B         — Clean and normalise join keys in Source B (Dimension)
  DEDUPLICATE_B       — Remove duplicate keys from Source B
  EXECUTE_JOIN_INNER  — Inner join A ⋈ B
  EXECUTE_JOIN_LEFT   — Left join A ⋈ B (keeps all Fact rows; orphans → quarantine)
  EXECUTE_JOIN_ANTI   — Anti-join: extract Fact rows with no Dimension match
  APPLY_SCD_1         — Overwrite Silver records (SCD Type 1)
  APPLY_SCD_2         — Close old records and insert new with timestamps (SCD Type 2)
  COMMIT              — Finalise pipeline and trigger audit

STRATEGY:
1. Always call SYNC_CHECK first to verify freshness.
2. If schema drift signals are non-zero (features[9] or [10] > 0), call EVOLVE_SCHEMA.
3. If null key ratios (features[4] or [5] > 0), call PREP_KEYS_A and/or PREP_KEYS_B.
4. If Dimension uniqueness (features[7]) < 1.0, call DEDUPLICATE_B.
5. Prefer EXECUTE_JOIN_LEFT to preserve all Fact rows.
6. Prefer APPLY_SCD_2 for tracked history.
7. Call COMMIT when pipeline is complete.

The feature vector indices:
  [0]  time_delta_a_norm   [1]  time_delta_b_norm
  [2]  is_stale_a          [3]  is_stale_b
  [4]  null_ratio_key_a    [5]  null_ratio_key_b
  [6]  uniqueness_a        [7]  uniqueness_b
  [8]  match_rate          [9]  new_cols_a_norm
  [10] new_cols_b_norm     [11] schema_compat
  [12] did_prep_a          [13] did_prep_b
  [14] did_dedup_b         [15] step_frac
""").strip()

# ---------------------------------------------------------------------------
# LLM action chooser
# ---------------------------------------------------------------------------

VALID_ACTIONS = {a.value for a in MedusaActionType}


def choose_action(
    features: List[float],
    history: List[dict],
    step: int,
) -> str:
    """Ask the LLM to choose the next action given the current observation."""
    feature_str = ", ".join(f"{v:.3f}" for v in features)
    user_msg = (
        f"Step {step}. Feature vector: [{feature_str}]\n"
        "What is the single best next action? Respond with ONLY the action name."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include the last 4 steps of history for context (keep prompt short)
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=20,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip().upper().replace(" ", "_")

    # Fuzzy match: accept if the response contains a valid action name
    for action in VALID_ACTIONS:
        if action in raw:
            return action

    # Fallback: extract the longest matching token
    for action in sorted(VALID_ACTIONS, key=len, reverse=True):
        if action.replace("_", "") in raw.replace("_", ""):
            return action

    # Hard fallback: commit to end gracefully
    return MedusaActionType.COMMIT.value


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_id: str, max_steps: int = 15) -> TaskResult:
    """Run the LLM agent for one MEDUSA task. Returns the TaskResult."""
    task = TASKS[task_id]
    print(f"\n{'='*60}")
    print(f"TASK: {task.name} [{task.difficulty.upper()}]  (seed={task.seed})")
    print(f"  {task.description}")
    print(f"{'='*60}")

    env = MedusaEnv(n_fact_rows=200, n_dim_rows=150, max_steps=max_steps)
    obs = env.reset(seed=task.seed)

    history: List[dict] = []
    step = 0
    t0 = time.time()

    while not obs.done and step < max_steps:
        step += 1
        action_str = choose_action(obs.features, history, step)
        action_type = MedusaActionType(action_str)
        action = MedusaAction(action=action_type)

        obs = env.step(action)
        reward = obs.reward or 0.0

        print(f"  Step {step:2d}: {action_str:25s}  reward={reward:+7.2f}  "
              f"cumulative={env.state.cumulative_reward:+8.2f}")

        history.append({
            "user": (f"Step {step}. Features: [{', '.join(f'{v:.3f}' for v in obs.features)}]"
                     " What action?"),
            "assistant": action_str,
        })

    elapsed = time.time() - t0
    result = score_episode(task_id, env.state, env._tables)

    print(f"\n  → Score: {result.score:.4f}  Grade: {result.grade}  "
          f"Passed: {result.passed}  ({elapsed:.1f}s)")
    if result.notes:
        for note in result.notes:
            print(f"    ⚠  {note}")
    print(f"  → Breakdown: " +
          ", ".join(f"{k}={v:.2f}" for k, v in result.breakdown.items()))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("MEDUSA — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print()

    task_ids = ["clean_pipeline", "dirty_integration", "full_medallion"]
    results: dict[str, TaskResult] = {}
    total_start = time.time()

    for task_id in task_ids:
        result = run_task(task_id)
        results[task_id] = result

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25}  {'Difficulty':<8}  {'Score':>6}  {'Grade':>5}  {'Pass?':>5}")
    print("-" * 60)
    all_passed = True
    for task_id, result in results.items():
        task = TASKS[task_id]
        print(f"{task.name:<25}  {task.difficulty:<8}  "
              f"{result.score:>6.4f}  {result.grade:>5}  {'YES' if result.passed else 'NO':>5}")
        if not result.passed:
            all_passed = False

    print("-" * 60)
    avg = sum(r.score for r in results.values()) / len(results)
    print(f"{'Average':<25}  {'':8}  {avg:>6.4f}")
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Machine-readable output for the evaluator
    output = {
        "model": MODEL_NAME,
        "tasks": {
            tid: {
                "score": r.score,
                "grade": r.grade,
                "passed": r.passed,
                "breakdown": r.breakdown,
            }
            for tid, r in results.items()
        },
        "average_score": avg,
        "all_passed": all_passed,
    }
    print("\n--- JSON RESULTS ---")
    print(json.dumps(output, indent=2))

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
