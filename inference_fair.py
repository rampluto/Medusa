"""MEDUSA Fair LLM Inference — stripped system prompt for fair comparison.

This script is identical to ``inference.py`` EXCEPT that the system prompt
has been stripped of all hand-written policy guidance:

  REMOVED  : DEFAULT SUCCESSFUL PLAN, SHORTEST-PATH POLICY, ACTION DISCIPLINE,
             NEGATIVE RULES, COMMIT RULE — all explicit step-by-step instructions.
  KEPT     : Action names + one-line descriptions, feature vector index table,
             high-level objective.

This creates a fair head-to-head: the LLM must infer WHEN to call each action
from the raw 16-float feature vector alone — exactly the same information
available to the PPO agent in ``train_ppo.py``.

Required environment variables:
    API_BASE_URL   (default: https://router.huggingface.co/v1)
    MODEL_NAME     (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN / API_KEY

Usage:
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
    export HF_TOKEN="hf-..."
    python inference_fair.py

Output format is identical to inference.py ([START]/[STEP]/[END] lines).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "mock-key"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME    = os.getenv("TASK_NAME", "all")
BENCHMARK    = os.getenv("BENCHMARK", "medusa_env")

_missing = [k for k, v in {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME":   MODEL_NAME,
    "API_KEY (or HF_TOKEN)": API_KEY,
}.items() if not v]

if _missing:
    print(f"ERROR: Missing env vars: {', '.join(_missing)}", file=sys.stderr)
    sys.exit(1)

from openai import OpenAI  # noqa: E402

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# MEDUSA imports
# ---------------------------------------------------------------------------

from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from envs.medusa_env import MedusaEnv
    from envs.medusa_env.models import MedusaAction, MedusaActionType
    from envs.medusa_env.tasks import TASKS, TaskResult, score_episode
except ImportError:
    from medusa_env import MedusaEnv          # type: ignore
    from models import MedusaAction, MedusaActionType  # type: ignore
    from tasks import TASKS, TaskResult, score_episode   # type: ignore

# ---------------------------------------------------------------------------
# STRIPPED system prompt
#
# What was removed vs inference.py:
#   ✗  ACTION DISCIPLINE section   (do-not-repeat rules referencing feature flags)
#   ✗  SHORTEST-PATH POLICY section (features[X] > Y trigger rules)
#   ✗  DEFAULT SUCCESSFUL PLAN     (the explicit 8-step script)
#   ✗  COMMIT RULE                 (explicit commit trigger)
#   ✗  NEGATIVE RULES              (explicit anti-patterns)
#   ✗  PRIORITY ordering
#
# What was kept:
#   ✓  Action names + one-line descriptions (LLM must know what exists)
#   ✓  Feature vector index definitions (LLM can observe but must infer)
#   ✓  High-level objective
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are a data integration agent controlling a Bronze→Silver ETL pipeline.

You observe a 16-float feature vector describing data quality signals, and
you must choose one action per step from the list below.

Your objective: produce a valid, historically-consistent Silver table from
two Bronze sources (Source A = Fact, Source B = Dimension).

ACTIONS (respond with ONLY the action name — nothing else):
  SYNC_CHECK          — Verify source freshness before processing
  EVOLVE_SCHEMA       — Add new columns from sources into Silver schema
  PREP_KEYS_A         — Clean and normalise join keys in Source A (Fact)
  PREP_KEYS_B         — Clean and normalise join keys in Source B (Dimension)
  DEDUPLICATE_B       — Remove duplicate keys from Source B
  EXECUTE_JOIN_INNER  — Inner join A ⋈ B
  EXECUTE_JOIN_LEFT   — Left join A ⋈ B (keeps all Fact rows)
  EXECUTE_JOIN_ANTI   — Anti-join: extract Fact rows with no Dimension match
  APPLY_SCD_1         — Overwrite Silver records (SCD Type 1)
  APPLY_SCD_2         — Close old records and insert new with timestamps (SCD Type 2)
  COMMIT              — Finalise pipeline and trigger audit

Feature vector indices:
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
# Action chooser
# ---------------------------------------------------------------------------

VALID_ACTIONS = {a.value for a in MedusaActionType}


def choose_action(features: List[float], history: List[dict], step: int) -> str:
    """Ask the LLM to choose the next action given the feature vector."""
    feature_str = ", ".join(f"{v:.3f}" for v in features)
    user_msg = (
        f"Step {step}. Feature vector: [{feature_str}]\n"
        "What is the single best next action? Respond with ONLY the action name."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-4:]:
        messages.append({"role": "user",      "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=64,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip().upper().replace(" ", "_")

    for action in VALID_ACTIONS:
        if action in raw:
            return action
    for action in sorted(VALID_ACTIONS, key=len, reverse=True):
        if action.replace("_", "") in raw.replace("_", ""):
            return action
    return MedusaActionType.COMMIT.value


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_id: str, max_steps: int = 15) -> TaskResult:
    """Run the fair-LLM agent for one MEDUSA task."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    task = TASKS[task_id]
    env  = MedusaEnv(n_fact_rows=200, n_dim_rows=150, max_steps=max_steps)
    obs  = env.reset(seed=task.seed)

    history: List[dict] = []
    rewards_list: List[float] = []
    step = 0
    success = False
    score   = 0.1
    result  = TaskResult(task_id=task_id, score=0.1, grade="F",
                         breakdown={}, passed=False, notes=[])

    try:
        while not obs.done and step < max_steps:
            step += 1
            action_str = choose_action(obs.features, history, step)
            try:
                action_type = MedusaActionType(action_str)
            except ValueError:
                action_type = MedusaActionType.COMMIT

            obs = env.step(MedusaAction(action=action_type))
            reward = obs.reward or 0.0
            rewards_list.append(reward)

            log_step(step=step, action=action_str, reward=reward,
                     done=obs.done, error=None)
            history.append({
                "user": (
                    f"Step {step}. Features: [{', '.join(f'{v:.3f}' for v in obs.features)}]"
                    " What action?"
                ),
                "assistant": action_str,
            })
            if obs.done:
                break

        result  = score_episode(task_id, env.state, env._tables)
        score   = result.score
        success = result.passed

    except Exception as e:
        log_step(step=step + 1 if step > 0 else 1, action="ERROR",
                 reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=step, score=score, rewards=rewards_list)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    task_ids = list(TASKS)[:5] if TASK_NAME.lower() == "all" else [TASK_NAME]

    print(
        f"\n{'='*60}\n"
        f"  MEDUSA Fair LLM Inference\n"
        f"  Model  : {MODEL_NAME}\n"
        f"  Prompt : STRIPPED (no policy guide)\n"
        f"{'='*60}\n",
        flush=True,
    )

    results = [run_task(tid) for tid in task_ids]

    # Summary table
    print(f"\n{'─'*60}", flush=True)
    print(f"  {'Task':<30} {'Score':>6}  {'Grade':>5}  {'Pass?':>5}", flush=True)
    print(f"{'─'*60}", flush=True)
    for r in results:
        print(
            f"  {r.task_id:<30} {r.score:>6.3f}  {r.grade:>5}  "
            f"{'✓' if r.passed else '✗':>5}",
            flush=True,
        )
    mean_score = sum(r.score for r in results) / max(len(results), 1)
    print(f"{'─'*60}", flush=True)
    print(f"  {'MEAN':<30} {mean_score:>6.3f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
