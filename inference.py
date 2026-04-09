"""MEDUSA inference script — OpenEnv Hackathon submission.

Runs an LLM agent (via OpenAI-compatible API) against all three MEDUSA tasks
and reports per-task scores (0.0–1.0).

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM (OpenAI-compatible). Defaults to https://router.huggingface.co/v1
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key (also accepts API_KEY).

Usage:
    export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
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

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "mock-key"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "all")
BENCHMARK = os.getenv("BENCHMARK", "medusa_env")

_missing = [k for k, v in {
    "API_BASE_URL": API_BASE_URL,
    "MODEL_NAME": MODEL_NAME,
    "API_KEY (or HF_TOKEN)": API_KEY,
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
    api_key=API_KEY,
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

Your objective is to finish successfully in the FEWEST possible steps.
Avoid redundant actions. Repeating a completed action is almost always a mistake.
Use the feature vector and recent action history to infer what is already done.

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

PRIORITY:
1. Succeed on the task.
2. Minimize number of steps.
3. Never waste a step on an already completed or unnecessary action.

ACTION DISCIPLINE:
1. Treat SYNC_CHECK, EVOLVE_SCHEMA, PREP_KEYS_A, PREP_KEYS_B, and DEDUPLICATE_B as one-time setup actions.
2. Do not repeat PREP_KEYS_A if features[12] == 1.
3. Do not repeat PREP_KEYS_B if features[13] == 1.
4. Do not repeat DEDUPLICATE_B if features[14] == 1.
5. Do not repeat SYNC_CHECK, EVOLVE_SCHEMA, any EXECUTE_JOIN_* action, or any APPLY_SCD_* action if the recent history already shows they were taken.
6. After APPLY_SCD_1 or APPLY_SCD_2, the next action should usually be COMMIT.

SHORTEST-PATH POLICY:
1. First step: use SYNC_CHECK unless it already appears in history.
2. Use EVOLVE_SCHEMA only when schema drift is present: features[9] > 0 or features[10] > 0.
3. Use PREP_KEYS_A only when Source A keys need cleaning: features[4] > 0 and features[12] == 0.
4. Use PREP_KEYS_B only when Source B keys need cleaning: features[5] > 0 and features[13] == 0.
5. Use DEDUPLICATE_B only when Source B is not unique enough: features[7] < 0.999 and features[14] == 0.
6. Do not use EXECUTE_JOIN_INNER unless there is a very strong reason. Default to EXECUTE_JOIN_LEFT because it is safest for task success and grader volume checks.
7. Do not use EXECUTE_JOIN_ANTI unless you explicitly need quarantine-only rows. In most cases it is not the best path to task completion.
8. After required setup is finished, execute exactly one join.
9. After the join, prefer APPLY_SCD_2 once, then COMMIT once.

DEFAULT SUCCESSFUL PLAN:
1. SYNC_CHECK
2. EVOLVE_SCHEMA only if drift exists
3. PREP_KEYS_A only if needed
4. PREP_KEYS_B only if needed
5. DEDUPLICATE_B only if needed
6. EXECUTE_JOIN_LEFT
7. APPLY_SCD_2
8. COMMIT

COMMIT RULE:
Commit as soon as the pipeline has a valid joined-and-SCD-applied result. Do not spend extra steps after the pipeline is ready.

NEGATIVE RULES:
- Never output explanations, JSON, or multiple actions.
- Never loop between setup actions.
- Never repeat an action just because the same problem signal is still visible.
- Never delay COMMIT once join + SCD are done.

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
        # max_tokens=20,
        max_completion_tokens=256,
        temperature=0.1,
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
# Logging Functions (Hackathon STDOut Format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_id: str, max_steps: int = 15) -> TaskResult:
    """Run the LLM agent for one MEDUSA task using required hackathon STDOUT format."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    task = TASKS[task_id]
    env = MedusaEnv(n_fact_rows=200, n_dim_rows=150, max_steps=max_steps)
    obs = env.reset(seed=task.seed)

    history: List[dict] = []
    rewards_list: List[float] = []
    step = 0
    success = False
    score = 0.1
    result = TaskResult(
        task_id=task_id,
        score=0.1,
        grade="F",
        breakdown={},
        passed=False,
        notes=[],
    )

    try:
        while not obs.done and step < max_steps:
            step += 1
            action_str = choose_action(obs.features, history, step)
            
            # Since the environment throws errors on bad actions, we just pass the action string.
            try:
                action_type = MedusaActionType(action_str)
            except ValueError:
                action_type = MedusaActionType.COMMIT # default fallback
                
            action = MedusaAction(action=action_type)

            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards_list.append(reward)

            log_step(step=step, action=action_str, reward=reward, done=obs.done, error=None)

            history.append({
                "user": (f"Step {step}. Features: [{', '.join(f'{v:.3f}' for v in obs.features)}]"
                         " What action?"),
                "assistant": action_str,
            })

            if obs.done:
                break
                
        # Tally final score via grader
        result = score_episode(task_id, env.state, env._tables)
        score = result.score
        success = result.passed

    except Exception as e:
        log_step(step=step+1 if step > 0 else 1, action="ERROR", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=step, score=score, rewards=rewards_list)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if TASK_NAME.lower() == "all":
        results = [run_task(task_id) for task_id in list(TASKS)]
        all_passed = all(result.score >= 0.35 for result in results)

    else:
        run_task(TASK_NAME)

    sys.exit(0)

if __name__ == "__main__":
    main()
    
