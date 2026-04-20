"""MEDUSA PPO Inference — evaluate trained PPO model against all tasks.

Loads a trained checkpoint (default: checkpoints/ppo_best.pt) and runs
one episode per task, printing results in the same [START]/[STEP]/[END]
format used by inference.py and inference_fair.py for direct comparison.

Usage:
    python inference_ppo.py                              # uses ppo_best.pt
    python inference_ppo.py --checkpoint checkpoints/ppo_medusa_0500.pt
    python inference_ppo.py --task full_medallion        # single task
    python inference_ppo.py --runs 5                     # 5 runs per task (avg)
    python inference_ppo.py --compare                    # side-by-side vs fair LLM

Requires: torch  (no API key needed — fully offline)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Path setup — support both installed-package and in-repo usage
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve().parent
_repo_root = str(_here.parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from medusa_env.server import MedusaEnv
    from medusa_env.models import MedusaAction, MedusaActionType
    from medusa_env.tasks import TASKS, TaskResult, score_episode
except ImportError:
    sys.path.insert(0, str(_here))
    from server.medusa_env import MedusaEnv          # type: ignore
    from models import MedusaAction, MedusaActionType  # type: ignore
    from tasks import TASKS, TaskResult, score_episode   # type: ignore

# train_ppo.py is in the same directory — import helpers from it directly
sys.path.insert(0, str(_here))
from train_ppo import (  # type: ignore
    MedusaActorCritic,
    compute_action_mask,
    _update_ep_flags,
    augment_obs,
    IDX_TO_ACTION,
    N_OBS,
    N_ACTIONS,
)

# ---------------------------------------------------------------------------
# Logging — identical format to inference.py / inference_fair.py
# ---------------------------------------------------------------------------

def log_start(task: str, model_path: str) -> None:
    print(f"[START] task={task} env=medusa_env model=ppo:{Path(model_path).stem}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error=null",
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
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    model: MedusaActorCritic,
    task_id: str,
    checkpoint_path: str,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> TaskResult:
    """Run one episode with the PPO model and return the scored result."""
    task = TASKS[task_id]
    effective_seed = seed if seed is not None else task.seed

    env = MedusaEnv(n_fact_rows=200, n_dim_rows=150, max_steps=20)
    obs_list = env.reset(seed=effective_seed).features
    obs_t = torch.tensor(obs_list, dtype=torch.float32)
    ep_flags: Dict[str, bool] = {}

    if verbose:
        log_start(task_id, checkpoint_path)

    rewards_list: List[float] = []
    step = 0
    h = model.initial_state()  # GRU hidden state, reset per episode

    with torch.no_grad():
        while step < 20:
            aug_obs = augment_obs(obs_list, ep_flags)
            obs_t = torch.tensor(aug_obs, dtype=torch.float32)
            mask = compute_action_mask(obs_list, ep_flags)
            action_t, _, _, _, h = model.get_action_and_value(
                obs_t.unsqueeze(0), mask.unsqueeze(0), h
            )
            action_type = IDX_TO_ACTION[action_t.item()]
            medusa_action = MedusaAction(action=action_type)

            obs_next = env.step(medusa_action)
            step += 1
            raw_reward = obs_next.reward or 0.0
            rewards_list.append(raw_reward)

            if verbose:
                log_step(step, action_type.value, raw_reward, obs_next.done)

            _update_ep_flags(ep_flags, action_type)

            if obs_next.done:
                break
            obs_list = obs_next.features
            obs_t = torch.tensor(obs_list, dtype=torch.float32)

    result = score_episode(task_id, env.state, env._tables)

    if verbose:
        log_end(result.passed, step, result.score, rewards_list)

    return result


# ---------------------------------------------------------------------------
# Multi-run averaging (accounts for policy stochasticity)
# ---------------------------------------------------------------------------

def run_task_averaged(
    model: MedusaActorCritic,
    task_id: str,
    checkpoint_path: str,
    n_runs: int = 1,
) -> TaskResult:
    """Run n_runs episodes and return the result with the median score."""
    if n_runs == 1:
        return run_episode(model, task_id, checkpoint_path, verbose=True)

    print(f"\n[MULTI-RUN] task={task_id}  runs={n_runs}", flush=True)
    results = []
    for i in range(n_runs):
        r = run_episode(model, task_id, checkpoint_path, verbose=False,
                        seed=TASKS[task_id].seed + i * 100)
        results.append(r)
        print(f"  run {i+1}/{n_runs}: score={r.score:.3f} grade={r.grade}", flush=True)

    # Return the median result
    results.sort(key=lambda r: r.score)
    median = results[len(results) // 2]
    mean_score = sum(r.score for r in results) / len(results)
    print(f"  → mean={mean_score:.3f}  median={median.score:.3f}  "
          f"best={max(r.score for r in results):.3f}  "
          f"worst={min(r.score for r in results):.3f}", flush=True)
    return median


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="MEDUSA PPO Inference")
    p.add_argument("--checkpoint", default="checkpoints/ppo_best.pt",
                   help="Path to .pt checkpoint file")
    p.add_argument("--task", default="all",
                   help="Task ID to run (default: all)")
    p.add_argument("--runs", type=int, default=1,
                   help="Runs per task for averaging (default: 1)")
    args = p.parse_args()

    # Load model
    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Run `python train_ppo.py` first to generate a checkpoint.", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MedusaActorCritic()
    model.load_state_dict(ckpt["model"])
    model.eval()
    saved_at = ckpt.get("iteration", "?")

    print(
        f"\n{'='*60}\n"
        f"  MEDUSA PPO Inference\n"
        f"  Checkpoint : {ckpt_path}  (saved at iter {saved_at})\n"
        f"  Mode       : {'single task' if args.task != 'all' else 'all tasks'}"
        f"  {'x' + str(args.runs) + ' runs' if args.runs > 1 else ''}\n"
        f"{'='*60}\n",
        flush=True,
    )

    # Determine which tasks to run
    if args.task.lower() == "all":
        task_ids = list(TASKS.keys())
    else:
        if args.task not in TASKS:
            print(f"ERROR: Unknown task '{args.task}'. Valid: {list(TASKS.keys())}")
            sys.exit(1)
        task_ids = [args.task]

    # Run episodes
    results: List[TaskResult] = []
    for task_id in task_ids:
        diff = TASKS[task_id].difficulty.upper()
        print(f"\n{'─'*60}", flush=True)
        print(f"  Task: {task_id}  [{diff}]", flush=True)
        print(f"{'─'*60}", flush=True)
        r = run_task_averaged(model, task_id, ckpt_path, n_runs=args.runs)
        results.append(r)

    # Summary table
    print(f"\n{'='*60}", flush=True)
    print(f"  {'Task':<30} {'Diff':<8} {'Score':>6}  {'Grade':>5}  {'Pass?':>5}", flush=True)
    print(f"{'─'*60}", flush=True)

    by_diff: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    for r in results:
        diff = TASKS[r.task_id].difficulty
        by_diff[diff].append(r.score)
        print(
            f"  {r.task_id:<30} {diff:<8} {r.score:>6.3f}  {r.grade:>5}  "
            f"{'✓' if r.passed else '✗':>5}",
            flush=True,
        )

    print(f"{'─'*60}", flush=True)
    mean_all = sum(r.score for r in results) / max(len(results), 1)
    easy_m   = sum(by_diff["easy"])   / max(len(by_diff["easy"]), 1)
    med_m    = sum(by_diff["medium"]) / max(len(by_diff["medium"]), 1)
    hard_m   = sum(by_diff["hard"])   / max(len(by_diff["hard"]), 1)

    print(f"  {'EASY mean':<30} {'':8} {easy_m:>6.3f}", flush=True)
    print(f"  {'MEDIUM mean':<30} {'':8} {med_m:>6.3f}", flush=True)
    print(f"  {'HARD mean':<30} {'':8} {hard_m:>6.3f}", flush=True)
    print(f"  {'OVERALL MEAN':<30} {'':8} {mean_all:>6.3f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
