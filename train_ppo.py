"""MEDUSA PPO Trainer — vanilla PyTorch, no external RL framework required.

Architecture
============
  MLP: 16 (obs) → 64 → 64
    ├─ Actor head  : 64 → 11 (masked softmax)
    └─ Critic head : 64 → 1  (value estimate)

Key features
============
  • Action masking      : enforces pipeline ordering (SCD after join, etc.)
  • Terminal reward     : agent paid only on final score_episode(0–1) at COMMIT,
                          not on gameable per-step bonuses.
  • Reward clipping     : terminal score scaled to 0–10, clipped via normalize_reward.
  • Curriculum          : Phase-1 easy seeds [0,3,6,10] → Phase-2 add medium
                          [1,4,8,9] → Phase-3 all 12 seeds.
  • GAE             : λ=0.95 advantage estimation.
  • Checkpoints     : saved every 100 iters; best checkpoint by eval score.
  • Evaluation      : 1 episode per task every 50 iters → eval_log.csv.

Usage
=====
  python train_ppo.py [--iterations 1000] [--lr 1e-4] [--seed 42] [--smoke-test]

  --smoke-test runs only 5 iterations (~20 rollout episodes) and prints scores.
  No GPU required — episodes are fast (≤20 steps on 200-row tables).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Environment imports — support both installed-package and in-repo usage
# ---------------------------------------------------------------------------

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from medusa_env.server import MedusaEnv
    from medusa_env.models import MedusaAction, MedusaActionType, MedusaState
    from medusa_env.tasks import TASKS, score_episode
except ImportError:
    # Running from inside the Medusa repo without package install
    _medusa_root = str(Path(__file__).resolve().parent)
    if _medusa_root not in sys.path:
        sys.path.insert(0, _medusa_root)
    from server.medusa_env import MedusaEnv          # type: ignore
    from models import MedusaAction, MedusaActionType, MedusaState  # type: ignore
    from tasks import TASKS, score_episode            # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of all 11 actions (index → MedusaActionType)
ALL_ACTIONS: List[MedusaActionType] = list(MedusaActionType)
N_ACTIONS = len(ALL_ACTIONS)   # 11
N_OBS = 19   # 16 env features + 3 episode flags (did_join, did_sync, did_evolve)

# Action index lookup
ACTION_TO_IDX: Dict[MedusaActionType, int] = {a: i for i, a in enumerate(ALL_ACTIONS)}
IDX_TO_ACTION: Dict[int, MedusaActionType] = {i: a for i, a in enumerate(ALL_ACTIONS)}

# Feature vector indices (from medusa_env._build_features)
F_STALE_A    = 2
F_STALE_B    = 3
F_NULL_A     = 4
F_NULL_B     = 5
F_UNIQ_B     = 7
F_NEW_COLS_A = 9
F_NEW_COLS_B = 10
F_DID_PREP_A = 12
F_DID_PREP_B = 13
F_DID_DEDUP  = 14

# Curriculum phases: (start_iteration, seed_list)
CURRICULUM: List[Tuple[int, List[int]]] = [
    (0,   [0, 3, 6, 10]),          # Phase 1 — easy
    (200, [0, 1, 3, 4, 6, 8, 9, 10]),   # Phase 2 — easy + medium
    (500, list(range(12))),        # Phase 3 — all 12
]

# Reward normalisation
REWARD_CLIP = 5.0   # Clip raw reward to ±5 before dividing
REWARD_SCALE = 5.0  # Divide clipped reward → ±1

# PPO hyperparameters (good defaults for small envs)
GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.05   # Higher than default to prevent early policy collapse
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_STEPS = 512        # Collect this many env timesteps per iteration
N_EPOCHS = 4         # PPO update epochs per iteration
MINIBATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------

def compute_action_mask(features: List[float], ep_flags: Dict[str, bool]) -> torch.Tensor:
    """Return a boolean mask (True = action ALLOWED) for the 11 discrete actions.

    Uses both the 16-float feature vector (for prep/dedup state) and an
    episode-level flag dict (for one-shot actions like SYNC_CHECK and JOIN).

    Args:
        features   : 16-float observation from the environment.
        ep_flags   : dict tracking which one-shot actions were already taken.

    Returns:
        BoolTensor of shape (11,) — True where action is legal.
    """
    mask = [True] * N_ACTIONS

    did_prep_a  = features[F_DID_PREP_A] > 0.5
    did_prep_b  = features[F_DID_PREP_B] > 0.5
    did_dedup   = features[F_DID_DEDUP]  > 0.5
    need_cols_a = features[F_NEW_COLS_A] > 0.0
    need_cols_b = features[F_NEW_COLS_B] > 0.0

    # --- One-shot setup actions: disable once done --------------------------
    if did_prep_a:
        mask[ACTION_TO_IDX[MedusaActionType.PREP_KEYS_A]] = False
    if did_prep_b:
        mask[ACTION_TO_IDX[MedusaActionType.PREP_KEYS_B]] = False
    if did_dedup:
        mask[ACTION_TO_IDX[MedusaActionType.DEDUPLICATE_B]] = False

    # Disable DEDUPLICATE_B if source B is already unique (no point)
    if features[F_UNIQ_B] >= 0.999 and not did_dedup:
        mask[ACTION_TO_IDX[MedusaActionType.DEDUPLICATE_B]] = False

    # Disable EVOLVE_SCHEMA if no new columns present
    if not (need_cols_a or need_cols_b):
        mask[ACTION_TO_IDX[MedusaActionType.EVOLVE_SCHEMA]] = False

    # --- Episode-level one-shot flags ---------------------------------------
    if ep_flags.get("did_sync_check"):
        mask[ACTION_TO_IDX[MedusaActionType.SYNC_CHECK]] = False

    if ep_flags.get("did_evolve_schema"):
        mask[ACTION_TO_IDX[MedusaActionType.EVOLVE_SCHEMA]] = False

    # Disable all JOIN variants after first join
    if ep_flags.get("did_join"):
        mask[ACTION_TO_IDX[MedusaActionType.EXECUTE_JOIN_INNER]] = False
        mask[ACTION_TO_IDX[MedusaActionType.EXECUTE_JOIN_LEFT]]  = False
        mask[ACTION_TO_IDX[MedusaActionType.EXECUTE_JOIN_ANTI]]  = False

    # Fix: block repeat SCD calls (SCD result is final, calling again is a no-op waste)
    if ep_flags.get("did_scd"):
        mask[ACTION_TO_IDX[MedusaActionType.APPLY_SCD_1]] = False
        mask[ACTION_TO_IDX[MedusaActionType.APPLY_SCD_2]] = False

    # ── Pipeline ordering (critical for real-world correctness) ────────────
    # SCD and COMMIT require a join result first
    if not ep_flags.get("did_join"):
        mask[ACTION_TO_IDX[MedusaActionType.APPLY_SCD_1]] = False
        mask[ACTION_TO_IDX[MedusaActionType.APPLY_SCD_2]] = False
        mask[ACTION_TO_IDX[MedusaActionType.COMMIT]] = False

    # Safety: at least one action must be legal
    if not any(mask):
        mask[ACTION_TO_IDX[MedusaActionType.COMMIT]] = True

    return torch.tensor(mask, dtype=torch.bool)


def _update_ep_flags(flags: Dict[str, bool], action_type: MedusaActionType) -> None:
    """Update episode-level action flags in-place."""
    if action_type == MedusaActionType.SYNC_CHECK:
        flags["did_sync_check"] = True
    elif action_type == MedusaActionType.EVOLVE_SCHEMA:
        flags["did_evolve_schema"] = True
    elif action_type in (
        MedusaActionType.EXECUTE_JOIN_INNER,
        MedusaActionType.EXECUTE_JOIN_LEFT,
        MedusaActionType.EXECUTE_JOIN_ANTI,
    ):
        flags["did_join"] = True
    elif action_type in (MedusaActionType.APPLY_SCD_1, MedusaActionType.APPLY_SCD_2):
        flags["did_scd"] = True


# ---------------------------------------------------------------------------
# Observation augmentation (fix: add episode history to obs)
# ---------------------------------------------------------------------------

def augment_obs(obs_list: List[float], ep_flags: Dict[str, bool]) -> List[float]:
    """Extend the 16-float env observation with 3 episode-level history flags.

    The environment only surfaces current-state features. Adding these flags
    lets the GRU explicitly see where in the pipeline the agent is, enabling
    it to learn order-conditional policies like 'prep before join'.

    Features 16-18 (appended):
      [16] did_join          - a join has been executed this episode
      [17] did_sync_check    - SYNC_CHECK has been called this episode
      [18] did_evolve_schema - EVOLVE_SCHEMA has been called this episode
    """
    return obs_list + [
        float(ep_flags.get("did_join", False)),
        float(ep_flags.get("did_sync_check", False)),
        float(ep_flags.get("did_evolve_schema", False)),
    ]


# ---------------------------------------------------------------------------
# Reward normalisation
# ---------------------------------------------------------------------------

def normalize_reward(raw: float) -> float:
    """Clip raw reward to ±REWARD_CLIP and scale to ±1."""
    return max(-REWARD_CLIP, min(REWARD_CLIP, raw)) / REWARD_SCALE


def compute_shaping_reward(
    obs_before: List[float],
    obs_after: List[float],
    action_type: MedusaActionType,
) -> float:
    """Potential-based shaping reward for state quality improvements.

    Gives a small immediate signal when setup actions genuinely improve
    data quality, so the agent receives gradient feedback *before* COMMIT.
    Rewards are bounded small (max ~+2.0/step) so they cannot override
    the terminal score_episode signal.

    Shaping criteria
    ----------------
    PREP_KEYS_A/B  : null_ratio drops → reward proportional to improvement
    DEDUPLICATE_B  : uniqueness rises  → reward proportional to improvement
    SYNC_CHECK     : +0.5 if at least one source was stale (check was needed)
    EVOLVE_SCHEMA  : +0.5 if new schema columns were present (evolution needed)
    """
    shaping = 0.0

    if action_type == MedusaActionType.PREP_KEYS_A:
        delta = obs_before[F_NULL_A] - obs_after[F_NULL_A]  # positive when nulls reduced
        if delta > 0.01:
            shaping += min(delta * 3.0, 1.5)  # cap at +1.5

    elif action_type == MedusaActionType.PREP_KEYS_B:
        delta = obs_before[F_NULL_B] - obs_after[F_NULL_B]
        if delta > 0.01:
            shaping += min(delta * 3.0, 1.5)

    elif action_type == MedusaActionType.DEDUPLICATE_B:
        delta = obs_after[F_UNIQ_B] - obs_before[F_UNIQ_B]  # positive when uniqueness rises
        if delta > 0.01:
            shaping += min(delta * 3.0, 1.5)

    elif action_type == MedusaActionType.SYNC_CHECK:
        # Reward if the check was actually needed (sources were stale)
        if obs_before[F_STALE_A] > 0.5 or obs_before[F_STALE_B] > 0.5:
            shaping += 0.5

    elif action_type == MedusaActionType.EVOLVE_SCHEMA:
        # Reward if schema evolution was actually needed
        if obs_before[F_NEW_COLS_A] > 0.0 or obs_before[F_NEW_COLS_B] > 0.0:
            shaping += 0.5

    return shaping


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class MedusaActorCritic(nn.Module):
    """GRU-based actor-critic for sequential MEDUSA decision making.

    Architecture:
        obs (N_OBS=19) → Linear(64) → Tanh → GRU(64→64)
                        ├─ actor_head:  Linear(64→11 logits)
                        └─ critic_head: Linear(64→1 value)

    The GRU maintains a hidden state across steps within each episode,
    allowing the policy to condition on prior actions — the key capability
    needed for learning 'prep before join' ordering.

    Callers must pass h (hidden state) and receive h_next each step.
    Hidden state is reset (zeroed) at episode boundaries.
    """

    def __init__(self, obs_dim: int = N_OBS, act_dim: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.hidden_size = hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor_head  = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)

        # Orthogonal init
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        """Return zeroed GRU hidden state: shape (1, batch, hidden)."""
        return torch.zeros(1, batch, self.hidden_size)

    def forward_step(
        self,
        obs: torch.Tensor,  # (batch, obs_dim)
        h: torch.Tensor,    # (1, batch, hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward: returns (logits, value, h_next)."""
        enc = self.encoder(obs)                        # (batch, hidden)
        out, h_next = self.gru(enc.unsqueeze(1), h)   # (batch,1,hid), (1,batch,hid)
        feat = out.squeeze(1)                          # (batch, hidden)
        return self.actor_head(feat), self.critic_head(feat).squeeze(-1), h_next

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        h: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action with masking. Returns (action, log_prob, entropy, value, h_next)."""
        logits, value, h_next = self.forward_step(obs, h)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, h_next


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Fixed-size circular buffer for PPO rollouts."""

    capacity: int
    obs:      torch.Tensor = field(init=False)
    actions:  torch.Tensor = field(init=False)
    log_probs:torch.Tensor = field(init=False)
    rewards:  torch.Tensor = field(init=False)
    dones:    torch.Tensor = field(init=False)
    values:   torch.Tensor = field(init=False)
    masks:    torch.Tensor = field(init=False)  # action masks
    ptr: int = field(default=0, init=False)

    def __post_init__(self):
        C = self.capacity
        self.obs       = torch.zeros(C, N_OBS)
        self.actions   = torch.zeros(C, dtype=torch.long)
        self.log_probs = torch.zeros(C)
        self.rewards   = torch.zeros(C)
        self.dones     = torch.zeros(C)
        self.values    = torch.zeros(C)
        self.masks     = torch.ones(C, N_ACTIONS, dtype=torch.bool)

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        mask: torch.Tensor,
    ) -> None:
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.dones[i]     = float(done)
        self.values[i]    = value
        self.masks[i]     = mask
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.capacity

    def reset(self) -> None:
        self.ptr = 0

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = GAMMA, lam: float = LAMBDA_GAE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages for the collected rollout."""
        n = self.ptr
        advantages = torch.zeros(n)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val   = last_value if t == n - 1 else self.values[t + 1].item()
            next_done  = 0.0        if t == n - 1 else self.dones[t + 1].item()
            delta      = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            last_gae   = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        return returns, advantages


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------

def get_curriculum_seeds(iteration: int) -> List[int]:
    """Return the seed pool for the current training iteration."""
    seeds = CURRICULUM[0][1]
    for start_iter, seed_list in CURRICULUM:
        if iteration >= start_iter:
            seeds = seed_list
    return seeds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    n_iterations: int = 1000,
    lr: float = 1e-4,
    seed: int = 42,
    smoke_test: bool = False,
    checkpoint_dir: str = "checkpoints_2",
) -> None:
    """Main PPO training loop.

    Args:
        n_iterations  : Number of PPO iterations (each collects N_STEPS timesteps).
        lr            : Adam learning rate.
        seed          : Global RNG seed for reproducibility.
        smoke_test    : Run only 5 iterations then exit.
        checkpoint_dir: Directory to save checkpoints and eval_log.csv.
    """
    if smoke_test:
        n_iterations = 5
        print("[smoke-test] Running 5 iterations only.")

    torch.manual_seed(seed)
    random.seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)
    eval_log_path = os.path.join(checkpoint_dir, "eval_log.csv")

    # Model + optimiser
    model = MedusaActorCritic()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # Environment (shared across rollout; reset per episode)
    env = MedusaEnv(n_fact_rows=200, n_dim_rows=150, max_steps=20)

    buffer = RolloutBuffer(capacity=N_STEPS)

    best_mean_score = 0.0
    eval_log_rows: List[Dict] = []

    # ── Rollout state ────────────────────────────────────────────────────────
    curriculum_seeds = get_curriculum_seeds(0)
    current_seed = random.choice(curriculum_seeds)
    obs_list = env.reset(seed=current_seed).features
    ep_flags: Dict[str, bool] = {}
    aug_obs  = augment_obs(obs_list, ep_flags)
    obs_t    = torch.tensor(aug_obs, dtype=torch.float32)
    h_t      = model.initial_state()   # GRU hidden state, reset each episode
    ep_step  = 0

    t_start = time.time()

    for iteration in range(1, n_iterations + 1):

        # Update curriculum seeds for this iteration
        curriculum_seeds = get_curriculum_seeds(iteration)

        # ── Collect N_STEPS timesteps ────────────────────────────────────────
        buffer.reset()
        model.eval()

        ep_rewards_this_iter: List[float] = []
        current_ep_raw_reward = 0.0
        episode_slices: List[tuple] = []   # (start_idx, end_idx) per episode
        ep_start_idx  = 0

        with torch.no_grad():
            while not buffer.is_full():
                mask = compute_action_mask(obs_list, ep_flags)
                action_t, log_prob_t, _, value_t, h_next = model.get_action_and_value(
                    obs_t.unsqueeze(0), mask.unsqueeze(0), h_t
                )
                action_idx = action_t.item()
                action_type = IDX_TO_ACTION[action_idx]
                medusa_action = MedusaAction(action=action_type)

                obs_next = env.step(medusa_action)
                obs_after = obs_next.features if obs_next.features else obs_list
                raw_reward = obs_next.reward if obs_next.reward is not None else 0.0
                done = obs_next.done
                current_ep_raw_reward += raw_reward

                shaping = compute_shaping_reward(obs_list, obs_after, action_type)

                if done:
                    task_id = next(
                        (tid for tid, t in TASKS.items() if t.seed == current_seed),
                        "clean_pipeline",
                    )
                    final_result = score_episode(task_id, env.state, env._tables)
                    terminal_reward = final_result.score * 10.0
                    norm_reward = normalize_reward(terminal_reward - 0.2 + shaping)
                else:
                    norm_reward = normalize_reward(-0.2 + shaping)

                buffer.add(
                    obs=obs_t,
                    action=action_idx,
                    log_prob=log_prob_t.item(),
                    reward=norm_reward,
                    done=done,
                    value=value_t.item(),
                    mask=mask,
                )

                ep_step += 1
                _update_ep_flags(ep_flags, action_type)
                h_t = h_next

                if done or ep_step >= 20:
                    ep_rewards_this_iter.append(current_ep_raw_reward)
                    episode_slices.append((ep_start_idx, buffer.ptr))  # record slice
                    ep_start_idx = buffer.ptr
                    current_seed = random.choice(curriculum_seeds)
                    obs_list = env.reset(seed=current_seed).features
                    ep_flags = {}
                    aug_obs  = augment_obs(obs_list, ep_flags)
                    obs_t    = torch.tensor(aug_obs, dtype=torch.float32)
                    h_t      = model.initial_state()   # reset hidden state
                    ep_step  = 0
                    current_ep_raw_reward = 0.0
                else:
                    obs_list = obs_after
                    aug_obs  = augment_obs(obs_list, ep_flags)
                    obs_t    = torch.tensor(aug_obs, dtype=torch.float32)

            # Bootstrap last value for GAE (use current h_t)
            mask_last = compute_action_mask(obs_list, ep_flags)
            _, _, _, last_value_t, _ = model.get_action_and_value(
                obs_t.unsqueeze(0), mask_last.unsqueeze(0), h_t
            )
            last_value = last_value_t.item()

        returns, advantages = buffer.compute_returns_and_advantages(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── PPO update: episode-shuffled GRU (Fix 3+2) ───────────────────────
        # Process episodes in random order per epoch. Each episode is an
        # independent optimization step with proper BPTT (no cross-episode
        # gradient contamination). Entropy coefficient anneals 0.15 → 0.02.
        model.train()
        n = buffer.ptr

        # Entropy annealing (Fix 2): starts high (exploration), decays to min
        entropy_coef = max(0.02, 0.15 * (1.0 - iteration / n_iterations))

        pg_losses, vf_losses, ent_losses = [], [], []

        for epoch in range(N_EPOCHS):
            random.shuffle(episode_slices)               # shuffle episode order
            for ep_start, ep_end in episode_slices:
                if ep_start >= ep_end:
                    continue
                h_ep = model.initial_state()
                ep_lp, ep_ent, ep_val = [], [], []

                for t in range(ep_start, ep_end):
                    _, lp, ent, val, h_ep = model.get_action_and_value(
                        buffer.obs[t:t+1],
                        buffer.masks[t:t+1],
                        h_ep,
                        buffer.actions[t:t+1],
                    )
                    ep_lp.append(lp)
                    ep_ent.append(ent)
                    ep_val.append(val)

                if not ep_lp:
                    continue

                new_lp  = torch.cat(ep_lp)
                new_val = torch.cat(ep_val)
                new_ent = torch.cat(ep_ent)

                old_lp = buffer.log_probs[ep_start:ep_end]
                ep_ret = returns[ep_start:ep_end]
                ep_adv = advantages[ep_start:ep_end]

                ratio   = torch.exp(new_lp - old_lp)
                pg1     = -ep_adv * ratio
                pg2     = -ep_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = F.mse_loss(new_val, ep_ret)
                ent_loss = new_ent.mean()

                loss = pg_loss + VALUE_COEF * vf_loss - entropy_coef * ent_loss

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimiser.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        # ── Logging ──────────────────────────────────────────────────────────
        mean_ep_reward = (
            sum(ep_rewards_this_iter) / len(ep_rewards_this_iter)
            if ep_rewards_this_iter else 0.0
        )
        phase = next(
            (i + 1 for i, (start, _) in reversed(list(enumerate(CURRICULUM)))
             if iteration >= start),
            1,
        )
        elapsed = time.time() - t_start
        print(
            f"[iter {iteration:04d}] phase={phase} "
            f"ep_reward={mean_ep_reward:+.2f} "
            f"pg={sum(pg_losses)/len(pg_losses):.4f} "
            f"vf={sum(vf_losses)/len(vf_losses):.4f} "
            f"ent={sum(ent_losses)/len(ent_losses):.4f} "
            f"t={elapsed:.1f}s",
            flush=True,
        )

        # ── Evaluation ───────────────────────────────────────────────────────
        if iteration % 50 == 0 or iteration == n_iterations:
            model.eval()
            eval_scores = _evaluate(model, list(TASKS.keys()), holdout_seeds=list(range(12, 18)))
            mean_score = sum(eval_scores[t] for t in TASKS.keys()) / len(TASKS)
            easy_seeds   = [t for t in TASKS if TASKS[t].difficulty == "easy"]
            medium_seeds = [t for t in TASKS if TASKS[t].difficulty == "medium"]
            hard_seeds   = [t for t in TASKS if TASKS[t].difficulty == "hard"]
            easy_mean   = sum(eval_scores[t] for t in easy_seeds)   / max(len(easy_seeds), 1)
            medium_mean = sum(eval_scores[t] for t in medium_seeds) / max(len(medium_seeds), 1)
            hard_mean   = sum(eval_scores[t] for t in hard_seeds)   / max(len(hard_seeds), 1)
            holdout_mean = eval_scores.get("holdout_mean", 0.0)

            print(
                f"  ──── EVAL iter={iteration} "
                f"easy={easy_mean:.3f} medium={medium_mean:.3f} hard={hard_mean:.3f} "
                f"holdout={holdout_mean:.3f} MEAN={mean_score:.3f} ────",
                flush=True,
            )
            for task_id, sc in eval_scores.items():
                print(f"    {task_id:30s}: {sc:.3f}", flush=True)

            row = {
                "iteration": iteration,
                "mean_score": round(mean_score, 4),
                "easy_mean":  round(easy_mean, 4),
                "medium_mean":round(medium_mean, 4),
                "hard_mean":  round(hard_mean, 4),
                **{f"score_{t}": round(s, 4) for t, s in eval_scores.items()},
            }
            eval_log_rows.append(row)
            _write_csv(eval_log_path, eval_log_rows)

            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_path = os.path.join(checkpoint_dir, "ppo_best.pt")
                torch.save({"model": model.state_dict(), "iteration": iteration}, best_path)
                print(f"  ★ New best checkpoint → {best_path}  (score={best_mean_score:.3f})",
                      flush=True)

        # ── Checkpoint ───────────────────────────────────────────────────────
        if iteration % 100 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"ppo_medusa_{iteration:04d}.pt")
            torch.save({"model": model.state_dict(), "iteration": iteration}, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}", flush=True)

    print(f"\nTraining complete. Best mean score: {best_mean_score:.3f}", flush=True)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model: MedusaActorCritic,
    task_ids: List[str],
    holdout_seeds: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Run one episode per task_id; optionally also run holdout seeds.

    holdout_seeds (Fix 4): seeds NOT in TASKS that measure generalisation.
    Each is scored against the nearest known-task rubric (round-robin).
    Results keyed as 'holdout_<seed>' plus 'holdout_mean' aggregate.
    """
    env = MedusaEnv(n_fact_rows=200, n_dim_rows=150, max_steps=20)
    scores: Dict[str, float] = {}
    known_task_ids = list(TASKS.keys())

    def _run(seed: int, rubric_id: str) -> float:
        obs_list = env.reset(seed=seed).features
        ep_flags: Dict[str, bool] = {}
        aug = augment_obs(obs_list, ep_flags)
        obs_t = torch.tensor(aug, dtype=torch.float32)
        h = model.initial_state()
        for _ in range(20):
            mask = compute_action_mask(obs_list, ep_flags)
            action_t, _, _, _, h = model.get_action_and_value(
                obs_t.unsqueeze(0), mask.unsqueeze(0), h, deterministic=True
            )
            action_type = IDX_TO_ACTION[action_t.item()]
            obs_next = env.step(MedusaAction(action=action_type))
            _update_ep_flags(ep_flags, action_type)
            if obs_next.done:
                break
            obs_list = obs_next.features
            aug = augment_obs(obs_list, ep_flags)
            obs_t = torch.tensor(aug, dtype=torch.float32)
        return score_episode(rubric_id, env.state, env._tables).score

    with torch.no_grad():
        for task_id in task_ids:
            scores[task_id] = _run(TASKS[task_id].seed, task_id)

        if holdout_seeds:
            holdout_scores = []
            for hs in holdout_seeds:
                rubric_id = known_task_ids[hs % len(known_task_ids)]
                sc = _run(hs, rubric_id)
                scores[f"holdout_{hs}"] = sc
                holdout_scores.append(sc)
            if holdout_scores:
                scores["holdout_mean"] = sum(holdout_scores) / len(holdout_scores)

    return scores


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEDUSA PPO Trainer")
    p.add_argument("--iterations",      type=int,   default=1000,          help="Number of PPO iterations")
    p.add_argument("--lr",              type=float, default=1e-4,           help="Adam learning rate")
    p.add_argument("--seed",            type=int,   default=42,             help="Global RNG seed")
    p.add_argument("--checkpoint-dir",  type=str,   default="checkpoints",  help="Directory for checkpoints")
    p.add_argument("--smoke-test",      action="store_true",                help="Run 5 iterations then exit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n_iterations=args.iterations,
        lr=args.lr,
        seed=args.seed,
        smoke_test=args.smoke_test,
        checkpoint_dir=args.checkpoint_dir,
    )
