"""Stable Baselines 3 Due Diligence Test Harness.

This file is isolated and should be DELETED before the hackathon submission.
It wraps the OpenEnv `MedusaEnv` in a tight `gymnasium.Env` wrapper, then uses
the official SB3 environment checker and PPO algorithm to prove the environment:
  1. Conforms to strict API standards.
  2. Does not produce NaN/Inf observations.
  3. Can be "solved" (reward > 0) on a static seed in <10k steps.
"""

import sys
import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("CRITICAL: stable_baselines3 or gymnasium is not installed.")
    print("Please run: pip install stable_baselines3 gymnasium")
    sys.exit(1)

from server.medusa_env import MedusaEnv
from models import MedusaAction

# ──────────────────────────────────────────────────────────────────────────────
# Gym Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MedusaGymWrapper(gym.Env):
    """Wraps the OpenEnv MedusaEnv to strictly comply with Gymnasium API."""
    
    def __init__(self, seed: int = 42):
        super().__init__()
        self.medusa = MedusaEnv(max_steps=50) # 5 days * 10 steps
        self._fixed_seed = seed
        
        # 16-element float observation vector strictly bounded [0, 1]
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(16,), 
            dtype=np.float32
        )
        
        # 7 Discrete Actions
        self.action_space = gym.spaces.Discrete(7)
        self.action_mapping = [
            '<action>PROFILE_TABLE</action><args>{"table": "bronze"}</args>',
            '<action>CLEAN_COLUMN</action><args>{"table": "bronze", "column": "revenue", "operation": "strip"}</args>',
            '<action>DEDUPLICATE</action><args>{"table": "bronze"}</args>',
            '<action>QUARANTINE_ROWS</action><args>{"condition": "user_id IS NULL"}</args>',
            '<action>EVOLVE_SILVER_SCHEMA</action><args>{}</args>',
            '<action>EXECUTE_MERGE</action><args>{}</args>',
            '<action>COMMIT_DAY</action><args>{}</args>',
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # We force a fixed seed for the overfit test
        obs = self.medusa.reset(seed=self._fixed_seed)
        return np.array(obs.features, dtype=np.float32), {}

    def step(self, action: int):
        act_str = self.action_mapping[action]
        obs = self.medusa.step(MedusaAction(action=act_str))
        
        # SB3 strict requirements: types must be exactly correct
        state_vec = np.array(obs.features, dtype=np.float32)
        reward = float(obs.reward if obs.reward is not None else 0.0)
        terminated = bool(obs.done)
        truncated = False
        info = {}
        
        return state_vec, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.current_return = 0.0
        self.episodes_completed = 0

    def _on_step(self):
        self.current_return += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_returns.append(self.current_return)
            self.current_return = 0.0
            self.episodes_completed += 1
            
            if self.episodes_completed % 10 == 0:
                avg = np.mean(self.episode_returns[-10:])
                print(f"Episodes: {self.episodes_completed:03d} | Moving Avg Return: {avg:.2f}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────

def run_due_diligence():
    print("==================================================")
    print("1. RUNNING GYM API CONFORMANCE CHECK")
    print("==================================================")
    env = MedusaGymWrapper(seed=42)
    try:
        check_env(env, warn=True)
        print("✓ Env check passed! Fully Gymnasium compliant, observations are clean.")
    except Exception as e:
        print(f"✗ ENV CHECK FAILED: {e}")
        return

    print("\n==================================================")
    print("2. RUNNING PPO OVERFIT SANITY TEST (10,000 steps)")
    print("==================================================")
    print("Goal: PPO should easily solve this fixed seed and achieve a positive return.\n")
    
    # Tiny, fast PPO configuration suitable for laptops
    model = PPO(
        "MlpPolicy", 
        env, 
        n_steps=1024,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        device="cpu"
    )
    
    logger = TrainingLoggerCallback()
    model.learn(total_timesteps=10000, callback=logger)

    print("\nTest Complete.")
    final_avg = np.mean(logger.episode_returns[-10:])
    
    if final_avg > 0:
        print(f"✓ PPO SUCCESSFULLY SOLVED THE ENVIRONMENT! Final Avg Return: {final_avg:.2f}")
    else:
        print(f"✗ PPO FAILED TO SOLVE THE ENVIRONMENT. Final Avg Return: {final_avg:.2f}")
        print("  - Check your reward shaping or observation features.")

if __name__ == "__main__":
    run_due_diligence()
