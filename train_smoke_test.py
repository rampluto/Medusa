"""Minimal Laptop Training Smoke Test for MEDUSA v4.0.

This is a zero-dependency (other than PyTorch & Numpy) Policy Gradient (REINFORCE)
training script. It serves as a rapid smoke test to prove that the MedusaEnv
produces a learnable signal.

Hardware: Designed to run efficiently on laptop CPUs in < 2 minutes.
Strategy:
  1. Isolates the "easy" pipeline (no schema drift / anomalies) by using a fixed 
     seed (which generates a clean `basic_pipeline` scenario).
  2. Uses a tiny MLP (16 -> 64 -> 7) to learn mapping of states to actions.
  3. Tracks moving average reward. If it goes from negative (random) to positive,
     the environment is providing valid gradients.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from server.medusa_env import MedusaEnv
from models import MedusaAction, MedusaActionType

# ---------------------------------------------------------
# 1. Hyperparameters
# ---------------------------------------------------------
LEARNING_RATE = 2e-3
GAMMA = 0.99
NUM_EPISODES = 300
HIDDEN_DIM = 64

# Action mapping (simplified to core v4.0 tools for the smoke test)
# For the easy pipeline, we mostly need profile, merge, and commit.
ACTION_SPACE = [
    '<action>PROFILE_TABLE</action><args>{"table": "bronze"}</args>',
    '<action>CLEAN_COLUMN</action><args>{"table": "bronze", "column": "revenue", "operation": "strip"}</args>',
    '<action>DEDUPLICATE</action><args>{"table": "bronze"}</args>',
    '<action>QUARANTINE_ROWS</action><args>{"condition": "user_id IS NULL"}</args>',
    '<action>EVOLVE_SILVER_SCHEMA</action><args>{}</args>',
    '<action>EXECUTE_MERGE</action><args>{}</args>',
    '<action>COMMIT_DAY</action><args>{}</args>',
]
NUM_ACTIONS = len(ACTION_SPACE)
OBS_DIM = 16  # v4.0 features vector length

# ---------------------------------------------------------
# 2. Policy Network
# ---------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_ACTIONS)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train_smoke_test():
    print("Initializing MEDUSA Training Smoke Test (REINFORCE)...")
    env = MedusaEnv(max_steps=50) # 10 steps max per day, over 5 days = 50 steps
    
    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    returns_history = []
    
    for episode in range(1, NUM_EPISODES + 1):
        # We explicitly seed it with a clean scenario (easy pipeline)
        # to ensure fast convergence for a smoke test.
        obs = env.reset(seed=42)
        state_vec = obs.features
        
        log_probs = []
        rewards = []
        
        done = False
        while not done:
            # 1. Forward pass
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            action_probs = policy(state_tensor)
            
            # 2. Sample action
            m = torch.distributions.Categorical(action_probs)
            action_idx = m.sample()
            
            # 3. Step env
            action_str = ACTION_SPACE[action_idx.item()]
            next_obs = env.step(MedusaAction(action=action_str))
            
            log_probs.append(m.log_prob(action_idx))
            rewards.append(next_obs.reward)
            
            state_vec = next_obs.features
            done = next_obs.done
        
        # 4. Compute discounted returns for REINFORCE
        discounted_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            discounted_returns.insert(0, R)
            
        discounted_returns = torch.tensor(discounted_returns)
        # Normalize returns for stability
        if len(discounted_returns) > 1:
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)
            
        # 5. Policy gradient update
        loss = []
        for log_prob, R_t in zip(log_probs, discounted_returns):
            loss.append(-log_prob * R_t)
            
        optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        optimizer.step()
        
        # 6. Logging
        ep_return = sum(rewards)
        returns_history.append(ep_return)
        
        if episode % 10 == 0:
            avg_return = np.mean(returns_history[-10:])
            stage = env.state.stage
            days = env.state.current_day
            print(f"Episode {episode:03d} | Avg Return (last 10): {avg_return:7.2f} | Last Ep Days Survived: {days} | End Stage: {stage}")

    print("\nTraining Smoke Test Complete!")
    print("If 'Avg Return' escalated from negative/zero to positive, your MEDUSA environment provides a learnable gradient.")

if __name__ == "__main__":
    train_smoke_test()
