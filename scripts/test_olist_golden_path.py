import json
import sys
from pathlib import Path

# Fix python path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from server.medusa_env import MedusaEnv
from scenarios import OlistDayGenerator
from models import MedusaAction

def get_action(act, args=None):
    if args:
        return MedusaAction(action=f'<action>{act}</action><args>{json.dumps(args)}</args>')
    return MedusaAction(action=f'<action>{act}</action>')

def test_olist_golden_path():
    print("==================================================")
    print("GOLDEN PATH TEST: OLIST DAY 1")
    print("==================================================")
    
    # Initialize env with Olist Generator
    gen = OlistDayGenerator()
    env = MedusaEnv(day_generator=gen)
    obs = env.reset(seed=42)
    
    # Let's peek at Day 1 anomalies dynamically
    day1_anomalies = env._state.day_anomalies.get(1, [])
    print(f"Target Anomalies for Day 1: {day1_anomalies}")

    # Build the Golden Path
    actions = [("PROFILE_TABLE", {"table": "bronze"})]
    
    # 1. Resolve gap day abnormalities
    for col, op in day1_anomalies:
        actions.append(("CLEAN_COLUMN", {"table": "bronze", "col": col, "op": op}))
    
    # 2. In Olist, there are no duplicates to dedupe on Day 1, nor NULL keys.
    # So we just merge and commit!
    actions.append(("EXECUTE_MERGE", {}))
    actions.append(("COMMIT_DAY", {}))
    
    total_reward = 0.0
    for act, args in actions:
        obs = env.step(get_action(act, args))
        total_reward += obs.reward
        print(f"-> {act:<20} | Step Reward: {obs.reward:>6.2f} | Info: {obs.message[:80]}...")

    print(f"\nTarget Achieved? {obs.done and obs.metadata.get('grader_passed', False)}")
    print(f"Total Day 1 Return: {total_reward:.2f}")
    if total_reward > 0:
        print(">>> SUCCESS: Olist Day 1 is mathematically solvable and grants POSITIVE rewards!")
    else:
        print(">>> FAILURE: The reward structure is broken.")

if __name__ == "__main__":
    test_olist_golden_path()
