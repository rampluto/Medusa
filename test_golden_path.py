import json
from medusa_env.server.medusa_env import MedusaEnv
from medusa_env.models import MedusaAction

def get_action(act, args=None):
    if args:
        return MedusaAction(action=f'<action>{act}</action><args>{json.dumps(args)}</args>')
    return MedusaAction(action=f'<action>{act}</action>')

def golden_path_test():
    print("==================================================")
    print("1. GOLDEN PATH DAY 1 & MERGE REWARD TEST")
    print("==================================================")
    env = MedusaEnv()
    obs = env.reset(seed=42)
    
    actions = [
        ("PROFILE_TABLE", {"table": "bronze"}),
        ("CLEAN_COLUMN", {"table": "bronze", "col": "revenue", "op": "cast"}),
        ("CLEAN_COLUMN", {"table": "bronze", "col": "discount_amount", "op": "fill_zero"}),
        ("DEDUPLICATE", {"key": "user_id"}),
        ("QUARANTINE_ROWS", {"condition": "user_id IS NULL"}),
        ("EXECUTE_MERGE", {}),
        ("COMMIT_DAY", {})
    ]
    
    total_reward = 0.0
    for act, args in actions:
        obs = env.step(get_action(act, args))
        total_reward += obs.reward
        print(f"-> {act:<20} | Reward: {obs.reward:>6.2f} | Result: {obs.message[:70]}")
        
        if act == "EXECUTE_MERGE":
            print(f">>> CHECK 1: Merge reward is exactly {obs.reward} (should be around +2.9 / +3.0 with step cost)")

    print(f"Total Day 1 Golden Return: {total_reward:.2f}\n")
    if total_reward > 0:
         print(">>> CONCLUSION: Golden Path yields POSITIVE reward! Training is mathematically possible.")
    else:
         print(">>> CONCLUSION: Golden Path yields NEGATIVE reward. The reward scale is broken.")


def crash_test_day1():
    print("\n==================================================")
    print("2. PROPORTIONAL CRASH TEST (DAY 1)")
    print("==================================================")
    env = MedusaEnv()
    env.reset(seed=42)
    
    # Just try to commit immediately, grader will fail it
    obs = env.step(get_action("COMMIT_DAY", {}))
    print(f"-> Bad Commit Day 1 | Reward: {obs.reward} | Done: {obs.done}")
    print(f">>> CHECK 2: Crash reward on Day 1 is {obs.reward} (should be -145.0)")


def almost_correct_test():
    print("\n==================================================")
    print("3. ALMOST CORRECT AGENT (DAY 1-13 PASS, DAY 14 CRASH)")
    print("==================================================")
    env = MedusaEnv()
    env.reset(seed=42)
    
    total_return = 0.0
    
    # Pass days 1 to 13
    for day in range(1, 14):
        actions = [
            ("PROFILE_TABLE", {"table": "bronze"}),
            ("CLEAN_COLUMN", {"table": "bronze", "col": "revenue", "op": "cast"}),
            ("CLEAN_COLUMN", {"table": "bronze", "col": "discount_amount", "op": "fill_zero"}),
            ("DEDUPLICATE", {"key": "user_id"}),
            ("QUARANTINE_ROWS", {"condition": "user_id IS NULL"}),
            ("EXECUTE_MERGE", {}),
            ("COMMIT_DAY", {})
        ]
        
        for act, args in actions:
            obs = env.step(get_action(act, args))
            total_return += obs.reward
    
    print(f"Return after 13 PERFECT days: {total_return:.2f}")
    
    # Fail on day 14 (Skip deduplicate and quarantine)
    bad_actions = [
        ("EXECUTE_MERGE", {}),
        ("COMMIT_DAY", {})
    ]
    for act, args in bad_actions:
        obs = env.step(get_action(act, args))
        total_return += obs.reward
        
    print(f"-> Day 14 Fail | Final Step Reward: {obs.reward} | Done: {obs.done}")
    print(f">>> CHECK 3: Total Episode Return for 'Almost Correct' Agent: {total_return:.2f}")
    print(f">>> Comparing Day 14 fails ({total_return:.2f}) vs Day 1 fails (-145.0).")
    print(">>> If Day 14 return is much higher, PPO gradient is active!")


if __name__ == "__main__":
    golden_path_test()
    crash_test_day1()
    almost_correct_test()
