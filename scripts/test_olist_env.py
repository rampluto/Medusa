import sys
from pathlib import Path

# Fix python path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

try:
    from server.medusa_env import MedusaEnv
    from scenarios import OlistDayGenerator

    def test_env():
        gen = OlistDayGenerator()
        env = MedusaEnv(n_fact_rows=100, n_dim_rows=50, day_generator=gen)
        
        obs = env.reset(seed=42)
        print("SUCCESS: Environment reset successfully with OlistDayGenerator!")
        
        print("\n=== Initial Day 1 Observations ===")
        print(obs.message[:500] + "...")
        print("\n=== Day Anomalies Map (first few days) ===")
        for d in range(1, 5):
            print(f"Day {d}:", env._state.day_anomalies.get(d))
            
    if __name__ == "__main__":
        test_env()
except Exception as e:
    import traceback
    traceback.print_exc()
