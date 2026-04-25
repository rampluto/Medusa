import argparse
import pandas as pd
from server.medusa_env import MedusaEnv
from scenarios import DayDataGenerator, detect_column_roles
from models import MedusaAction

# 1. Define a completely arbitrary schema that has NO Olist/synthetic columns
class RandomSpaceshipGenerator(DayDataGenerator):
    COLUMN_SPEC = {
        "crew_id": "id",
        "station_name": "categorical",
        "oxygen_levels": "numeric",
        "thrust_power": "numeric",
        "log_date": "date",
        "system_status": "string",
    }
    # Notice we don't define BASE_COLUMNS or PRIMARY_KEY — the environment
    # will detect them automatically from the DataFrames!


class CSVDayDataGenerator(DayDataGenerator):
    """Dynamically loads an arbitrary CSV as returning it for Day 1."""
    def __init__(self, csv_path: str, primary_key: str, episode_seed: int = 42):
        self.episode_seed = episode_seed
        self.csv_path = csv_path
        
        self._raw_data = pd.read_csv(csv_path)
        self.n_rows = len(self._raw_data)
        
        self._day_anomalies = {}
        
        # Detect roles
        self._roles = detect_column_roles(self._raw_data, primary_key=primary_key)
        self._pk_col = primary_key
        self._numeric_cols = list(self._roles.get("numeric", []))
        self._string_cols = list(self._roles.get("string", []) + self._roles.get("categorical", []))
        self._baseline_schema = list(self._raw_data.columns)
        
        self._build_anomaly_schedule()
        
    def _build_base_data(self, seed: int, n: int) -> pd.DataFrame:
        # Just return the loaded CSV unmodified for the base data
        return self._raw_data.copy()


def main():
    parser = argparse.ArgumentParser(description="Test MEDUSA environment on arbitrary datasets.")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument("--primary-key", type=str, help="Primary key column name")
    args = parser.parse_args()

    if args.input and not args.primary_key:
        parser.error("--primary-key is required if --input is provided")

    print("=== Testing MEDUSA Generalization ===")
    
    if args.input:
        print(f"Loading custom dataset: {args.input}")
        generator = CSVDayDataGenerator(csv_path=args.input, primary_key=args.primary_key, episode_seed=88)
    else:
        print("No --input provided. Using synthetic 'Spaceship' dataset fallback...")
        generator = RandomSpaceshipGenerator(episode_seed=88, n_rows=50, primary_key="crew_id")
    
    # Check what roles were detected under the hood (for logging purposes)
    print("\nDetected Column Roles:")
    print(f" - Primary Key: {generator._pk_col}")
    print(f" - Numeric Cols: {generator._numeric_cols}")
    print(f" - Baseline Schema: {generator._baseline_schema}\n")

    env = MedusaEnv(day_generator=generator)
    obs = env.reset()

    # We will step through Day 1 and see if we can commit it
    print(f"Day {env.state.current_day} loaded.")
    print("Current Contract Columns:", env.state.current_contract_columns)

    # Agent performs basic actions
    print("\nAction: PROFILE_TABLE")
    obs = env.step(MedusaAction(action="PROFILE_TABLE", params={}))
    print(obs.message)
    
    # Generic fix for test purposes
    print("\nAction: DEDUPLICATE")
    obs = env.step(MedusaAction(action="DEDUPLICATE", params={}))
    
    print("\nAction: EXECUTE_MERGE")
    obs = env.step(MedusaAction(action="EXECUTE_MERGE", params={}))
    print(obs.message)
    
    print("\nAction: COMMIT_DAY")
    try:
        obs = env.step(MedusaAction(action="COMMIT_DAY", params={}))
        print(obs.message)
    except Exception as e:
        print(f"\n❌ CRASH: Action raised an exception: {e}")
        return
    
    if obs.metrics.get("grader_passed"):
        print("\n✅ SUCCESS: The environment successfully processed and graded the dataset!")
    else:
        print("\n❌ FAILURE: Grader rejected the commit.")
        print(f"Grader report: {env.state.grader_report}")


if __name__ == "__main__":
    main()
