import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from server.medusa_env import MedusaEnv
from scenarios import DayDataGenerator, detect_column_roles
from models import MedusaAction
from medusa_prompts import SYSTEM_PROMPT

# Dynamic generator for randomization
class RandomizedSchemaGenerator(DayDataGenerator):
    """Generates completely randomized dataset schemas to prevent LLM overfitting."""
    def __init__(self, episode_seed: int, n_rows: int = 100):
        # Generate random schema specs
        rng = random.Random(episode_seed)
        
        # Pick random domain tokens
        domain = rng.choice(["medical", "finance", "logistics", "gaming", "retail"])
        
        cols = {
            f"{domain}_id_col": "id",
            f"{domain}_metric_1": "numeric",
            f"{domain}_metric_2": "numeric",
            f"{domain}_category": "categorical",
            f"{domain}_notes": "string",
            f"{domain}_date": "date"
        }
        
        self.COLUMN_SPEC = cols
        self.episode_seed = episode_seed
        self.n_rows = n_rows
        self._day_anomalies = {}
        
        self._sample_df = self._build_base_data(seed=episode_seed, n=n_rows)
        self._pk_col = f"{domain}_id_col"
        self._roles = detect_column_roles(self._sample_df, primary_key=self._pk_col)
        
        self._numeric_cols = list(self._roles.get("numeric", []))
        self._string_cols = list(self._roles.get("string", []) + self._roles.get("categorical", []))
        self._baseline_schema = list(self._sample_df.columns)
        
        self._build_anomaly_schedule()


def get_expert_action(env: MedusaEnv) -> tuple[str, MedusaAction]:
    """A perfect rule-based solver that returns CoT reasoning and the exact action."""
    state = env.state
    
    # 1. Check for error recovery first
    recovery_prefix = ""
    bad_keywords = ["BLOCK", "INVALID", "not found", "ERROR", "Penalty"]
    if any(k in state.last_action_result for k in bad_keywords):
        recovery_prefix = f"Wait, my previous action failed: {state.last_action_result.strip()} Let me correct my approach. "
    
    # 2. Evolve Schema if drift is detected
    if state.new_schema_cols and not state.did_evolve_schema:
        reason = f"Schema drift detected: new columns {state.new_schema_cols}. I must evolve the silver schema to data contract spec before merging."
        return recovery_prefix + reason, MedusaAction(action="EVOLVE_SILVER_SCHEMA", params={"column": state.new_schema_cols[0]})
        
    # 3. Fix anomalies
    if state.unhandled_anomalies_today:
        col = list(state.unhandled_anomalies_today.keys())[0]
        ops = state.unhandled_anomalies_today[col]
        op = ops[0]
        
        reason = f"I see an unhandled anomaly on column '{col}'. It requires a '{op}' operation."
        if op == "quarantine":
            return recovery_prefix + reason, MedusaAction(action="QUARANTINE_ROWS", params={"table": "bronze", "condition": f"{col} IS NULL"})
        elif op == "evolve":
            return recovery_prefix + reason, MedusaAction(action="EVOLVE_SILVER_SCHEMA", params={"column": col})
        elif op == "deduplicate":
            return recovery_prefix + reason, MedusaAction(action="DEDUPLICATE", params={"key": col})
        elif op == "type_mixed":
            return recovery_prefix + reason, MedusaAction(action="CLEAN_COLUMN", params={"col": col, "op": "cast"})
        elif op == "fill_null":
            return recovery_prefix + reason, MedusaAction(action="CLEAN_COLUMN", params={"col": col, "op": "fill_zero"})
        elif op == "whitespace":
            return recovery_prefix + reason, MedusaAction(action="CLEAN_COLUMN", params={"col": col, "op": "strip"})
        else:
            return recovery_prefix + reason, MedusaAction(action="CLEAN_COLUMN", params={"col": col, "op": op})
            
    # 4. Deduplicate
    if not state.did_dedup_today:
        reason = "All initial anomalies are handled, but I have not deduplicated the batch yet. Must deduplicate."
        return recovery_prefix + reason, MedusaAction(action="DEDUPLICATE", params={})
        
    # 5. Merge
    if not state.did_merge_today:
        reason = "The batch is cleaned and deduplicated. Ready to merge into Silver."
        return recovery_prefix + reason, MedusaAction(action="EXECUTE_MERGE", params={})
        
    # 6. Commit
    reason = "Merge executed successfully. Committing the day."
    return recovery_prefix + reason, MedusaAction(action="COMMIT_DAY", params={})


def main():
    parser = argparse.ArgumentParser(description="Generate SFT JSONL dataset.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of 30-day episodes to generate")
    parser.add_argument("--out", type=str, default="sft_dataset.jsonl", help="Output file path")
    args = parser.parse_args()

    out_path = Path(args.out)
    successful_steps = 0
    
    print(f"Generating expert SFT trajectories for {args.episodes} episodes...")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for ep in range(args.episodes):
            seed = 1000 + ep
            generator = RandomizedSchemaGenerator(episode_seed=seed, n_rows=50)
            env = MedusaEnv(day_generator=generator, max_steps=100)
            obs = env.reset()
            
            while not obs.done:
                # 1. Get LLM Prompt
                prompt = env.generate_llm_prompt()
                
                # 2. Get Expert Action
                reasoning, action = get_expert_action(env)
                
                # 10% chance to intentionally screw up to teach Error-Recovery
                # Only if we aren't ALREADY recovering from a mistake, to prevent crash chains
                bad_keywords = ["BLOCK", "INVALID", "not found", "ERROR", "Penalty"]
                is_recovering = any(k in env.state.last_action_result for k in bad_keywords)
                
                if random.random() < 0.10 and not is_recovering:
                    reasoning = "I will attempt this syntax blindly."
                    bad_actions = [
                        MedusaAction(action="WRONG_SYNTAX_ACTION", params={"col": "fake"}),
                        MedusaAction(action="CLEAN_COLUMN", params={"col": "hallucinated_col", "op": "cast"}),
                        MedusaAction(action="EXECUTE_MERGE", params={}) # If called before dedup, it will block
                    ]
                    action = random.choice(bad_actions)
                
                # 3. Format Target: <think> reasoning + fenced JSON action.
                # The fence is critical: env messages can contain '{...}'
                # (grader reports etc.), so leaking those into the reasoning
                # text would break a greedy '\{.*\}' regex parse. Wrapping the
                # action JSON in ```json...``` makes the eval-side parser
                # unambiguous regardless of reasoning content.
                action_json = json.dumps(
                    {"action": action.action, "params": action.params}
                )
                action_text = (
                    f"<think>\n{reasoning}\n</think>\n"
                    f"```json\n{action_json}\n```"
                )

                chatml = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": action_text},
                    ]
                }
                f.write(json.dumps(chatml) + "\n")
                successful_steps += 1
                
                # 4. Step environment
                obs = env.step(action)
                
            print(f"Finished episode {ep+1} (Score: {env.state.cumulative_reward:.2f})")
                
    print(f"\nSFT Dataset generated! Saved {successful_steps} perfect interaction steps to {out_path}.")


if __name__ == "__main__":
    main()
