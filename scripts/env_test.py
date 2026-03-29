from medusa_env import MedusaActionType, MedusaAction, MedusaEnv

env = MedusaEnv(n_fact_rows=200, n_dim_rows=150)
obs = env.reset(seed=0)  # seed 0 = clean scenario
print(obs.message)

for action_type in [
    MedusaActionType.SYNC_CHECK,
    MedusaActionType.EVOLVE_SCHEMA,
    MedusaActionType.PREP_KEYS_A,
    MedusaActionType.PREP_KEYS_B,
    MedusaActionType.DEDUPLICATE_B,
    MedusaActionType.EXECUTE_JOIN_LEFT,
    MedusaActionType.APPLY_SCD_2,
    MedusaActionType.COMMIT,
]:
    obs = env.step(MedusaAction(action=action_type))
    print(f"{action_type.value:25s} reward={obs.reward:+.1f}  done={obs.done}")

print(f"\nGrader: {env.state.grader_report}")
