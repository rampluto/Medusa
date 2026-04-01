import random

# Support both installed package usage (medusa_env) and in-repo local modules
try:
    from medusa_env import (
        medusa_env,
        MedusaAction,
        MedusaActionType,
        MedusaObservation,
    )
except ImportError:
    # Fallback to local modules when running from the repo root without installing
    from client import medusa_env
    from models import MedusaAction, MedusaActionType, MedusaObservation

MEDUSA_URL = 'https://anubhavkamal-medusa-env.hf.space'
# MEDUSA_URL = 'http://localhost:8000'


class RandomPolicy:
    """Pure random — baseline for MEDUSA."""
    name = "Random"

    def select_action(self, obs: MedusaObservation) -> MedusaActionType:
        # Pick randomly from the 11 valid operators
        return random.choice(list(MedusaActionType))


class AlwaysCommitPolicy:
    """Immediately terminates the episode by committing."""
    name = "Always Commit"

    def select_action(self, obs: MedusaObservation) -> MedusaActionType:
        return MedusaActionType.COMMIT


class CleanPipelinePolicy:
    """Hardcoded sequence to perfectly solve the Easy (Clean Pipeline) task."""
    name = "Clean Pipeline Heuristic"

    def __init__(self):
        # The correct sequence of operations for the clean pipeline scenario
        self.sequence = [
            MedusaActionType.SYNC_CHECK,
            MedusaActionType.PREP_KEYS_A,
            MedusaActionType.PREP_KEYS_B,
            MedusaActionType.EXECUTE_JOIN_LEFT,
            MedusaActionType.APPLY_SCD_2,
            MedusaActionType.COMMIT
        ]
        self.step = 0

    def select_action(self, obs: MedusaObservation) -> MedusaActionType:
        if self.step < len(self.sequence):
            action = self.sequence[self.step]
            self.step += 1
            return action
        return MedusaActionType.COMMIT


print("Policies defined: Random, Always Commit, Clean Pipeline Heuristic")


def run_episode(env, policy, seed=0, verbose=False):
    """Play one episode. Returns the final reward (-1.0 to 1.0)."""
    result = env.reset(seed=seed)
    step = 0

    while not result.done:
        action_type = policy.select_action(result.observation)
        if verbose:
            print(f'  Step {step}: {action_type.value}')
            
        result = env.step(MedusaAction(action=action_type))
        step += 1

    if verbose:
        print(f'  Result: Done (reward={result.reward})')
        print(f'  Terminal Message: {result.observation.message}')
        if result.observation.metrics:
            print(f'  Final Grade: {result.observation.metrics.get("grader_report")}')
    return result.reward


# Demo: one verbose episode with CleanPipelinePolicy
with medusa_env(base_url=MEDUSA_URL).sync() as env:
    print('\nTesting Clean Pipeline Policy — single episode (seed=0):')
    run_episode(env, CleanPipelinePolicy(), seed=0, verbose=True)