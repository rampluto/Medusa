import argparse
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


POLICY_REGISTRY = {
    "random": RandomPolicy,
    "always_commit": AlwaysCommitPolicy,
    "clean_pipeline": CleanPipelinePolicy,
}


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one MEDUSA episode with a selectable policy."
    )
    parser.add_argument(
        "--policy",
        choices=sorted(POLICY_REGISTRY),
        default="clean_pipeline",
        help="Policy to run for the episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Scenario seed to use when resetting the environment.",
    )
    parser.add_argument(
        "--base-url",
        default=MEDUSA_URL,
        help="MEDUSA environment server base URL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step actions and terminal details.",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="Print the available policy names and exit.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_policies:
        print("Available policies:")
        for policy_name, policy_cls in sorted(POLICY_REGISTRY.items()):
            print(f"  {policy_name}: {policy_cls.name}")
        return 0

    policy = POLICY_REGISTRY[args.policy]()

    with medusa_env(base_url=args.base_url).sync() as env:
        print(f"\nRunning policy '{args.policy}' ({policy.name}) with seed={args.seed}:")
        reward = run_episode(env, policy, seed=args.seed, verbose=args.verbose)
        print(f"Final reward: {reward}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
