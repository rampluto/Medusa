"""MEDUSA Environment Client.

Connects to a running MEDUSA server via WebSocket for persistent sessions.

Example:
    >>> # Connect to a running server
    >>> with medusa_env(base_url="http://localhost:8000") as client:
    ...     result = client.reset(seed=0)
    ...     print(result.observation.message)
    ...
    ...     from envs.medusa_env.models import MedusaActionType
    ...     result = client.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
    ...     print(f"Reward: {result.reward}")

Example with Docker:
    >>> client = medusa_env.from_docker_image("medusa_env:latest")
    >>> try:
    ...     result = client.reset()
    ...     result = client.step(MedusaAction(action=MedusaActionType.COMMIT))
    ... finally:
    ...     client.close()
"""

from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import MedusaAction, MedusaObservation, MedusaState
except ImportError:
    from models import MedusaAction, MedusaObservation, MedusaState

    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class medusa_env(EnvClient[MedusaAction, MedusaObservation, MedusaState]):
    """Client for the MEDUSA Bronze→Silver integration environment.

    Maintains a persistent WebSocket connection to the MEDUSA server.
    Each client instance has its own dedicated environment session.

    The agent observes a 16-float data quality feature vector and chooses
    from 11 discrete ETL actions to build a correct Silver entity from
    two Bronze sources (Fact + Dimension).

    Example:
        >>> with medusa_env(base_url="http://localhost:8000") as env:
        ...     result = env.reset(seed=0)          # clean scenario
        ...     result = env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
        ...     result = env.step(MedusaAction(action=MedusaActionType.PREP_KEYS_A))
        ...     result = env.step(MedusaAction(action=MedusaActionType.PREP_KEYS_B))
        ...     result = env.step(MedusaAction(action=MedusaActionType.DEDUPLICATE_B))
        ...     result = env.step(MedusaAction(action=MedusaActionType.EXECUTE_JOIN_LEFT))
        ...     result = env.step(MedusaAction(action=MedusaActionType.APPLY_SCD_2))
        ...     result = env.step(MedusaAction(action=MedusaActionType.COMMIT))
        ...     print(result.reward)
    """

    def _step_payload(self, action: MedusaAction) -> Dict[str, Any]:
        """Convert MedusaAction to JSON payload for the step request."""
        return {
            "action": action.action.value,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MedusaObservation]:
        """Parse server response into StepResult[MedusaObservation]."""
        obs_data = payload.get("observation", {})
        observation = MedusaObservation(
            message=obs_data.get("message", ""),
            features=obs_data.get("features", []),
            metrics=obs_data.get("metrics", {}),
            metadata=obs_data.get("metadata", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MedusaState:
        """Parse server response into MedusaState."""
        return MedusaState(
            run_id=payload.get("run_id"),
            seed=payload.get("seed"),
            scenario_id=payload.get("scenario_id"),
            step_idx=payload.get("step_idx", 0),
            stage=payload.get("stage", "init"),
            # Freshness
            time_delta_a=payload.get("time_delta_a", 0.0),
            time_delta_b=payload.get("time_delta_b", 0.0),
            is_stale_a=payload.get("is_stale_a", False),
            is_stale_b=payload.get("is_stale_b", False),
            did_sync_check=payload.get("did_sync_check", False),
            # Key health
            null_ratio_key_a=payload.get("null_ratio_key_a", 0.0),
            null_ratio_key_b=payload.get("null_ratio_key_b", 0.0),
            uniqueness_a=payload.get("uniqueness_a", 1.0),
            uniqueness_b=payload.get("uniqueness_b", 1.0),
            did_prep_a=payload.get("did_prep_a", False),
            did_prep_b=payload.get("did_prep_b", False),
            did_dedup_b=payload.get("did_dedup_b", False),
            # Join
            match_rate=payload.get("match_rate", 0.0),
            did_join=payload.get("did_join", False),
            join_type=payload.get("join_type"),
            join_row_count=payload.get("join_row_count", 0),
            explosion_detected=payload.get("explosion_detected", False),
            # SCD
            did_scd=payload.get("did_scd", False),
            scd_type=payload.get("scd_type"),
            scd_inserts=payload.get("scd_inserts", 0),
            scd_updates=payload.get("scd_updates", 0),
            # Silver / Quarantine
            silver_row_count=payload.get("silver_row_count", 0),
            quarantine_row_count=payload.get("quarantine_row_count", 0),
            source_a_row_count=payload.get("source_a_row_count", 0),
            # Grader
            grader_passed=payload.get("grader_passed", False),
            grader_report=payload.get("grader_report", ""),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
