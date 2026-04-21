"""MEDUSA Environment Client.

Connects to a running MEDUSA server via WebSocket for persistent sessions.

Example:
    >>> # Connect to a running server
    >>> with medusa_env(base_url="http://localhost:8000") as client:
    ...     result = client.reset(seed=0)
    ...     print(result.observation.message)
    ...
    ...     from medusa_env.models import MedusaActionType
    ...     result = client.step(MedusaAction(action=MedusaActionType.PROFILE_TABLE))
    ...     print(f"Reward: {result.reward}")
"""

from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from medusa_env.models import MedusaAction, MedusaObservation, MedusaState
except ImportError:
    from models import MedusaAction, MedusaObservation, MedusaState

    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class medusa_env(EnvClient[MedusaAction, MedusaObservation, MedusaState]):
    """Client for the MEDUSA Bronze→Silver integration environment.

    Maintains a persistent WebSocket connection to the MEDUSA server.
    Each client instance has its own dedicated environment session.

    The agent observes a 16-float data quality feature vector and chooses
    from 7 discrete ETL actions (v4.0) to build a correct Silver entity
    across a 30-day gauntlet.

    Example:
        >>> with medusa_env(base_url="http://localhost:8000") as env:
        ...     result = env.reset(seed=0)
        ...     result = env.step(MedusaAction(action="<action>PROFILE_TABLE</action>"))
        ...     result = env.step(MedusaAction(action="<action>CLEAN_COLUMN</action>"))
        ...     result = env.step(MedusaAction(action="<action>EXECUTE_MERGE</action>"))
        ...     result = env.step(MedusaAction(action="<action>COMMIT_DAY</action>"))
        ...     print(result.reward)
    """

    def _step_payload(self, action: MedusaAction) -> Dict[str, Any]:
        """Convert MedusaAction to JSON payload for the step request."""
        act_val = action.action
        if hasattr(act_val, 'value'):
            act_val = act_val.value
        return {
            "action": act_val,
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
            # v4.0 30-day gauntlet
            current_day=payload.get("current_day", 1),
            step_count=payload.get("step_count", 0),
            retry_count=payload.get("retry_count", 0),
            did_dedup_today=payload.get("did_dedup_today", False),
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
            # Schema
            did_evolve_schema=payload.get("did_evolve_schema", False),
            # Silver / Quarantine
            silver_row_count=payload.get("silver_row_count", 0),
            quarantine_row_count=payload.get("quarantine_row_count", 0),
            source_row_count=payload.get("source_row_count", 0),
            source_a_row_count=payload.get("source_a_row_count", 0),
            total_raw_rows=payload.get("total_raw_rows", 0),
            total_quarantine_rows=payload.get("total_quarantine_rows", 0),
            # Grader
            current_contract_columns=payload.get("current_contract_columns", []),
            grader_passed=payload.get("grader_passed", False),
            grader_report=payload.get("grader_report", ""),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
