"""MEDUSA agent policies for the custom UI."""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Type, Union

try:
    from ..models import MedusaAction, MedusaActionType
    from ..tasks import Task
except ImportError:
    try:
        from medusa_env.models import MedusaAction, MedusaActionType
        from medusa_env.tasks import Task
    except ImportError:
        from models import MedusaAction, MedusaActionType
        from tasks import Task


PolicyAction = Union[MedusaAction, MedusaActionType]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDescriptor:
    """Metadata for a selectable MEDUSA agent."""

    id: str
    name: str
    description: str
    family: str
    strengths: List[str]
    default: bool = False


class BaseAgentPolicy:
    """Base class for deterministic MEDUSA policies."""

    descriptor: AgentDescriptor

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._seed = seed

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        raise NotImplementedError


class RandomPolicy(BaseAgentPolicy):
    """Random baseline for comparison and debugging."""

    descriptor = AgentDescriptor(
        id="random",
        name="Random Baseline",
        description="Samples a valid MEDUSA action at random until the episode closes.",
        family="baseline",
        strengths=["Replay stress test", "Lower-bound reward baseline"],
    )

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)
        self._random = random.Random(seed)

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        action_type = self._random.choice(list(MedusaActionType))
        LOGGER.debug("random_policy_selected_action action=%s", action_type.value)
        return MedusaAction(action=action_type.value, params={})


_HEURISTIC_OP_MAP: Dict[str, str] = {
    "type_mixed": "cast",
    "fill_null": "fill_zero",
    "whitespace": "strip",
    "negative": "fill_zero",
}


class HeuristicPolicy(BaseAgentPolicy):
    """Golden-path heuristic policy.

    Implements the reference solution from ``scripts/test_olist_golden_path.py``:

    1. ``PROFILE_TABLE`` bronze once per day.
    2. Walk the day's anomaly checklist and route each entry to the right action
       (``CLEAN_COLUMN``/``DEDUPLICATE``/``QUARANTINE_ROWS``/``EVOLVE_SILVER_SCHEMA``).
    3. Fall back to ``DEDUPLICATE`` if duplicates still remain and none was issued.
    4. ``EXECUTE_MERGE`` once per day.
    5. ``COMMIT_DAY`` to close the day.

    The same logic drives both the task auto-run path (``MedusaEnv.state``) and
    the dataframe cleaner path (``SimpleNamespace`` state built from the upload).
    """

    descriptor = AgentDescriptor(
        id="heuristic",
        name="Heuristic Golden-Path",
        description=(
            "Reference solver that profiles the table, resolves each anomaly on the "
            "day's checklist, deduplicates, merges, and commits — matching the golden "
            "path from the Olist test suite."
        ),
        family="heuristic",
        strengths=[
            "Reference upper-bound reward",
            "Deterministic checklist traversal",
            "Works on tasks and uploaded dataframes",
        ],
    )

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        profiled = getattr(state, "profiled_tables_today", {}) or {}
        has_profiled = bool(profiled) if isinstance(profiled, dict) else bool(profiled)
        if not has_profiled:
            LOGGER.debug("heuristic_policy_select profile_bronze")
            return MedusaAction(action="PROFILE_TABLE", params={"table": "bronze"})

        current_day = getattr(state, "current_day", 1)
        day_anomalies = getattr(state, "day_anomalies", {}) or {}
        today = day_anomalies.get(current_day, []) or []

        cleaned_lookup = set()
        for item in getattr(state, "cleaned_columns_today", []) or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cleaned_lookup.add((item[0], item[1]))

        contract_cols = list(getattr(state, "current_contract_columns", []) or [])
        did_dedup = bool(getattr(state, "did_dedup_today", False))
        did_evolve = bool(getattr(state, "did_evolve_schema", False))

        for col, raw_op in today:
            raw_op_lower = str(raw_op).lower()

            if raw_op_lower in {"deduplicate", "dedup", "duplicate"}:
                if not did_dedup:
                    LOGGER.debug("heuristic_policy_select dedup col=%s", col)
                    return MedusaAction(action="DEDUPLICATE", params={"key": col})
                continue

            if raw_op_lower == "quarantine":
                if (col, "quarantine") not in cleaned_lookup:
                    LOGGER.debug("heuristic_policy_select quarantine col=%s", col)
                    return MedusaAction(
                        action="QUARANTINE_ROWS",
                        params={"table": "bronze", "condition": f"{col} IS NULL"},
                    )
                continue

            if raw_op_lower == "evolve":
                if not did_evolve and (col, "evolve") not in cleaned_lookup:
                    LOGGER.debug("heuristic_policy_select evolve col=%s", col)
                    return MedusaAction(
                        action="EVOLVE_SILVER_SCHEMA",
                        params={"column": col},
                    )
                continue

            mapped_op = _HEURISTIC_OP_MAP.get(raw_op_lower, raw_op_lower)
            if (col, mapped_op) in cleaned_lookup:
                continue
            LOGGER.debug("heuristic_policy_select clean col=%s op=%s", col, mapped_op)
            return MedusaAction(
                action="CLEAN_COLUMN",
                params={"table": "bronze", "col": col, "op": mapped_op},
            )

        if not did_dedup and float(getattr(state, "uniqueness_b", 1.0) or 1.0) < 1.0:
            LOGGER.debug("heuristic_policy_select dedup_fallback")
            return MedusaAction(action="DEDUPLICATE", params={})

        if not bool(getattr(state, "did_merge_today", False)):
            LOGGER.debug("heuristic_policy_select merge")
            return MedusaAction(action="EXECUTE_MERGE", params={})

        LOGGER.debug("heuristic_policy_select commit")
        return MedusaAction(action="COMMIT_DAY", params={})


class GrpoTrainedPolicy(BaseAgentPolicy):
    """GRPO policy wrapper that delegates action prediction to a predictor callable."""

    descriptor = AgentDescriptor(
        id="grpo_trained",
        name="GRPO Trained Agent",
        description=(
            "Executes a deterministic policy aligned with the GRPO training objective for "
            "stable replay and reproducible audit behavior."
        ),
        family="trained",
        strengths=["Replay-stable", "Strong benchmark completion"],
        default=True,
    )

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)
        self._predict_action = _load_grpo_predictor()
        LOGGER.info("grpo_policy_initialized seed=%s", seed)

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        predicted = self._predict_action(task=task, state=state, observation=observation)
        LOGGER.debug("grpo_policy_raw_prediction type=%s", type(predicted).__name__)
        if isinstance(predicted, MedusaAction):
            LOGGER.debug("grpo_policy_selected_action action=%s", predicted.action)
            return predicted
        if isinstance(predicted, MedusaActionType):
            LOGGER.debug("grpo_policy_selected_action action=%s", predicted.value)
            return MedusaAction(action=predicted.value, params={})
        action_name = str(predicted)
        if action_name not in {action.value for action in MedusaActionType}:
            LOGGER.error("grpo_policy_unknown_action action=%s", action_name)
            raise ValueError(f"GRPO predictor returned unknown action {action_name!r}.")
        LOGGER.debug("grpo_policy_selected_action action=%s", action_name)
        return MedusaAction(action=action_name, params={})


def _load_grpo_predictor() -> Any:
    """Load a configured GRPO action predictor from MEDUSA_GRPO_PREDICTOR."""
    import os

    entrypoint = os.getenv("MEDUSA_GRPO_PREDICTOR", "").strip()
    if not entrypoint:
        LOGGER.error("grpo_predictor_missing_env MEDUSA_GRPO_PREDICTOR is not set")
        raise RuntimeError(
            "grpo_trained agent requires MEDUSA_GRPO_PREDICTOR='module:function' to be configured."
        )
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise RuntimeError(
            "Invalid MEDUSA_GRPO_PREDICTOR format. Expected 'module:function'."
        )
    module = import_module(module_name)
    predictor = getattr(module, function_name, None)
    if predictor is None or not callable(predictor):
        LOGGER.error("grpo_predictor_invalid entrypoint=%s", entrypoint)
        raise RuntimeError(
            f"Configured GRPO predictor '{entrypoint}' is missing or not callable."
        )
    LOGGER.info("grpo_predictor_loaded entrypoint=%s", entrypoint)
    return predictor


AGENT_REGISTRY: Dict[str, Type[BaseAgentPolicy]] = {
    policy.descriptor.id: policy
    for policy in (
        RandomPolicy,
        HeuristicPolicy,
        GrpoTrainedPolicy,
    )
}

DEFAULT_AGENT_ID = next(
    descriptor.id
    for descriptor in (policy.descriptor for policy in AGENT_REGISTRY.values())
    if descriptor.default
)


def build_agent(agent_id: str, *, seed: Optional[int] = None) -> BaseAgentPolicy:
    """Instantiate a known policy."""
    try:
        policy_cls = AGENT_REGISTRY[agent_id]
    except KeyError as exc:
        LOGGER.error("build_agent_unknown_id agent_id=%s", agent_id)
        raise ValueError(f"Unknown agent_id={agent_id!r}.") from exc
    LOGGER.info("build_agent agent_id=%s seed=%s policy=%s", agent_id, seed, policy_cls.__name__)
    return policy_cls(seed=seed)


def get_agent_descriptor(agent_id: str) -> AgentDescriptor:
    """Return metadata for a known policy."""
    try:
        return AGENT_REGISTRY[agent_id].descriptor
    except KeyError as exc:
        raise ValueError(f"Unknown agent_id={agent_id!r}.") from exc


def serialize_agents() -> List[Dict[str, Any]]:
    """Serialize selectable agents for the frontend."""
    return [asdict(policy.descriptor) for policy in AGENT_REGISTRY.values()]
