"""Deterministic MEDUSA agent policies for the custom UI."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
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
        candidates = [
            MedusaAction(action=MedusaActionType.PROFILE_TABLE.value, params={"table": "bronze"}),
            MedusaAction(action=MedusaActionType.EXECUTE_MERGE.value, params={}),
            MedusaAction(action=MedusaActionType.COMMIT_DAY.value, params={}),
        ]
        return self._random.choice(candidates)


class AlwaysCommitPolicy(BaseAgentPolicy):
    """Immediate-commit lower bound."""

    descriptor = AgentDescriptor(
        id="always_commit",
        name="Always Commit",
        description="Terminates immediately to expose pre-commit audit failures and guardrails.",
        family="baseline",
        strengths=["Fastest failure case", "Useful for grader demos"],
    )

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        return MedusaAction(action=MedusaActionType.COMMIT_DAY.value, params={})


class GovernanceFirstPolicy(BaseAgentPolicy):
    """A conservative checklist-driven agent."""

    descriptor = AgentDescriptor(
        id="governance_first",
        name="Governance First",
        description=(
            "Clears freshness, schema drift, key quality, and dedup needs before taking the "
            "audited left-join plus SCD-2 path."
        ),
        family="heuristic",
        strengths=["Safe default", "Strong on stale and drift-heavy tasks"],
    )

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        return _next_v4_action(state)


class TaskAwarePlannerPolicy(BaseAgentPolicy):
    """A higher-scoring deterministic planner that adapts to task intent."""

    descriptor = AgentDescriptor(
        id="task_aware",
        name="Task-Aware Planner",
        description=(
            "Uses task intent plus live pipeline state to choose the safest audited path, "
            "including SCD-1 for snapshot tasks and SCD-2 for history-preserving tasks."
        ),
        family="heuristic",
        strengths=["Best general benchmark fit", "Adapts snapshot vs history paths"],
        default=True,
    )

    def select_action(
        self,
        *,
        task: Optional[Task],
        state: Any,
        observation: Any,
    ) -> PolicyAction:
        return _next_v4_action(state)


def _next_v4_action(state: Any) -> MedusaAction:
    current_day = getattr(state, "current_day", 1)
    anomalies = list(getattr(state, "day_anomalies", {}).get(current_day, []))

    if not getattr(state, "profiled_tables_today", {}).get("bronze"):
        return MedusaAction(action=MedusaActionType.PROFILE_TABLE.value, params={"table": "bronze"})

    for column, operation in anomalies:
        if operation in {"strip", "cast", "fill_zero"}:
            if (column, operation) not in getattr(state, "cleaned_columns_today", set()):
                return MedusaAction(
                    action=MedusaActionType.CLEAN_COLUMN.value,
                    params={"table": "bronze", "col": column, "op": operation},
                )
        elif operation == "deduplicate" and not getattr(state, "did_dedup_today", False):
            return MedusaAction(
                action=MedusaActionType.DEDUPLICATE.value,
                params={"table": "bronze", "key": column},
            )
        elif operation == "evolve" and column not in getattr(state, "current_contract_columns", []):
            return MedusaAction(
                action=MedusaActionType.EVOLVE_SILVER_SCHEMA.value,
                params={"column": column},
            )
        elif operation == "quarantine" and getattr(state, "day28_quarantine_rows", 0) == 0:
            return MedusaAction(
                action=MedusaActionType.QUARANTINE_ROWS.value,
                params={"table": "bronze", "condition": f"{column} IS NULL"},
            )

    if getattr(state, "uniqueness_b", 1.0) < 1.0 and not getattr(state, "did_dedup_today", False):
        return MedusaAction(
            action=MedusaActionType.DEDUPLICATE.value,
            params={"table": "bronze", "key": "user_id"},
        )

    if not getattr(state, "did_merge_today", False):
        return MedusaAction(action=MedusaActionType.EXECUTE_MERGE.value, params={})

    return MedusaAction(action=MedusaActionType.COMMIT_DAY.value, params={})


AGENT_REGISTRY: Dict[str, Type[BaseAgentPolicy]] = {
    policy.descriptor.id: policy
    for policy in (
        RandomPolicy,
        AlwaysCommitPolicy,
        GovernanceFirstPolicy,
        TaskAwarePlannerPolicy,
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
        raise ValueError(f"Unknown agent_id={agent_id!r}.") from exc
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
