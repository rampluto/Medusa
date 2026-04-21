"""Deterministic MEDUSA agent policies for the custom UI."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Type

try:
    from ..models import MedusaActionType
    from ..tasks import Task
except ImportError:
    try:
        from medusa_env.models import MedusaActionType
        from medusa_env.tasks import Task
    except ImportError:
        from models import MedusaActionType
        from tasks import Task


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
    ) -> MedusaActionType:
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
    ) -> MedusaActionType:
        return self._random.choice(list(MedusaActionType))


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
    ) -> MedusaActionType:
        return MedusaActionType.COMMIT


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
    ) -> MedusaActionType:
        if (state.is_stale_a or state.is_stale_b) and not state.did_sync_check:
            return MedusaActionType.SYNC_CHECK
        if (state.new_cols_a or state.new_cols_b) and not state.did_evolve_schema:
            return MedusaActionType.EVOLVE_SCHEMA
        if not state.did_prep_a:
            return MedusaActionType.PREP_KEYS_A
        if not state.did_prep_b:
            return MedusaActionType.PREP_KEYS_B
        if state.uniqueness_b < 1.0 and not state.did_dedup_b:
            return MedusaActionType.DEDUPLICATE_B
        if not state.did_join:
            return MedusaActionType.EXECUTE_JOIN_LEFT
        if not state.did_scd:
            return MedusaActionType.APPLY_SCD_2
        return MedusaActionType.COMMIT


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
    ) -> MedusaActionType:
        task_id = task.id if task is not None else None

        if self._needs_sync(task_id, state) and not state.did_sync_check:
            return MedusaActionType.SYNC_CHECK
        if self._needs_schema(task_id, state) and not state.did_evolve_schema:
            return MedusaActionType.EVOLVE_SCHEMA
        if not state.did_prep_a:
            return MedusaActionType.PREP_KEYS_A
        if not state.did_prep_b:
            return MedusaActionType.PREP_KEYS_B
        if self._needs_dedup(task_id, state) and not state.did_dedup_b:
            return MedusaActionType.DEDUPLICATE_B
        if not state.did_join:
            return self._preferred_join(task_id)
        if not state.did_scd:
            return self._preferred_scd(task_id)
        return MedusaActionType.COMMIT

    @staticmethod
    def _needs_sync(task_id: Optional[str], state: Any) -> bool:
        return task_id in {
            "full_medallion",
            "stale_sync_recovery",
            "stale_history_guard",
        } or state.is_stale_a or state.is_stale_b

    @staticmethod
    def _needs_schema(task_id: Optional[str], state: Any) -> bool:
        return task_id in {
            "schema_bootstrap",
            "drift_alignment",
            "schema_history_guard",
            "full_medallion",
        } or bool(state.new_cols_a or state.new_cols_b)

    @staticmethod
    def _needs_dedup(task_id: Optional[str], state: Any) -> bool:
        return task_id in {
            "dirty_integration",
            "dedup_guardrail",
            "full_medallion",
        } or state.uniqueness_b < 1.0

    @staticmethod
    def _preferred_join(task_id: Optional[str]) -> MedusaActionType:
        return MedusaActionType.EXECUTE_JOIN_LEFT

    @staticmethod
    def _preferred_scd(task_id: Optional[str]) -> MedusaActionType:
        if task_id == "snapshot_upsert":
            return MedusaActionType.APPLY_SCD_1
        return MedusaActionType.APPLY_SCD_2


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
