"""Custom MEDUSA API and no-session frontend routes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

try:
    from ..models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
    from ..tasks import TASKS, Task, score_episode
    from .agent_policies import (
        AGENT_REGISTRY,
        DEFAULT_AGENT_ID,
        build_agent,
        get_agent_descriptor,
        serialize_agents,
    )
    from .medusa_env import MedusaEnv
except ImportError:
    try:
        from medusa_env.models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
        from medusa_env.tasks import TASKS, Task, score_episode
        from medusa_env.server.agent_policies import (
            AGENT_REGISTRY,
            DEFAULT_AGENT_ID,
            build_agent,
            get_agent_descriptor,
            serialize_agents,
        )
        from medusa_env.server.medusa_env import MedusaEnv
    except ImportError:
        from models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
        from server.agent_policies import (
            AGENT_REGISTRY,
            DEFAULT_AGENT_ID,
            build_agent,
            get_agent_descriptor,
            serialize_agents,
        )
        from server.medusa_env import MedusaEnv
        from tasks import TASKS, Task, score_episode


FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
AVAILABLE_TABLES = (
    "bronze_a",
    "bronze_a_prepped",
    "bronze_b",
    "bronze_b_prepped",
    "joined",
    "silver",
    "quarantine",
)
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}

ACTION_METADATA: Dict[MedusaActionType, Dict[str, str]] = {
    MedusaActionType.SYNC_CHECK: {
        "category": "freshness",
        "label": "Sync Check",
        "description": "Verify how stale each Bronze source is before doing downstream work.",
    },
    MedusaActionType.EVOLVE_SCHEMA: {
        "category": "schema",
        "label": "Evolve Schema",
        "description": "Add drifted columns from Bronze A and B into Silver before materializing data.",
    },
    MedusaActionType.PREP_KEYS_A: {
        "category": "keys",
        "label": "Prep Keys A",
        "description": "Normalize and clean the join key on Source A.",
    },
    MedusaActionType.PREP_KEYS_B: {
        "category": "keys",
        "label": "Prep Keys B",
        "description": "Normalize and clean the join key on Source B.",
    },
    MedusaActionType.DEDUPLICATE_B: {
        "category": "keys",
        "label": "Deduplicate B",
        "description": "Remove duplicate dimension keys to avoid join explosions.",
    },
    MedusaActionType.EXECUTE_JOIN_INNER: {
        "category": "join",
        "label": "Inner Join",
        "description": "Join matched rows only; useful when unmatched facts can be dropped.",
    },
    MedusaActionType.EXECUTE_JOIN_LEFT: {
        "category": "join",
        "label": "Left Join",
        "description": "Preserve all fact rows and quarantine true orphans for auditability.",
    },
    MedusaActionType.EXECUTE_JOIN_ANTI: {
        "category": "join",
        "label": "Anti Join",
        "description": "Extract only fact rows with no match in the dimension.",
    },
    MedusaActionType.APPLY_SCD_1: {
        "category": "history",
        "label": "Apply SCD-1",
        "description": "Overwrite existing Silver records in snapshot-style loads.",
    },
    MedusaActionType.APPLY_SCD_2: {
        "category": "history",
        "label": "Apply SCD-2",
        "description": "Maintain history with valid_from / valid_to windows.",
    },
    MedusaActionType.COMMIT: {
        "category": "finalize",
        "label": "Commit",
        "description": "Finalize the episode and run the deterministic audit.",
    },
}

FEATURE_LABELS = [
    "time_delta_a_norm",
    "time_delta_b_norm",
    "is_stale_a",
    "is_stale_b",
    "null_ratio_key_a",
    "null_ratio_key_b",
    "uniqueness_a",
    "uniqueness_b",
    "match_rate",
    "new_cols_a_norm",
    "new_cols_b_norm",
    "schema_compat",
    "did_prep_a",
    "did_prep_b",
    "did_dedup_b",
    "step_frac",
]

FEATURE_DESCRIPTIONS = {
    "time_delta_a_norm": "Source A freshness signal, normalized against 48 hours.",
    "time_delta_b_norm": "Source B freshness signal, normalized against 48 hours.",
    "is_stale_a": "Whether Source A is stale relative to the configured threshold.",
    "is_stale_b": "Whether Source B is stale relative to the configured threshold.",
    "null_ratio_key_a": "Fraction of null join keys currently present in Source A.",
    "null_ratio_key_b": "Fraction of null join keys currently present in Source B.",
    "uniqueness_a": "Key uniqueness ratio in Source A.",
    "uniqueness_b": "Key uniqueness ratio in Source B.",
    "match_rate": "How many Source A keys are currently resolvable in Source B.",
    "new_cols_a_norm": "Normalized count of drifted columns arriving from Source A.",
    "new_cols_b_norm": "Normalized count of drifted columns arriving from Source B.",
    "schema_compat": "Schema compatibility score for the active join path.",
    "did_prep_a": "Whether key preparation has been applied to Source A.",
    "did_prep_b": "Whether key preparation has been applied to Source B.",
    "did_dedup_b": "Whether Source B has been deduplicated.",
    "step_frac": "Episode progress as a fraction of the step budget.",
}


class TraceRequest(BaseModel):
    """Base request for no-session trace replay."""

    task_id: Optional[str] = None
    seed: Optional[int] = None
    actions: List[MedusaAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> "TraceRequest":
        if self.task_id is None and self.seed is None:
            raise ValueError("Provide either task_id or seed.")
        if self.task_id is not None:
            task = TASKS.get(self.task_id)
            if task is None:
                raise ValueError(f"Unknown task_id={self.task_id!r}.")
            if self.seed is not None and self.seed != task.seed:
                raise ValueError(
                    f"seed={self.seed} does not match task_id={self.task_id!r} "
                    f"(expected {task.seed})."
                )
        return self


class StepTraceRequest(TraceRequest):
    """Trace replay request that appends one more action."""

    next_action: MedusaAction


class ResetTraceRequest(BaseModel):
    """Explicit reset request for a deterministic task or seed."""

    task_id: Optional[str] = None
    seed: Optional[int] = None

    @model_validator(mode="after")
    def validate_source(self) -> "ResetTraceRequest":
        if self.task_id is None and self.seed is None:
            raise ValueError("Provide either task_id or seed.")
        if self.task_id is not None:
            task = TASKS.get(self.task_id)
            if task is None:
                raise ValueError(f"Unknown task_id={self.task_id!r}.")
            if self.seed is not None and self.seed != task.seed:
                raise ValueError(
                    f"seed={self.seed} does not match task_id={self.task_id!r} "
                    f"(expected {task.seed})."
                )
        return self


class TableRequest(TraceRequest):
    """Trace replay request for tabular previews."""

    table: Literal[
        "bronze_a",
        "bronze_a_prepped",
        "bronze_b",
        "bronze_b_prepped",
        "joined",
        "silver",
        "quarantine",
    ]
    page: int = 1
    page_size: int = 25


class AutoRunRequest(BaseModel):
    """Request for a fully automatic agent-driven replay."""

    agent_id: str = DEFAULT_AGENT_ID

    @model_validator(mode="after")
    def validate_agent(self) -> "AutoRunRequest":
        if self.agent_id not in AGENT_REGISTRY:
            raise ValueError(f"Unknown agent_id={self.agent_id!r}.")
        return self


def register_custom_routes(app: FastAPI) -> None:
    """Attach MEDUSA-specific API and frontend routes to the OpenEnv app."""
    router = APIRouter(prefix="/api", tags=["Medusa UI"])

    @router.get("/tasks")
    async def get_tasks() -> Dict[str, Any]:
        tasks = [asdict(task) for task in TASKS.values()]
        tasks.sort(key=lambda item: (DIFFICULTY_ORDER.get(item["difficulty"], 99), item["name"]))
        return {"tasks": tasks}

    @router.get("/tasks/{task_id}")
    async def get_task(task_id: str) -> Dict[str, Any]:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown task_id={task_id!r}.")
        return {"task": asdict(task)}

    @router.get("/action-space")
    async def get_action_space() -> Dict[str, Any]:
        return {
            "actions": [
                {
                    "action": action.value,
                    **ACTION_METADATA[action],
                }
                for action in MedusaActionType
            ]
        }

    @router.get("/agents")
    async def get_agents() -> Dict[str, Any]:
        return {
            "default_agent_id": DEFAULT_AGENT_ID,
            "agents": serialize_agents(),
        }

    @router.post("/run/preview")
    async def preview_run(request: TraceRequest) -> Dict[str, Any]:
        return _with_replay(request, _build_run_payload)

    @router.post("/run/reset/{task_id}")
    async def reset_run(task_id: str) -> Dict[str, Any]:
        return _with_replay(TraceRequest(task_id=task_id, actions=[]), _build_run_payload)

    @router.post("/run/autorun/{task_id}")
    async def autorun(task_id: str, request: AutoRunRequest) -> Dict[str, Any]:
        task = TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown task_id={task_id!r}.")

        reset_payload = await _post_local_json(
            app,
            f"/api/run/reset/{task_id}",
            {},
        )
        seed = reset_payload["seed"]
        policy = build_agent(request.agent_id, seed=seed)
        actions: List[Dict[str, Any]] = []
        current_payload = reset_payload
        agent_steps = 0

        while not current_payload["summary"]["done"]:
            state = MedusaState(**current_payload["state"])
            observation = MedusaObservation(**current_payload["observation"])
            next_action_type = policy.select_action(task=task, state=state, observation=observation)
            next_action = {"action": next_action_type.value, "params": {}}
            current_payload = await _post_local_json(
                app,
                "/api/run/step",
                {
                    "task_id": task_id,
                    "actions": actions,
                    "next_action": next_action,
                },
            )
            actions = [*actions, next_action]
            agent_steps += 1

        current_payload["agent"] = _serialize_agent(request.agent_id)
        current_payload["auto_run"] = {
            "agent_steps": agent_steps,
            "completed": bool(current_payload["summary"]["done"]),
            "terminated_early": False,
        }
        return current_payload

    @router.post("/run/step")
    async def step_run(request: StepTraceRequest) -> Dict[str, Any]:
        replay_request = TraceRequest(
            task_id=request.task_id,
            seed=request.seed,
            actions=[*request.actions, request.next_action],
        )
        return _with_replay(replay_request, _build_run_payload)

    @router.post("/run/tables")
    async def get_table(request: TableRequest) -> Dict[str, Any]:
        return _with_replay(
            request,
            lambda env, task, seed, observation, actions: _build_table_payload(
                env,
                task,
                seed,
                observation,
                actions,
                request,
            ),
        )

    @router.post("/run/timeline")
    async def get_timeline(request: TraceRequest) -> Dict[str, Any]:
        return _with_replay(request, _build_timeline_payload)

    @router.post("/run/analysis")
    async def get_analysis(request: TraceRequest) -> Dict[str, Any]:
        return _with_replay(request, _build_analysis_payload)

    @router.post("/run/feature-vector")
    async def get_feature_vector(request: TraceRequest) -> Dict[str, Any]:
        return _with_replay(request, _build_feature_payload)

    @router.post("/run/grader")
    async def get_grader(request: TraceRequest) -> Dict[str, Any]:
        return _with_replay(request, _build_grader_payload)

    @router.post("/run/evaluate")
    async def evaluate_run(request: TraceRequest) -> Dict[str, Any]:
        return _with_replay(request, _build_evaluation_payload)

    app.include_router(router)

    if FRONTEND_DIR.exists():
        mount_paths = {getattr(route, "path", None) for route in app.routes}
        if "/medusa-static" not in mount_paths:
            app.mount("/medusa-static", StaticFiles(directory=FRONTEND_DIR), name="medusa-static")

        @app.get("/medusa", include_in_schema=False)
        async def medusa_frontend() -> RedirectResponse:
            return RedirectResponse(url="/medusa/studio", status_code=307)

        @app.get("/medusa/tasks", include_in_schema=False)
        async def medusa_tasks_page() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "tasks.html")

        @app.get("/medusa/studio", include_in_schema=False)
        async def medusa_studio_page() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "studio.html")

        @app.get("/medusa/audit", include_in_schema=False)
        async def medusa_audit_page() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "audit.html")


def _with_replay(
    request: TraceRequest,
    builder: Any,
) -> Dict[str, Any]:
    task = TASKS.get(request.task_id) if request.task_id else None
    seed = task.seed if task is not None else request.seed
    assert seed is not None
    env = MedusaEnv()
    try:
        observation = env.reset(seed=seed)
        for action in request.actions:
            observation = env.step(action)
        return builder(env, task, seed, observation, request.actions)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


async def _post_local_json(app: FastAPI, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://medusa-local") as client:
        response = await client.post(path, json=payload)
    data = response.json()
    if not response.is_success:
        detail = data.get("detail", response.text)
        raise HTTPException(status_code=response.status_code, detail=detail)
    return data


def _serialize_task(task: Optional[Task]) -> Optional[Dict[str, Any]]:
    return asdict(task) if task is not None else None


def _serialize_action(action: MedusaAction) -> Dict[str, Any]:
    return {
        "action": action.action.value,
        "params": action.params,
    }


def _serialize_agent(agent_id: str) -> Dict[str, Any]:
    return asdict(get_agent_descriptor(agent_id))


def _serialize_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "action": action["action"],
            "params": action.get("params", {}),
            "category": ACTION_METADATA[MedusaActionType(action["action"])]["category"],
        }
        for action in actions
    ]


def _table_snapshot(env: MedusaEnv) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for name in AVAILABLE_TABLES:
        df = getattr(env._tables, name)
        snapshot[name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }
    return snapshot


def _build_run_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
) -> Dict[str, Any]:
    state = env.state
    scenario = env._scenario
    observation_payload = observation.model_dump()
    return {
        "task": _serialize_task(task),
        "seed": seed,
        "scenario": {
            "id": getattr(scenario, "id", None),
            "description": getattr(scenario, "description", ""),
            "join_key": getattr(scenario, "join_key", None),
            "tracked_cols": getattr(scenario, "tracked_cols", []),
        },
        "actions": _serialize_actions([_serialize_action(action) for action in actions]),
        "action_count": len(actions),
        "available_tables": list(AVAILABLE_TABLES),
        "observation": observation_payload,
        "state": state.model_dump(),
        "summary": {
            "stage": state.stage,
            "done": observation_payload["done"],
            "step_idx": state.step_idx,
            "max_steps": state.max_steps,
            "cumulative_reward": state.cumulative_reward,
            "match_rate": state.match_rate,
            "join_type": state.join_type,
            "silver_row_count": state.silver_row_count,
            "quarantine_row_count": state.quarantine_row_count,
            "explosion_detected": state.explosion_detected,
            "did_sync_check": state.did_sync_check,
            "did_evolve_schema": state.did_evolve_schema,
            "did_prep_a": state.did_prep_a,
            "did_prep_b": state.did_prep_b,
            "did_dedup_b": state.did_dedup_b,
            "did_join": state.did_join,
            "did_scd": state.did_scd,
            "grader_passed": state.grader_passed,
        },
        "tables": _table_snapshot(env),
    }


def _build_table_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
    request: TableRequest,
) -> Dict[str, Any]:
    df = getattr(env._tables, request.table)
    table_payload = _serialize_dataframe(df, request.page, request.page_size)
    return {
        "task": _serialize_task(task),
        "seed": seed,
        "table": request.table,
        **table_payload,
    }


def _build_timeline_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
) -> Dict[str, Any]:
    return {
        "task": _serialize_task(task),
        "seed": seed,
        "timeline": [
            {
                **entry,
                "timestamp": round(float(entry["timestamp"]), 3),
            }
            for entry in env._tables.governance_log
        ],
    }


def _build_feature_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
) -> Dict[str, Any]:
    features = observation.features
    explained = []
    for index, label in enumerate(FEATURE_LABELS):
        value = features[index] if index < len(features) else None
        explained.append(
            {
                "index": index,
                "label": label,
                "value": value,
                "description": FEATURE_DESCRIPTIONS[label],
            }
        )
    return {
        "task": _serialize_task(task),
        "seed": seed,
        "features": explained,
    }


def _build_analysis_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
) -> Dict[str, Any]:
    state = env.state
    blockers = _commit_blockers(state)
    suggestions = _suggest_actions(state)
    return {
        "task": _serialize_task(task),
        "seed": seed,
        "analysis": {
            "freshness": {
                "source_a_hours": state.time_delta_a,
                "source_b_hours": state.time_delta_b,
                "source_a_stale": state.is_stale_a,
                "source_b_stale": state.is_stale_b,
                "sync_checked": state.did_sync_check,
            },
            "key_health": {
                "null_ratio_a": state.null_ratio_key_a,
                "null_ratio_b": state.null_ratio_key_b,
                "uniqueness_a": state.uniqueness_a,
                "uniqueness_b": state.uniqueness_b,
                "prepped_a": state.did_prep_a,
                "prepped_b": state.did_prep_b,
                "deduped_b": state.did_dedup_b,
            },
            "join": {
                "did_join": state.did_join,
                "join_type": state.join_type,
                "match_rate": state.match_rate,
                "join_rows": state.join_row_count,
                "quarantine_rows": state.quarantine_row_count,
                "explosion_detected": state.explosion_detected,
            },
            "schema": {
                "did_evolve_schema": state.did_evolve_schema,
                "new_cols_a": state.new_cols_a,
                "new_cols_b": state.new_cols_b,
            },
            "history": {
                "did_scd": state.did_scd,
                "scd_type": state.scd_type,
                "silver_rows": state.silver_row_count,
            },
            "commit": {
                "ready": not blockers,
                "blockers": blockers,
                "suggested_actions": suggestions,
            },
        },
    }


def _build_grader_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
) -> Dict[str, Any]:
    state = env.state
    task_result = score_episode(task.id, state, env._tables) if task is not None else None
    return {
        "task": _serialize_task(task),
        "seed": seed,
        "committed": state.stage == "committed",
        "ready_for_commit": not _commit_blockers(state),
        "blockers": _commit_blockers(state),
        "grader": {
            "passed": state.grader_passed,
            "report": state.grader_report,
            "lines": state.grader_report.splitlines() if state.grader_report else [],
        },
        "evaluation": asdict(task_result) if task_result is not None else None,
    }


def _build_evaluation_payload(
    env: MedusaEnv,
    task: Optional[Task],
    seed: int,
    observation: Any,
    actions: List[MedusaAction],
) -> Dict[str, Any]:
    resolved_task = task or _task_from_seed(seed)
    if resolved_task is None:
        raise HTTPException(
            status_code=422,
            detail="Evaluation requires a known task_id or a seed mapped to a task.",
        )
    task_result = score_episode(resolved_task.id, env.state, env._tables)
    return {
        "task": asdict(resolved_task),
        "seed": seed,
        "evaluation": asdict(task_result),
    }

def _task_from_seed(seed: int) -> Optional[Task]:
    return next((task for task in TASKS.values() if task.seed == seed), None)


def _serialize_dataframe(df: pd.DataFrame, page: int, page_size: int) -> Dict[str, Any]:
    if page < 1:
        raise HTTPException(status_code=422, detail="page must be >= 1.")
    if page_size < 1:
        raise HTTPException(status_code=422, detail="page_size must be >= 1.")

    total_rows = len(df)
    bounded_page_size = min(page_size, 100)
    total_pages = max(math.ceil(total_rows / bounded_page_size), 1)
    bounded_page = min(page, total_pages)
    start = (bounded_page - 1) * bounded_page_size
    end = start + bounded_page_size
    page_df = df.iloc[start:end].copy()

    rows = json.loads(page_df.to_json(orient="records", date_format="iso"))
    return {
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "rows": rows,
        "page": bounded_page,
        "page_size": bounded_page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
    }


def _commit_blockers(state: Any) -> List[str]:
    blockers: List[str] = []
    if (state.is_stale_a or state.is_stale_b) and not state.did_sync_check:
        blockers.append("Run SYNC_CHECK before processing stale sources.")
    if (state.new_cols_a or state.new_cols_b) and not state.did_evolve_schema:
        blockers.append("Run EVOLVE_SCHEMA before committing schema drift.")
    if not state.did_prep_a:
        blockers.append("Prepare Source A keys before joining.")
    if not state.did_prep_b:
        blockers.append("Prepare Source B keys before joining.")
    if state.uniqueness_b < 1.0 and not state.did_dedup_b:
        blockers.append("Deduplicate Source B before joining to avoid row explosion.")
    if not state.did_join:
        blockers.append("Execute a join before committing.")
    if not state.did_scd:
        blockers.append("Apply an SCD merge before committing.")
    if state.explosion_detected:
        blockers.append("Current join path exploded row counts; revise before committing.")
    return blockers


def _suggest_actions(state: Any) -> List[str]:
    suggestions: List[str] = []
    if (state.is_stale_a or state.is_stale_b) and not state.did_sync_check:
        suggestions.append(MedusaActionType.SYNC_CHECK.value)
    if (state.new_cols_a or state.new_cols_b) and not state.did_evolve_schema:
        suggestions.append(MedusaActionType.EVOLVE_SCHEMA.value)
    if not state.did_prep_a:
        suggestions.append(MedusaActionType.PREP_KEYS_A.value)
    if not state.did_prep_b:
        suggestions.append(MedusaActionType.PREP_KEYS_B.value)
    if state.uniqueness_b < 1.0 and not state.did_dedup_b:
        suggestions.append(MedusaActionType.DEDUPLICATE_B.value)
    if not state.did_join:
        suggestions.append(MedusaActionType.EXECUTE_JOIN_LEFT.value)
    elif not state.did_scd:
        suggestions.append(MedusaActionType.APPLY_SCD_2.value)
    else:
        suggestions.append(MedusaActionType.COMMIT.value)
    return suggestions
