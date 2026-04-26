"""Custom MEDUSA API and no-session frontend routes."""

from __future__ import annotations

import importlib
import json
import logging
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

# Ensure the repo root is on sys.path so data_quality_score can be imported
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from ..models import MedusaAction, MedusaActionType
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
        from medusa_env.models import MedusaAction, MedusaActionType
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
        from models import MedusaAction, MedusaActionType
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
    "daily_raw",
    "daily_cleaned",
    "bronze_a",
    "bronze_a_prepped",
    "bronze_b",
    "bronze_b_prepped",
    "joined",
    "silver",
    "quarantine",
)
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
LOGGER = logging.getLogger(__name__)

FEATURE_LABELS = [
    "day_frac",
    "step_frac",
    "silver_size_norm",
    "quarantine_size_norm",
    "retry_norm",
    "anomaly_count_norm",
    "cleaned_count_norm",
    "time_delta_a_norm",
    "time_delta_b_norm",
    "is_stale_a",
    "is_stale_b",
    "null_ratio_key_a",
    "null_ratio_key_b",
    "match_rate",
    "did_dedup_today",
    "did_evolve_schema",
]

FEATURE_DESCRIPTIONS = {
    "day_frac": "Episode day progress across the 30-day gauntlet.",
    "step_frac": "Progress through the current day's 10-step action budget.",
    "silver_size_norm": "Cumulative Silver row count normalized against the environment scale.",
    "quarantine_size_norm": "Current quarantine row count normalized against the environment scale.",
    "retry_norm": "Current blocked-action retry pressure for the day.",
    "anomaly_count_norm": "Number of known anomalies scheduled for the current day.",
    "cleaned_count_norm": "How many column-cleaning operations have been completed today.",
    "time_delta_a_norm": "Source A freshness signal, normalized against 48 hours.",
    "time_delta_b_norm": "Source B freshness signal, normalized against 48 hours.",
    "is_stale_a": "Whether Source A is stale relative to the configured threshold.",
    "is_stale_b": "Whether Source B is stale relative to the configured threshold.",
    "null_ratio_key_a": "Fraction of null join keys currently present in Source A.",
    "null_ratio_key_b": "Fraction of null join keys currently present in Source B.",
    "match_rate": "Merge or legacy join match signal.",
    "did_dedup_today": "Whether deduplication has been performed for the current day.",
    "did_evolve_schema": "Whether schema evolution has been performed in this episode.",
}


class TraceRequest(BaseModel):
    """Base request for no-session trace replay."""

    task_id: Optional[str] = None
    seed: Optional[int] = None
    actions: List[MedusaAction] = Field(default_factory=list)

    @field_validator("actions", mode="before")
    @classmethod
    def normalize_actions(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [_normalize_action_payload(action) for action in value]

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

    @field_validator("next_action", mode="before")
    @classmethod
    def normalize_next_action(cls, value: Any) -> Any:
        return _normalize_action_payload(value)


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
        "daily_raw",
        "daily_cleaned",
        "joined",
        "silver",
        "quarantine",
    ]
    page: int = 1
    page_size: int = 25


def _normalize_action_payload(action: Any) -> Any:
    if isinstance(action, dict):
        return {
            "action": action.get("action"),
            "params": action.get("params") or {},
        }
    return action


class AutoRunRequest(BaseModel):
    """Request for a fully automatic agent-driven replay."""

    agent_id: str = DEFAULT_AGENT_ID

    @model_validator(mode="after")
    def validate_agent(self) -> "AutoRunRequest":
        if self.agent_id not in AGENT_REGISTRY:
            raise ValueError(f"Unknown agent_id={self.agent_id!r}.")
        return self


def _detect_cleaning_anomalies(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Infer a day-1 style anomaly checklist directly from an uploaded CSV.

    Emits raw-op labels (``whitespace``/``type_mixed``/``fill_null``/``quarantine``/
    ``deduplicate``) that heuristic/checklist-driven policies already know how to
    translate into concrete MEDUSA actions.
    """
    anomalies: List[Tuple[str, str]] = []
    if df is None or df.empty:
        return anomalies

    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            non_null = series.dropna().astype(str)
            if len(non_null) > 0 and (non_null != non_null.str.strip()).any():
                anomalies.append((col, "whitespace"))
            if len(non_null) > 0 and not pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(non_null, errors="coerce")
                hit_rate = float(numeric.notna().mean()) if len(numeric) else 0.0
                if 0.2 < hit_rate < 1.0:
                    anomalies.append((col, "type_mixed"))
        if pd.api.types.is_numeric_dtype(series) and series.isna().any():
            anomalies.append((col, "fill_null"))

    null_keyish = [
        col
        for col in df.columns
        if ("id" in col.lower() or "key" in col.lower()) and df[col].isna().any()
    ]
    for col in null_keyish:
        anomalies.append((col, "quarantine"))

    if df.duplicated().any():
        pk = df.columns[0] if len(df.columns) else ""
        anomalies.append((pk, "deduplicate"))

    return anomalies


def _build_cleaning_state(df: pd.DataFrame) -> SimpleNamespace:
    detected = _detect_cleaning_anomalies(df)
    has_duplicates = bool(df.duplicated().any()) if not df.empty else False
    return SimpleNamespace(
        current_day=1,
        day_anomalies={1: detected},
        cleaned_columns_today=[],
        profiled_tables_today={},
        did_dedup_today=False,
        did_evolve_schema=False,
        current_contract_columns=list(df.columns),
        day28_quarantine_rows=0,
        uniqueness_b=0.5 if has_duplicates else 1.0,
        did_merge_today=False,
    )
def _apply_cleaning_action(
    df: pd.DataFrame, state: SimpleNamespace, action: MedusaAction
) -> Tuple[pd.DataFrame, str]:
    LOGGER.debug("clean_apply_action_start action=%s params=%s", action.action, action.params)
    try:
        action_type = MedusaActionType(action.action)
    except ValueError:
        LOGGER.warning("clean_apply_action_unsupported action=%s", action.action)
        return df, f"Ignored unsupported action '{action.action}'."

    if action_type == MedusaActionType.PROFILE_TABLE:
        state.profiled_tables_today["bronze"] = 1
        return df, f"Profiled uploaded table ({len(df)} row(s), {len(df.columns)} column(s))."

    if action_type == MedusaActionType.CLEAN_COLUMN:
        column = str(action.params.get("col") or "")
        operation = str(action.params.get("op") or "")
        if not column or column not in df.columns or not operation:
            return df, "Skipped CLEAN_COLUMN because params were incomplete or invalid."
        if operation == "strip":
            df[column] = df[column].astype(str).str.strip()
            df[column] = df[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        elif operation == "cast":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif operation == "fill_zero":
            as_numeric = pd.to_numeric(df[column], errors="coerce")
            if as_numeric.notna().any():
                df[column] = as_numeric.fillna(0)
        else:
            return df, f"Skipped CLEAN_COLUMN because op '{operation}' is unsupported."
        state.cleaned_columns_today.append((column, operation))
        return df, f"Applied CLEAN_COLUMN on '{column}' with op '{operation}'."

    if action_type == MedusaActionType.DEDUPLICATE:
        key = str(action.params.get("key") or "")
        before = len(df)
        if key and key in df.columns:
            deduped = df.drop_duplicates(subset=[key]).reset_index(drop=True)
        else:
            deduped = df.drop_duplicates().reset_index(drop=True)
        removed = int(before - len(deduped))
        state.did_dedup_today = True
        return deduped, f"Deduplicated rows using key '{key or 'all_columns'}' ({removed} removed)."

    if action_type == MedusaActionType.QUARANTINE_ROWS:
        condition = str(action.params.get("condition") or "")
        target_column = condition.split(" IS NULL")[0].strip() if " IS NULL" in condition else ""
        if target_column and target_column in df.columns:
            before = len(df)
            filtered = df[df[target_column].notna()].reset_index(drop=True)
            removed = int(before - len(filtered))
            state.day28_quarantine_rows += removed
            return filtered, f"Quarantined {removed} row(s) where '{target_column}' is null."
        return df, f"Skipped quarantine; unsupported condition '{condition}'."

    if action_type == MedusaActionType.EVOLVE_SILVER_SCHEMA:
        candidate = str(action.params.get("column") or "")
        if candidate and candidate in df.columns and candidate not in state.current_contract_columns:
            state.current_contract_columns.append(candidate)
            return df, f"Evolved schema to include '{candidate}'."
        return df, "Schema already compatible; no evolution needed."

    if action_type == MedusaActionType.EXECUTE_MERGE:
        state.did_merge_today = True
        return df, "Executed merge stage for cleaned dataframe."

    if action_type == MedusaActionType.COMMIT_DAY:
        return df, "Committed cleaning run."

    return df, f"No cleaning executor mapped for action '{action_type.value}'."


def _clean_dataframe_with_agent(df: pd.DataFrame, agent_id: str) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    cleaned = df.copy()
    state = _build_cleaning_state(cleaned)
    policy = build_agent(agent_id, seed=None)
    observation = SimpleNamespace(done=False, features=[], metrics={}, metadata={})
    trace: List[Dict[str, str]] = []
    # Scale the budget with file complexity: profile + up to 2 ops per column + dedup + merge + commit
    max_steps = max(10, len(cleaned.columns) * 2 + 5)
    LOGGER.info(
        "clean_run_start agent_id=%s input_rows=%s input_cols=%s max_steps=%s",
        agent_id,
        len(cleaned),
        len(cleaned.columns),
        max_steps,
    )

    for step_idx in range(max_steps):
        selected_action = policy.select_action(task=None, state=state, observation=observation)
        if isinstance(selected_action, MedusaAction):
            next_action = selected_action
        else:
            next_action = MedusaAction(action=selected_action.value, params={})
        LOGGER.info(
            "clean_policy_action step=%s agent_id=%s action=%s params=%s",
            step_idx + 1,
            agent_id,
            next_action.action,
            next_action.params,
        )
        cleaned, result_message = _apply_cleaning_action(cleaned, state, next_action)
        LOGGER.info(
            "clean_action_result step=%s action=%s rows=%s message=%s",
            step_idx + 1,
            next_action.action,
            len(cleaned),
            result_message,
        )
        trace.append(
            {
                "agent_id": agent_id,
                "action": (
                    f"{next_action.action} {json.dumps(next_action.params, sort_keys=True)}"
                    f" -> {result_message}"
                ),
            }
        )
        if next_action.action == MedusaActionType.COMMIT_DAY.value:
            LOGGER.info("clean_run_commit_reached step=%s", step_idx + 1)
            break

    if not trace:
        trace.append({"agent_id": agent_id, "action": "Agent produced no actions for this file."})
        LOGGER.warning("clean_run_no_actions agent_id=%s", agent_id)

    LOGGER.info(
        "clean_run_done agent_id=%s output_rows=%s trace_steps=%s",
        agent_id,
        len(cleaned),
        len(trace),
    )
    return cleaned.reset_index(drop=True), trace


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

        seed = task.seed
        LOGGER.info("autorun_start task_id=%s agent_id=%s seed=%s", task_id, request.agent_id, seed)
        try:
            # Pass seed=None so stochastic policies (e.g. RandomPolicy) are truly
            # non-deterministic across runs. The task seed is only for the env reset.
            policy = build_agent(request.agent_id, seed=None)
        except (RuntimeError, ValueError) as exc:
            LOGGER.exception("autorun_build_agent_failed task_id=%s agent_id=%s", task_id, request.agent_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        actions: List[MedusaAction] = []
        agent_steps = 0
        max_agent_steps = 300

        env = MedusaEnv()
        try:
            observation = env.reset(seed=seed)

            while not observation.done and agent_steps < max_agent_steps:
                selected_action = policy.select_action(
                    task=task,
                    state=env.state,
                    observation=observation,
                )
                if isinstance(selected_action, MedusaAction):
                    next_action = selected_action
                else:
                    next_action = MedusaAction(action=selected_action.value, params={})
                LOGGER.debug(
                    "autorun_step task_id=%s step=%s action=%s params=%s",
                    task_id,
                    agent_steps + 1,
                    next_action.action,
                    next_action.params,
                )

                observation = env.step(next_action)
                actions.append(next_action)
                agent_steps += 1

            payload = _build_run_payload(env, task, seed, observation, actions)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        terminated_early = not payload["summary"]["done"]
        payload["agent"] = _serialize_agent(request.agent_id)
        payload["auto_run"] = {
            "agent_steps": agent_steps,
            "completed": bool(payload["summary"]["done"]),
            "terminated_early": terminated_early,
            "max_agent_steps": max_agent_steps,
        }
        LOGGER.info(
            "autorun_done task_id=%s agent_id=%s steps=%s completed=%s terminated_early=%s",
            task_id,
            request.agent_id,
            agent_steps,
            bool(payload["summary"]["done"]),
            terminated_early,
        )
        return payload

    @router.post("/run/clean-dataframe")
    async def clean_dataframe(
        agent_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> Dict[str, Any]:
        if agent_id not in AGENT_REGISTRY:
            raise HTTPException(status_code=422, detail=f"Unknown agent_id={agent_id!r}.")
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=422, detail="Upload a CSV file.")
        LOGGER.info("clean_endpoint_start agent_id=%s filename=%s", agent_id, file.filename)

        raw_bytes = await file.read()
        try:
            source_df = pd.read_csv(pd.io.common.BytesIO(raw_bytes))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("clean_endpoint_parse_failed agent_id=%s filename=%s", agent_id, file.filename)
            raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

        try:
            cleaned_df, action_trace = _clean_dataframe_with_agent(source_df, agent_id)
        except (RuntimeError, ValueError) as exc:
            LOGGER.exception("clean_endpoint_agent_failed agent_id=%s filename=%s", agent_id, file.filename)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        output_csv = cleaned_df.to_csv(index=False)
        source_name = Path(file.filename).stem
        LOGGER.info(
            "clean_endpoint_done agent_id=%s filename=%s input_rows=%s output_rows=%s trace_steps=%s",
            agent_id,
            file.filename,
            len(source_df),
            len(cleaned_df),
            len(action_trace),
        )
        return {
            "agent_id": agent_id,
            "input_rows": int(len(source_df)),
            "output_rows": int(len(cleaned_df)),
            "output_filename": f"{source_name}_cleaned_{agent_id}.csv",
            "cleaned_csv": output_csv,
            "action_trace": action_trace,
        }

    @router.post("/run/score-dataframes")
    async def score_dataframes_endpoint(
        source: UploadFile = File(...),
        cleaned: UploadFile = File(None),
    ) -> Dict[str, Any]:
        """Score source and (optionally) cleaned CSV files with data_quality_score."""
        try:
            dqs = importlib.import_module("data_quality_score")
        except ModuleNotFoundError as exc:
            LOGGER.exception("score_dataframes_module_missing")
            raise HTTPException(status_code=501, detail="data_quality_score module not found.") from exc

        async def _score_upload(upload: UploadFile) -> Dict[str, Any]:
            raw = await upload.read()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = Path(tmp.name)
            try:
                return dqs.score_csv(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

        source_result = await _score_upload(source)
        cleaned_result = await _score_upload(cleaned) if cleaned is not None else None

        def _extract(result: Dict[str, Any]) -> Dict[str, Any]:
            d = result.get("details", {})
            return {
                "score": result.get("score", 0.0),
                "passed": result.get("passed", False),
                "rows": d.get("rows", 0),
                "columns": d.get("columns", 0),
                "missing_cells": d.get("completeness", {}).get("missing_cells", 0),
                "missing_ratio": d.get("completeness", {}).get("missing_ratio", 0.0),
                "nan_values": d.get("numeric_sanity", {}).get("nan_values", 0),
                "null_values": d.get("numeric_sanity", {}).get("null_values", 0),
                "duplicate_rows": d.get("uniqueness", {}).get("duplicate_rows", 0),
                "duplicate_ratio": d.get("uniqueness", {}).get("duplicate_ratio", 0.0),
                "duplicate_column_names": d.get("column_quality", {}).get("duplicate_column_names", 0),
                "duplicate_column_groups": d.get("column_quality", {}).get("duplicate_column_groups", {}),
                "dirty_string_cells": d.get("string_cleanliness", {}).get("dirty_string_cells", 0),
                "dirty_string_ratio": d.get("string_cleanliness", {}).get("dirty_string_ratio", 0.0),
                "bad_numeric_cells": d.get("numeric_sanity", {}).get("bad_numeric_cells", 0),
                "component_scores": result.get("component_scores", {}),
                "column_names": d.get("column_names", []),
            }

        return {
            "source": _extract(source_result),
            "cleaned": _extract(cleaned_result) if cleaned_result is not None else None,
        }

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


def _serialize_task(task: Optional[Task]) -> Optional[Dict[str, Any]]:
    return asdict(task) if task is not None else None


def _serialize_action(action: MedusaAction) -> Dict[str, Any]:
    action_name = action.action.value if hasattr(action.action, "value") else str(action.action)
    return {
        "action": action_name,
        "params": action.params,
    }


def _serialize_agent(agent_id: str) -> Dict[str, Any]:
    return asdict(get_agent_descriptor(agent_id))


def _serialize_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "action": action["action"],
            "params": action.get("params", {}),
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
                "contract_columns": state.current_contract_columns,
                "pending_columns": _pending_schema_columns(state),
            },
            "day": {
                "current_day": state.current_day,
                "step_count": state.step_count,
                "retry_count": state.retry_count,
                "anomalies": _today_anomalies(state),
                "cleaned_columns": sorted(list(state.cleaned_columns_today)),
                "profiled_tables": state.profiled_tables_today,
                "deduped_today": state.did_dedup_today,
                "merged_today": state.did_merge_today,
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
    if state.stage != "running":
        return blockers
    if not state.profiled_tables_today.get("bronze"):
        blockers.append("Profile today's Bronze batch before committing.")
    for column, operation in _today_anomalies(state):
        if operation in {"strip", "cast", "fill_zero"}:
            if (column, operation) not in state.cleaned_columns_today:
                blockers.append(f"Clean {column} with {operation} before committing Day {state.current_day}.")
        elif operation == "deduplicate" and not state.did_dedup_today:
            blockers.append("Deduplicate today's batch before committing.")
        elif operation == "evolve" and column not in state.current_contract_columns:
            blockers.append(f"Evolve the Silver schema for {column} before committing.")
        elif operation == "quarantine" and state.day28_quarantine_rows == 0:
            blockers.append(f"Quarantine rows matching {column} IS NULL before committing.")
    if state.uniqueness_b < 1.0 and not state.did_dedup_today:
        blockers.append("Deduplicate today's batch before merging non-unique keys.")
    if _needs_merge_today(state):
        blockers.append("Execute today's merge before committing.")
    return blockers


def _suggest_actions(state: Any) -> List[str]:
    suggestions: List[str] = []
    if not state.profiled_tables_today.get("bronze"):
        suggestions.append(MedusaActionType.PROFILE_TABLE.value)
        return suggestions
    for column, operation in _today_anomalies(state):
        if operation in {"strip", "cast", "fill_zero"} and (column, operation) not in state.cleaned_columns_today:
            suggestions.append(MedusaActionType.CLEAN_COLUMN.value)
            return suggestions
        if operation == "deduplicate" and not state.did_dedup_today:
            suggestions.append(MedusaActionType.DEDUPLICATE.value)
            return suggestions
        if operation == "evolve" and column not in state.current_contract_columns:
            suggestions.append(MedusaActionType.EVOLVE_SILVER_SCHEMA.value)
            return suggestions
        if operation == "quarantine" and state.day28_quarantine_rows == 0:
            suggestions.append(MedusaActionType.QUARANTINE_ROWS.value)
            return suggestions
    if state.uniqueness_b < 1.0 and not state.did_dedup_today:
        suggestions.append(MedusaActionType.DEDUPLICATE.value)
        return suggestions
    if _needs_merge_today(state):
        suggestions.append(MedusaActionType.EXECUTE_MERGE.value)
    else:
        suggestions.append(MedusaActionType.COMMIT_DAY.value)
    return suggestions


def _today_anomalies(state: Any) -> List[List[str]]:
    return [list(item) for item in state.day_anomalies.get(state.current_day, [])]


def _pending_schema_columns(state: Any) -> List[str]:
    return [
        column
        for column, operation in _today_anomalies(state)
        if operation == "evolve" and column not in state.current_contract_columns
    ]


def _needs_merge_today(state: Any) -> bool:
    return not state.did_merge_today
