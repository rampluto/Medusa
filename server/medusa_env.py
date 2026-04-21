"""MEDUSA — full environment implementation (v4.0).

Implements the 30-day gauntlet with 7 ETL tools, deterministic grader,
cumulative Silver layer, per-day data generation with trap days, and
BLOCK/retry mechanics as defined in the MEDUSA-Chronos v4.0 spec.

Also supports legacy Phase-1 single-episode mode for backward compat.
"""

from __future__ import annotations

import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from medusa_env.grader import Grader
    from medusa_env.models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
    from medusa_env.operators import (
        apply_scd,
        deduplicate,
        evolve_schema,
        execute_join,
        prep_keys,
        sync_check,
    )
    from medusa_env.rewards import RewardEngine
    from medusa_env.scenarios import DayDataGenerator, DayBatch, Scenario, ScenarioGenerator
    from medusa_env.tasks import TASKS, score_episode
except ImportError:
    from grader import Grader
    from models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
    from operators import (
        apply_scd,
        deduplicate,
        evolve_schema,
        execute_join,
        prep_keys,
        sync_check,
    )
    from rewards import RewardEngine
    from scenarios import DayDataGenerator, DayBatch, Scenario, ScenarioGenerator
    from tasks import TASKS, score_episode


# ---------------------------------------------------------------------------
# Internal episode tables
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeTables:
    """In-memory tables for one episode."""

    bronze_a: pd.DataFrame = field(default_factory=pd.DataFrame)
    bronze_a_prepped: pd.DataFrame = field(default_factory=pd.DataFrame)
    bronze_b: pd.DataFrame = field(default_factory=pd.DataFrame)
    bronze_b_prepped: pd.DataFrame = field(default_factory=pd.DataFrame)
    joined: pd.DataFrame = field(default_factory=pd.DataFrame)
    silver: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarantine: pd.DataFrame = field(default_factory=pd.DataFrame)
    governance_log: List[Dict[str, Any]] = field(default_factory=list)
    # v4.0: daily raw batch (the Bronze data for the current day)
    daily_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_cleaned: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

_MAX_TIME_DELTA = 48.0   # Normalisation ceiling (hours)
_MAX_COLS = 10.0          # Normalisation ceiling (new columns)


def _build_features(state: MedusaState) -> List[float]:
    """Build the 16-float normalised observation vector."""
    day_frac = min(state.current_day / 30.0, 1.0)
    step_frac = min(state.step_count / 10.0, 1.0)
    silver_norm = min(state.silver_row_count / 3000.0, 1.0)
    quarantine_norm = min(state.quarantine_row_count / 500.0, 1.0)
    retry_norm = min(state.retry_count / 3.0, 1.0)

    # Anomaly signals for current day
    n_anomalies = len(state.day_anomalies.get(state.current_day, []))
    anomaly_norm = min(n_anomalies / 4.0, 1.0)
    n_cleaned = len(state.cleaned_columns_today)
    cleaned_norm = min(n_cleaned / 4.0, 1.0)

    # Legacy compatibility signals
    td_a = min(state.time_delta_a / _MAX_TIME_DELTA, 1.0) if state.time_delta_a else 0.0
    td_b = min(state.time_delta_b / _MAX_TIME_DELTA, 1.0) if state.time_delta_b else 0.0
    is_stale_a = 1.0 if state.is_stale_a else 0.0
    is_stale_b = 1.0 if state.is_stale_b else 0.0

    null_a = min(state.null_ratio_key_a, 1.0)
    null_b = min(state.null_ratio_key_b, 1.0)
    match = min(state.match_rate, 1.0)
    dedup_done = 1.0 if state.did_dedup_today else 0.0

    return [
        day_frac,         # 0: day progress
        step_frac,        # 1: step progress within day
        silver_norm,      # 2: silver size
        quarantine_norm,  # 3: quarantine size
        retry_norm,       # 4: retry count
        anomaly_norm,     # 5: anomaly count today
        cleaned_norm,     # 6: cleaned count today
        td_a,             # 7: time delta A
        td_b,             # 8: time delta B
        is_stale_a,       # 9: stale A
        is_stale_b,       # 10: stale B
        null_a,           # 11: null ratio A
        null_b,           # 12: null ratio B
        match,            # 13: match rate
        dedup_done,       # 14: dedup done today
        1.0 if state.did_evolve_schema else 0.0,  # 15: schema evolved
    ]


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class MedusaEnv(Environment[MedusaAction, MedusaObservation, MedusaState]):
    """MEDUSA: Medallion-Engineered Deterministic Unified Storage Agent.

    Simulates a Bronze→Silver data integration pipeline. The agent observes
    data quality signals and chooses ETL actions to produce a correct,
    historically consistent Silver entity.

    v4.0: 30-day gauntlet mode with cumulative Silver and daily Bronze batches.

    Args:
        scenario_seed: Fixed seed for deterministic episodes. ``None`` = random.
        max_steps: Maximum steps per episode before forced termination.
        stale_threshold_hours: Age (hours) at which a source is deemed stale.
        n_fact_rows: Size of Fact / Source A table (legacy mode) or daily batch.
        n_dim_rows: Size of Dimension / Source B table (legacy mode).
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        scenario_seed: Optional[int] = None,
        max_steps: int = 50,
        stale_threshold_hours: float = 6.0,
        n_fact_rows: int = 200,
        n_dim_rows: int = 150,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._scenario_seed = scenario_seed
        self._max_steps = max_steps
        self._stale_threshold = stale_threshold_hours
        self._n_rows = n_fact_rows

        self._generator = ScenarioGenerator(
            n_fact_rows=n_fact_rows, n_dim_rows=n_dim_rows
        )
        self._reward_engine = RewardEngine()
        self._grader = Grader()

        self._state = MedusaState()
        self._tables = _EpisodeTables()
        self._scenario: Optional[Scenario] = None
        self._day_gen: Optional[DayDataGenerator] = None
        self._current_batch: Optional[DayBatch] = None

    def _generate_day_anomalies(self, seed: int) -> Dict[int, List[tuple]]:
        """Generate the deterministic anomaly checklist for all 30 days."""
        gen = DayDataGenerator(episode_seed=seed, n_rows=self._n_rows)
        self._day_gen = gen
        return gen.day_anomalies

    def _load_day_batch(self, day: int) -> None:
        """Generate and load the Bronze batch for the given day."""
        if self._day_gen is None:
            return
        batch = self._day_gen.generate_day(day)
        self._current_batch = batch
        self._tables.daily_raw = batch.raw_data.copy()
        self._tables.daily_cleaned = batch.raw_data.copy()
        self._state.source_row_count = len(batch.raw_data)
        self._state.total_raw_rows += len(batch.raw_data)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="medusa_env",
            description=(
                "MEDUSA: simulated Bronze→Silver integration controller for "
                "multi-source joins, schema drift, and SCD merges."
            ),
            version="4.0.0",
            documentation="envs/medusa_env/README.md",
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> MedusaState:
        return self._state

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation_context(self) -> str:
        """Build the structured text observation per spec §6."""
        lines = []
        lines.append(f"=== Data Contract ===")
        lines.append(f"  Columns: {self._state.current_contract_columns}")
        lines.append(f"  Primary Key: user_id")
        lines.append(f"  Current Day: {self._state.current_day}/30")

        if self._current_batch is not None:
            lines.append(f"  Today's batch: {len(self._current_batch.raw_data)} rows")
            lines.append(f"  Columns in batch: {list(self._current_batch.raw_data.columns)}")

        lines.append(f"  Silver rows: {self._state.silver_row_count}")
        lines.append(f"  Quarantine rows: {self._state.quarantine_row_count}")

        if self._state.cleaned_columns_today:
            lines.append(f"  Cleaned today: {sorted(self._state.cleaned_columns_today)}")

        if self._state.profiled_tables_today:
            lines.append(f"  Profiled today: {dict(self._state.profiled_tables_today)}")

        # §6: Dynamic context — last tool output and BLOCK status
        if self._state.last_action_result:
            lines.append(f"  Last action output: {self._state.last_action_result}")

        if self._state.last_block_reason:
            lines.append(f"  BLOCK: {self._state.last_block_reason}")
        else:
            lines.append(f"  BLOCK: None")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MedusaObservation:
        self._reset_rubric()

        effective_seed = seed if seed is not None else self._scenario_seed
        if effective_seed is None:
            effective_seed = 42
        run_id = episode_id or str(uuid.uuid4())

        # Generate legacy scenario (backward compat)
        self._scenario = self._generator.generate(seed=effective_seed)
        scen = self._scenario

        # Initialise cumulative episode tables
        self._tables = _EpisodeTables(
            silver=pd.DataFrame(),
            quarantine=pd.DataFrame(),
            governance_log=[],
            bronze_a=scen.bronze_a.copy(),
            bronze_a_prepped=pd.DataFrame(),
            bronze_b=scen.bronze_b.copy(),
            bronze_b_prepped=pd.DataFrame(),
        )

        # Generate anomaly checklist + day data generator
        day_anomalies = self._generate_day_anomalies(effective_seed)

        # Initialize schema contract from v4.0 daily batch base columns
        # (the DayDataGenerator defines the canonical column set)
        contract_cols = list(DayDataGenerator.BASE_COLUMNS)

        self._state = MedusaState(
            run_id=run_id,
            seed=effective_seed,
            scenario_id=scen.id,
            max_steps=self._max_steps,
            step_idx=0,
            stage="running",
            current_day=1,
            step_count=0,
            retry_count=0,
            day_anomalies=day_anomalies,
            cleaned_columns_today=set(),
            profiled_tables_today={},
            did_dedup_today=False,
            # Legacy freshness
            time_delta_a=scen.time_delta_a,
            time_delta_b=scen.time_delta_b,
            is_stale_a=scen.is_stale_a,
            is_stale_b=scen.is_stale_b,
            # Contract
            current_contract_columns=contract_cols,
            silver_row_count=0,
            quarantine_row_count=0,
            source_row_count=len(scen.bronze_a),
            source_a_row_count=len(scen.bronze_a),
            silver_row_count_at_day_start=0,
            total_raw_rows=0,
            total_quarantine_rows=0,
        )

        # Load Day 1 batch
        self._load_day_batch(1)

        context = self._build_observation_context()
        features = _build_features(self._state)
        obs = MedusaObservation(
            message=(
                f"MEDUSA 30-day episode started. Scenario: {scen.id}. "
                f"Day 1 ready.\n{context}"
            ),
            features=features,
            metrics={
                "scenario_id": scen.id,
            },
            metadata={"run_id": run_id, "seed": effective_seed},
            reward=None,
            done=False,
        )
        return self._apply_transform(obs)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: MedusaAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MedusaObservation:
        if self._state.stage != "running":
            return self._apply_transform(MedusaObservation(
                message=f"Episode not running (stage={self._state.stage}). Call reset().",
                done=True,
                reward=0.0,
                features=_build_features(self._state),
                metadata={"run_id": self._state.run_id},
            ))

        # 1. Parse LLM XML Action
        text = action.action
        if hasattr(text, 'value'):
            text = text.value
        text = str(text)

        act_str = text
        parsed_args = action.params.copy()

        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        args_match = re.search(r'<args>(.*?)</args>', text, re.DOTALL)
        if action_match:
            act_str = action_match.group(1).strip()
            if args_match:
                try:
                    parsed_args = json.loads(args_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # ── Legacy action translation ─────────────────────────────────
        # Map old Phase-1 actions to v4.0 actions for backward compat
        legacy_result = self._handle_legacy_action(act_str, parsed_args)
        if legacy_result is not None:
            return legacy_result

        # ── Check step limit ──────────────────────────────────────────
        if self._state.step_count >= 10 and act_str != "COMMIT_DAY":
            self._state.stage = "failed"
            crash_pen = self._reward_engine.crash_reward(self._state.current_day)
            self._state.cumulative_reward += crash_pen
            return self._apply_transform(MedusaObservation(
                message=f"[MAX STEPS REACHED] Terminal Crash triggered. ({crash_pen})",
                done=True,
                reward=crash_pen,
                features=_build_features(self._state),
                metadata={"run_id": self._state.run_id},
            ))

        # Snapshot state *before* applying action
        state_before = copy.copy(self._state)
        self._state.step_count += 1
        self._state.step_idx += 1

        metrics: dict = {}
        step_message = ""
        scen = self._scenario
        assert scen is not None, "reset() must be called before step()"

        reward = self._reward_engine.REWARD_TABLE["step_cost"]
        is_blocked = False

        # ── Dispatch v4.0 tools ───────────────────────────────────────
        try:
            if act_str == "PROFILE_TABLE":
                reward, step_message, metrics, is_blocked = self._do_profile_table(
                    parsed_args, reward
                )

            elif act_str == "CLEAN_COLUMN":
                reward, step_message, metrics, is_blocked = self._do_clean_column(
                    parsed_args, reward
                )

            elif act_str == "DEDUPLICATE":
                reward, step_message, metrics, is_blocked = self._do_deduplicate(
                    parsed_args, reward
                )

            elif act_str == "EVOLVE_SILVER_SCHEMA":
                reward, step_message, metrics, is_blocked = self._do_evolve_silver_schema(
                    parsed_args, reward
                )

            elif act_str == "QUARANTINE_ROWS":
                reward, step_message, metrics, is_blocked = self._do_quarantine_rows(
                    parsed_args, reward
                )

            elif act_str == "EXECUTE_MERGE":
                reward, step_message, metrics, is_blocked = self._do_execute_merge(
                    parsed_args, reward
                )

            elif act_str == "COMMIT_DAY":
                return self._do_commit(state_before)

            else:
                step_message = f"INVALID ACTION: {act_str}"

        except Exception as exc:
            step_message = f"ERROR in {act_str}: {exc}"
            metrics = {"error": str(exc)}

        # ── Handle Blocks & Retries ───────────────────────────────────
        done = False
        if is_blocked:
            self._state.retry_count += 1
            reward = self._reward_engine.REWARD_TABLE["block_penalty"]
            self._state.last_block_reason = step_message
            if self._state.retry_count >= 3:
                reward = self._reward_engine.crash_reward(self._state.current_day)
                done = True
                self._state.stage = "failed"
                step_message += " [TERMINAL CRASH: 3 Retries Exceeded]"
        else:
            self._state.last_block_reason = ""

        # Track last action result for observation context (§6)
        self._state.last_action_result = step_message

        if not done and self._state.step_count >= 10:
            self._state.stage = "failed"
            reward = self._reward_engine.crash_reward(self._state.current_day)
            done = True
            step_message += " [MAX STEPS REACHED]"

        self._state.cumulative_reward += reward

        # ── Governance log ────────────────────────────────────────────
        self._tables.governance_log.append({
            "step": self._state.step_idx,
            "day": self._state.current_day,
            "action": act_str,
            "reward": reward,
            "cumulative_reward": self._state.cumulative_reward,
            "metrics": metrics,
            "blocked": is_blocked,
            "timestamp": time.time(),
        })

        context = self._build_observation_context()
        features = _build_features(self._state)
        obs = MedusaObservation(
            message=f"{step_message}\n{context}",
            features=features,
            metrics=metrics,
            metadata={
                "run_id": self._state.run_id,
                "step": self._state.step_idx,
                "day": self._state.current_day,
                "cumulative_reward": self._state.cumulative_reward,
            },
            reward=reward,
            done=done,
        )
        return self._apply_transform(obs)

    # ------------------------------------------------------------------
    # v4.0 Tool Implementations
    # ------------------------------------------------------------------

    def _do_profile_table(
        self, args: dict, base_reward: float
    ) -> Tuple[float, str, dict, bool]:
        """PROFILE_TABLE: Returns schema, types, null %, duplicate counts."""
        table = args.get("table", "bronze")
        reward = base_reward

        # Escalating cost: 2nd call on same table = -1.0
        call_count = self._state.profiled_tables_today.get(table, 0)
        self._state.profiled_tables_today[table] = call_count + 1

        if call_count >= 1:
            reward = self._reward_engine.REWARD_TABLE["profile_2nd_call"]

        # Profile the daily raw data
        df = self._tables.daily_cleaned
        if df.empty:
            return reward, "PROFILE_TABLE: No data loaded.", {}, False

        profile: Dict[str, Any] = {}
        for col in df.columns:
            null_pct = round(df[col].isnull().mean() * 100, 1)
            dtype = str(df[col].dtype)
            n_unique = int(df[col].nunique())
            n_dupes = int(df[col].duplicated().sum())
            profile[col] = {
                "dtype": dtype,
                "null_pct": null_pct,
                "n_unique": n_unique,
                "n_duplicates": n_dupes,
            }

        # Update state with key health info
        if "user_id" in df.columns:
            self._state.null_ratio_key_a = float(df["user_id"].isnull().mean())
            self._state.uniqueness_a = float(
                df["user_id"].nunique() / max(len(df), 1)
            )

        profile_str = "\n".join(
            f"  {col}: type={info['dtype']}, null={info['null_pct']}%, "
            f"unique={info['n_unique']}, dupes={info['n_duplicates']}"
            for col, info in profile.items()
        )
        msg = (
            f"PROFILE_TABLE ({table}): {len(df)} rows, {len(df.columns)} columns\n"
            f"{profile_str}"
        )
        return reward, msg, {"profile": profile, "call_count": call_count + 1}, False

    def _do_clean_column(
        self, args: dict, base_reward: float
    ) -> Tuple[float, str, dict, bool]:
        """CLEAN_COLUMN: Applies cast, strip, or fill_zero to a column."""
        table = args.get("table", "bronze")
        col = args.get("col", "")
        op = args.get("op", "")
        reward = base_reward

        # Check for block: cannot repeat the exact same (col, op) today
        col_op = (col, op)
        if col_op in self._state.cleaned_columns_today:
            return -2.0, f"BLOCK: ({col}, {op}) already applied today.", {}, True

        self._state.cleaned_columns_today.add(col_op)

        df = self._tables.daily_cleaned
        if col not in df.columns:
            return reward, f"CLEAN_COLUMN: Column '{col}' not found.", {}, False

        # Apply actual transformation
        rows_affected = 0
        if op == "strip":
            mask = df[col].notna()
            before = df[col].copy()
            df[col] = df[col].apply(
                lambda x: str(x).strip() if pd.notna(x) else x
            )
            rows_affected = int((before != df[col]).sum())

        elif op == "cast":
            before_dtype = str(df[col].dtype)
            # Try to cast to numeric — handles "$50.50" → 50.50 style
            df[col] = df[col].apply(
                lambda x: str(x).replace("$", "").replace(",", "").strip()
                if pd.notna(x) else x
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            rows_affected = int(df[col].notna().sum())

        elif op == "fill_zero":
            null_before = int(df[col].isnull().sum())
            df[col] = df[col].fillna(0)
            rows_affected = null_before

        else:
            return reward, f"CLEAN_COLUMN: Unknown op '{op}'.", {}, False

        self._tables.daily_cleaned = df

        # Award dense reward if matches today's anomaly checklist
        today_anomalies = self._state.day_anomalies.get(self._state.current_day, [])
        if col_op in today_anomalies:
            reward += self._reward_engine.REWARD_TABLE["clean_checklist_hit"]

        msg = f"CLEAN_COLUMN: applied {op} to {col} ({rows_affected} rows affected)."
        return reward, msg, {"col": col, "op": op, "rows_affected": rows_affected}, False

    def _do_deduplicate(
        self, args: dict, base_reward: float
    ) -> Tuple[float, str, dict, bool]:
        """DEDUPLICATE: Removes duplicate rows from the daily batch."""
        key = args.get("key", "user_id")
        reward = base_reward

        if self._state.did_dedup_today:
            return -2.0, "BLOCK: DEDUPLICATE already called today.", {}, True

        self._state.did_dedup_today = True
        self._state.did_dedup_b = True  # Legacy compat

        df = self._tables.daily_cleaned
        before_len = len(df)

        if key in df.columns:
            df = df.drop_duplicates(subset=[key], keep="last")
        else:
            df = df.drop_duplicates(keep="last")

        after_len = len(df)
        dupes_removed = before_len - after_len
        self._tables.daily_cleaned = df

        # Reward: +0.5 if actually removed duplicates
        if dupes_removed > 0:
            reward += self._reward_engine.REWARD_TABLE["deduplicate_effective"]

        msg = f"DEDUPLICATE: removed {dupes_removed} duplicate rows (key={key})."
        return reward, msg, {"dupes_removed": dupes_removed, "key": key}, False

    def _do_evolve_silver_schema(
        self, args: dict, base_reward: float
    ) -> Tuple[float, str, dict, bool]:
        """EVOLVE_SILVER_SCHEMA: Adds a new column to contract + Silver."""
        column = args.get("column", "")
        reward = base_reward

        df = self._tables.daily_cleaned

        # Validate: column must exist in today's raw data
        if column not in df.columns:
            reward = -1.0
            return reward, (
                f"EVOLVE_SILVER_SCHEMA: Column '{column}' does not exist in "
                f"today's raw data. Penalty applied."
            ), {"column": column, "exists": False}, False

        # Add to contract
        if column not in self._state.current_contract_columns:
            self._state.current_contract_columns.append(column)

        # Add to Silver schema if Silver exists
        if not self._tables.silver.empty and column not in self._tables.silver.columns:
            self._tables.silver[column] = np.nan

        self._state.did_evolve_schema = True
        reward += 1.0

        msg = f"EVOLVE_SILVER_SCHEMA: Added '{column}' to Data Contract and Silver schema."
        return reward, msg, {"column": column, "exists": True}, False

    def _do_quarantine_rows(
        self, args: dict, base_reward: float
    ) -> Tuple[float, str, dict, bool]:
        """QUARANTINE_ROWS: Routes rows matching condition to quarantine."""
        table = args.get("table", "bronze")
        condition = args.get("condition", "")
        reward = base_reward

        df = self._tables.daily_cleaned

        # Parse condition (simple IS NULL check)
        quarantined = pd.DataFrame()
        kept = df.copy()

        if "IS NULL" in condition.upper():
            # Extract column name from "col IS NULL"
            col_name = condition.upper().replace("IS NULL", "").strip()
            # Find actual column name (case-insensitive)
            for c in df.columns:
                if c.upper() == col_name:
                    col_name = c
                    break
            if col_name in df.columns:
                mask = df[col_name].isnull()
                quarantined = df[mask]
                kept = df[~mask]
        elif "IS NOT NULL" in condition.upper():
            col_name = condition.upper().replace("IS NOT NULL", "").strip()
            for c in df.columns:
                if c.upper() == col_name:
                    col_name = c
                    break
            if col_name in df.columns:
                mask = df[col_name].notna()
                quarantined = df[mask]
                kept = df[~mask]

        n_quarantined = len(quarantined)
        self._tables.daily_cleaned = kept

        # Add quarantined rows to episode quarantine
        if n_quarantined > 0:
            self._tables.quarantine = pd.concat(
                [self._tables.quarantine, quarantined], ignore_index=True
            )
            self._state.quarantine_row_count = len(self._tables.quarantine)
            self._state.total_quarantine_rows += n_quarantined

        # Reward: +0.5 if matches today's checklist
        today_anomalies = self._state.day_anomalies.get(self._state.current_day, [])
        for col, op in today_anomalies:
            if op == "quarantine" and col.lower() in condition.lower():
                reward += 0.5
                break

        # Track Day 28 approved quarantine separately for ceiling calc
        if self._state.current_day == 28 and n_quarantined > 0:
            self._state.day28_quarantine_rows += n_quarantined

        msg = f"QUARANTINE_ROWS: {n_quarantined} rows quarantined (condition: {condition})."
        return reward, msg, {
            "quarantined_rows": n_quarantined,
            "condition": condition,
            "remaining_rows": len(kept),
        }, False

    def _do_execute_merge(
        self, args: dict, base_reward: float
    ) -> Tuple[float, str, dict, bool]:
        """EXECUTE_MERGE: Joins daily batch into cumulative Silver."""
        reward = base_reward

        df = self._tables.daily_cleaned
        silver = self._tables.silver

        if df.empty:
            return reward, "EXECUTE_MERGE: No data to merge (daily batch is empty).", {}, False

        # Estimate output size
        raw_len = len(df)
        if "user_id" in df.columns:
            unique_keys = max(df["user_id"].nunique(), 1)
            dup_ratio = 1.0 - (unique_keys / max(raw_len, 1))
        else:
            unique_keys = raw_len
            dup_ratio = 0.0

        # Block if estimated output is too large (OOM guard)
        silv_len = max(len(silver), 1)
        estimate = raw_len * max(1, int(raw_len / unique_keys))
        memory_limit = 100000

        if estimate > memory_limit:
            return (
                -2.0,
                f"BLOCK: Estimated output ({estimate}) exceeds memory limit. "
                f"Current duplicate ratio: {dup_ratio:.1%}. "
                f"Try DEDUPLICATE first.",
                {"estimate": estimate, "dup_ratio": dup_ratio},
                True,
            )

        # Align schemas: add missing columns
        for col in silver.columns:
            if col not in df.columns:
                df[col] = np.nan
        for col in df.columns:
            if not silver.empty and col not in silver.columns:
                silver[col] = np.nan

        # Key-based upsert merge on user_id (not simple append)
        silver_before_len = len(silver)
        if silver.empty:
            # First merge: just set Silver to the daily batch
            self._tables.silver = df.copy()
        elif "user_id" in df.columns and "user_id" in silver.columns:
            # Upsert: update existing keys, append new keys
            existing_keys = set(silver["user_id"].dropna())
            new_mask = ~df["user_id"].isin(existing_keys) | df["user_id"].isna()
            new_rows = df[new_mask]
            update_rows = df[~new_mask]

            # Update existing rows (SCD-1 style overwrite)
            if not update_rows.empty:
                silver = silver.set_index("user_id")
                update_rows_indexed = update_rows.set_index("user_id")
                silver.update(update_rows_indexed)
                silver = silver.reset_index()

            # Append genuinely new rows
            if not new_rows.empty:
                silver = pd.concat([silver, new_rows], ignore_index=True)

            self._tables.silver = silver
        else:
            # Fallback: plain concat if no user_id column
            self._tables.silver = pd.concat([silver, df], ignore_index=True)

        self._state.silver_row_count = len(self._tables.silver)
        self._state.match_rate = 1.0  # Direct merge = all rows match

        # Legacy compat
        self._state.did_join = True
        self._state.join_type = "left"
        self._state.join_row_count = len(df)

        msg = (
            f"EXECUTE_MERGE: Merged {len(df)} rows into Silver. "
            f"Silver now has {self._state.silver_row_count} rows "
            f"(was {silver_before_len})."
        )
        
        if self._state.silver_row_count > silver_before_len:
            reward += self._reward_engine.REWARD_TABLE["execute_merge_success"]

        return reward, msg, {
            "merged_rows": len(df),
            "silver_total": self._state.silver_row_count,
            "dup_ratio": dup_ratio,
        }, False

    # ------------------------------------------------------------------
    # Commit (terminal step per day or episode end)
    # ------------------------------------------------------------------

    def _do_commit(self, state_before: MedusaState) -> MedusaObservation:
        """Run grader then finalise the episode or advance the day."""
        scen = self._scenario
        assert scen is not None

        self._state.step_idx += 1
        self._state.step_count += 1

        # Deterministic Physics Checks — delegated to Grader
        silver_after_len = len(self._tables.silver)
        grader_result = self._grader.audit(
            silver=self._tables.silver,
            silver_at_day_start=self._state.silver_row_count_at_day_start,
            current_day=self._state.current_day,
            contract_columns=self._state.current_contract_columns,
        )
        grader_passed = grader_result.passed
        grader_report = grader_result.report

        self._state.grader_passed = grader_passed
        self._state.grader_report = grader_report
        self._state.silver_row_count = silver_after_len
        self._state.quarantine_row_count = len(self._tables.quarantine)

        done = False
        if not grader_passed:
            # Failure = Terminal Crash
            reward = self._reward_engine.crash_reward(self._state.current_day)
            done = True
            self._state.stage = "failed"
            step_message = f"COMMIT_DAY: Grader FAIL (Terminal Crash). Report: {grader_report}"
        else:
            # Pass
            reward = self._reward_engine.commit_reward(self._state.current_day)
            step_message = f"COMMIT_DAY: Grader PASS. Day {self._state.current_day} complete. (+{self._state.current_day})"

            # Advance clock
            self._state.current_day += 1
            self._state.step_count = 0
            self._state.retry_count = 0
            self._state.cleaned_columns_today = set()
            self._state.profiled_tables_today = {}
            self._state.silver_row_count_at_day_start = silver_after_len
            self._state.did_dedup_today = False

            if self._state.current_day > 30:
                done = True
                self._state.stage = "committed"

                # Day 30 Completion Quarantine ceiling
                # Exclude Day 28 approved quarantine (null user_id) per plan §5
                tot_quar = self._state.total_quarantine_rows
                day28_approved = self._state.day28_quarantine_rows
                adjusted_quar = tot_quar - day28_approved
                tot_raw = max(self._state.total_raw_rows, 1)
                if (adjusted_quar / tot_raw) <= 0.05:
                    bonus = self._reward_engine.completion_bonus()
                    reward += bonus
                    step_message += f" EPISODE FINISHED! Perfect completion (+{bonus} bonus)."
                else:
                    step_message += (
                        f" EPISODE FINISHED! Quarantine ratio failed "
                        f"({adjusted_quar}/{tot_raw} = {adjusted_quar/tot_raw:.1%}, "
                        f"excluding {day28_approved} Day 28 approved). No bonus."
                    )
            else:
                # Load next day's batch
                self._load_day_batch(self._state.current_day)

        self._state.cumulative_reward += reward

        self._tables.governance_log.append({
            "step": self._state.step_idx,
            "day": state_before.current_day,
            "action": "COMMIT_DAY",
            "reward": reward,
            "cumulative_reward": self._state.cumulative_reward,
            "grader_passed": grader_passed,
            "grader_report": grader_report,
            "timestamp": time.time(),
        })

        # Calculate the final [0, 1] evaluation score for this episode only if done
        final_score = 1.0 if (done and self._state.stage == "committed") else 0.0
        if done and self._state.stage == "committed":
            task_id = next(
                (tid for tid, t in TASKS.items() if t.seed == self._state.seed),
                "clean_pipeline",
            )
            try:
                final_result = score_episode(task_id, self._state, self._tables)
                final_score = final_result.score
            except Exception:
                pass

        context = self._build_observation_context()
        features = _build_features(self._state)
        obs = MedusaObservation(
            message=f"{step_message}\n{context} | Total reward: {self._state.cumulative_reward:.1f}",
            features=features,
            metrics={
                "grader_passed": grader_passed,
                "grader_report": grader_report,
                "silver_rows": self._state.silver_row_count,
                "quarantine_rows": self._state.quarantine_row_count,
                "governance_log_entries": len(self._tables.governance_log),
                "score": final_score,
            },
            metadata={
                "run_id": self._state.run_id,
                "steps": self._state.step_idx,
                "day": self._state.current_day,
                "cumulative_reward": self._state.cumulative_reward,
                "score": final_score,
            },
            reward=reward,
            done=done,
        )
        return self._apply_transform(obs)

    # ------------------------------------------------------------------
    # Legacy Phase-1 action handler (backward compat)
    # ------------------------------------------------------------------

    def _handle_legacy_action(
        self, act_str: str, parsed_args: dict
    ) -> Optional[MedusaObservation]:
        """Handle old Phase-1 actions for backward compat with existing tests."""
        scen = self._scenario
        if scen is None:
            return None

        legacy_actions = {
            "SYNC_CHECK", "PREP_KEYS_A", "PREP_KEYS_B", "DEDUPLICATE_B",
            "EXECUTE_JOIN_INNER", "EXECUTE_JOIN_LEFT", "EXECUTE_JOIN_ANTI",
            "APPLY_SCD_1", "APPLY_SCD_2", "EVOLVE_SCHEMA", "COMMIT",
        }

        if act_str not in legacy_actions:
            return None

        # Check step limit (legacy uses max_steps, not 10/day)
        if self._state.step_idx >= self._state.max_steps:
            self._state.stage = "failed"
            crash_pen = self._reward_engine.crash_reward(self._state.current_day)
            self._state.cumulative_reward += crash_pen
            return self._apply_transform(MedusaObservation(
                message="Episode exceeded max steps — terminal.",
                done=True,
                reward=crash_pen,
                features=_build_features(self._state),
                metadata={"run_id": self._state.run_id},
            ))

        state_before = copy.copy(self._state)
        self._state.step_idx += 1

        metrics: dict = {}
        step_message = ""
        reward = self._reward_engine.REWARD_TABLE["step_cost"]

        try:
            if act_str == "SYNC_CHECK":
                _, m = sync_check(
                    self._tables.bronze_a, self._tables.bronze_b,
                    scen.time_delta_a, scen.time_delta_b,
                    self._stale_threshold,
                )
                metrics = m
                self._state.did_sync_check = True
                self._state.is_stale_a = m["is_stale_a"]
                self._state.is_stale_b = m["is_stale_b"]
                step_message = (
                    f"SYNC_CHECK: A is {'STALE' if m['is_stale_a'] else 'FRESH'}, "
                    f"B is {'STALE' if m['is_stale_b'] else 'FRESH'}."
                )

            elif act_str == "PREP_KEYS_A":
                result, m = prep_keys(self._tables.bronze_a, scen.join_key)
                self._tables.bronze_a_prepped = result
                self._state.did_prep_a = True
                self._state.null_ratio_key_a = m["null_ratio_before"]
                self._state.uniqueness_a = m.get("uniqueness", 1.0)
                metrics = m
                step_message = f"PREP_KEYS_A: {m.get('rows_cleaned', 0)} rows cleaned."

            elif act_str == "PREP_KEYS_B":
                result, m = prep_keys(self._tables.bronze_b, scen.join_key)
                self._tables.bronze_b_prepped = result
                self._state.did_prep_b = True
                self._state.null_ratio_key_b = m["null_ratio_before"]
                self._state.uniqueness_b = m.get("uniqueness", 1.0)
                metrics = m
                step_message = f"PREP_KEYS_B: {m.get('rows_cleaned', 0)} rows cleaned."

            elif act_str == "DEDUPLICATE_B":
                src = (
                    self._tables.bronze_b_prepped
                    if not self._tables.bronze_b_prepped.empty
                    else self._tables.bronze_b
                )
                result, m = deduplicate(src, scen.join_key)
                self._tables.bronze_b_prepped = result
                self._state.did_dedup_b = True
                metrics = m
                step_message = f"DEDUPLICATE_B: {m['dupes_removed']} duplicates removed."

            elif act_str.startswith("EXECUTE_JOIN"):
                join_type = act_str.split("_")[-1].lower()
                fact_src = (
                    self._tables.bronze_a_prepped
                    if not self._tables.bronze_a_prepped.empty
                    else self._tables.bronze_a
                )
                dim_src = (
                    self._tables.bronze_b_prepped
                    if not self._tables.bronze_b_prepped.empty
                    else self._tables.bronze_b
                )
                joined, quarantine, m = execute_join(
                    fact_src, dim_src, scen.join_key, join_type
                )
                self._tables.joined = joined
                self._tables.quarantine = quarantine
                self._state.did_join = True
                self._state.join_type = join_type
                self._state.join_row_count = m["join_rows"]
                self._state.match_rate = m["match_rate"]
                self._state.explosion_detected = m["explosion_detected"]
                self._state.quarantine_row_count = len(quarantine)
                metrics = m
                step_message = (
                    f"EXECUTE_JOIN_{join_type.upper()}: "
                    f"{m['join_rows']} rows, match_rate={m['match_rate']:.1%}."
                )

            elif act_str == "EVOLVE_SCHEMA":
                result, m = evolve_schema(
                    self._tables.silver,
                    self._tables.bronze_a, self._tables.bronze_b,
                    scen.new_cols_a, scen.new_cols_b,
                )
                self._tables.silver = result
                self._state.did_evolve_schema = True
                metrics = m
                step_message = f"EVOLVE_SCHEMA: {m['new_cols_count']} new columns added."

            elif act_str.startswith("APPLY_SCD"):
                scd_type = 1 if "SCD_1" in act_str else 2
                joined_src = self._tables.joined
                if joined_src.empty:
                    step_message = "APPLY_SCD: No joined data. Run EXECUTE_JOIN first."
                    metrics = {"error": "no_joined_data"}
                else:
                    result, m = apply_scd(
                        self._tables.silver, joined_src,
                        scen.join_key,
                        scen.tracked_cols[0] if scen.tracked_cols else "dim_status",
                        scd_type,
                    )
                    self._tables.silver = result
                    self._state.did_scd = True
                    self._state.scd_type = f"SCD-{scd_type}"
                    self._state.scd_inserts = m.get("inserts", 0)
                    self._state.scd_updates = m.get("updates", 0)
                    self._state.silver_row_count = len(result)
                    metrics = m
                    step_message = (
                        f"APPLY_SCD_{scd_type}: "
                        f"{m.get('inserts', 0)} inserts, {m.get('updates', 0)} updates."
                    )

            elif act_str == "COMMIT":
                # Legacy commit — run v4.0 grader with legacy-compat wrapper
                self._state.source_a_row_count = len(self._tables.bronze_a)
                grader_result = self._grader.audit(
                    silver=self._tables.silver,
                    silver_at_day_start=0,  # legacy: no day-start tracking
                    current_day=1,
                    contract_columns=list(self._tables.silver.columns) if not self._tables.silver.empty else [],
                )
                # Legacy compat: the old grader had .passed, .report, .bonus_reward
                grader_result.bonus_reward = 15.0 if grader_result.passed else -20.0
                self._state.grader_passed = grader_result.passed
                self._state.grader_report = grader_result.report
                self._state.silver_row_count = len(self._tables.silver)
                self._state.quarantine_row_count = len(self._tables.quarantine)
                self._state.stage = "committed"

                reward += grader_result.bonus_reward
                self._state.cumulative_reward += reward

                self._tables.governance_log.append({
                    "step": self._state.step_idx,
                    "action": "COMMIT",
                    "reward": reward,
                    "cumulative_reward": self._state.cumulative_reward,
                    "grader_passed": grader_result.passed,
                    "grader_report": grader_result.report,
                    "timestamp": time.time(),
                })

                # Task scoring
                final_score = 0.0
                task_id = next(
                    (tid for tid, t in TASKS.items() if t.seed == self._state.seed),
                    "clean_pipeline",
                )
                try:
                    final_result = score_episode(task_id, self._state, self._tables)
                    final_score = final_result.score
                except Exception:
                    pass

                features = _build_features(self._state)
                return self._apply_transform(MedusaObservation(
                    message=(
                        f"COMMIT: {grader_result.report}\n"
                        f"Total reward: {self._state.cumulative_reward:.1f}"
                    ),
                    features=features,
                    metrics={
                        "grader_passed": grader_result.passed,
                        "grader_report": grader_result.report,
                        "silver_rows": self._state.silver_row_count,
                        "quarantine_rows": self._state.quarantine_row_count,
                        "governance_log_entries": len(self._tables.governance_log),
                        "score": final_score,
                    },
                    metadata={
                        "run_id": self._state.run_id,
                        "steps": self._state.step_idx,
                        "cumulative_reward": self._state.cumulative_reward,
                        "score": final_score,
                    },
                    reward=reward,
                    done=True,
                ))

        except Exception as exc:
            step_message = f"ERROR in {act_str}: {exc}"
            metrics = {"error": str(exc)}

        # Evaluate reward via legacy engine
        legacy_reward = self._reward_engine.evaluate(
            act_str, metrics, state_before
        )
        reward = legacy_reward
        self._state.cumulative_reward += reward

        self._tables.governance_log.append({
            "step": self._state.step_idx,
            "action": act_str,
            "reward": reward,
            "cumulative_reward": self._state.cumulative_reward,
            "metrics": metrics,
            "timestamp": time.time(),
        })

        features = _build_features(self._state)
        obs = MedusaObservation(
            message=step_message,
            features=features,
            metrics=metrics,
            metadata={
                "run_id": self._state.run_id,
                "step": self._state.step_idx,
                "cumulative_reward": self._state.cumulative_reward,
            },
            reward=reward,
            done=False,
        )
        return self._apply_transform(obs)