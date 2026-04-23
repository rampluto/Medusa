from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class MedusaActionType(str, Enum):
    """Discrete action set for the MEDUSA controller (v4.0 — 7 tools)."""

    PROFILE_TABLE = "PROFILE_TABLE"
    CLEAN_COLUMN = "CLEAN_COLUMN"
    DEDUPLICATE = "DEDUPLICATE"
    EVOLVE_SILVER_SCHEMA = "EVOLVE_SILVER_SCHEMA"
    QUARANTINE_ROWS = "QUARANTINE_ROWS"
    EXECUTE_MERGE = "EXECUTE_MERGE"
    COMMIT_DAY = "COMMIT_DAY"

    # --- Legacy Phase-1 aliases (kept for backward compat with tests) ---
    SYNC_CHECK = "SYNC_CHECK"
    PREP_KEYS_A = "PREP_KEYS_A"
    PREP_KEYS_B = "PREP_KEYS_B"
    DEDUPLICATE_B = "DEDUPLICATE_B"
    EXECUTE_JOIN_INNER = "EXECUTE_JOIN_INNER"
    EXECUTE_JOIN_LEFT = "EXECUTE_JOIN_LEFT"
    EXECUTE_JOIN_ANTI = "EXECUTE_JOIN_ANTI"
    APPLY_SCD_1 = "APPLY_SCD_1"
    APPLY_SCD_2 = "APPLY_SCD_2"
    EVOLVE_SCHEMA = "EVOLVE_SCHEMA"
    COMMIT = "COMMIT"


class MedusaAction(Action):
    """One controller action (enum + optional params for future use)."""

    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MedusaState(State):
    """Full pipeline controller state.

    Tracks every book-keeping flag needed by the reward engine, grader,
    and task scorer across both the v4.0 30-day gauntlet and legacy
    Phase-1 single-episode modes.
    """

    run_id: Optional[str] = None
    seed: Optional[int] = None
    scenario_id: Optional[str] = None
    max_steps: int = 20

    step_idx: int = 0
    stage: str = "init"  # init | running | committed | failed

    # --- Timestepping (v4.0 30-day gauntlet) ---
    current_day: int = 1
    step_count: int = 0
    retry_count: int = 0

    # --- 30-Day Gauntlet State ---
    day_anomalies: Dict[int, List[tuple]] = Field(default_factory=dict)
    cleaned_columns_today: List[Tuple[str, str]] = Field(default_factory=list)
    profiled_tables_today: Dict[str, int] = Field(default_factory=dict)
    did_dedup_today: bool = False

    # --- Freshness (Legacy Phase-1 + v4.0) ---
    time_delta_a: float = 0.0
    time_delta_b: float = 0.0
    is_stale_a: bool = False
    is_stale_b: bool = False
    did_sync_check: bool = False

    # --- Key health (Legacy Phase-1) ---
    null_ratio_key_a: float = 0.0
    null_ratio_key_b: float = 0.0
    uniqueness_a: float = 1.0
    uniqueness_b: float = 1.0
    did_prep_a: bool = False
    did_prep_b: bool = False
    did_dedup_b: bool = False

    # --- Join (Legacy Phase-1) ---
    match_rate: float = 0.0
    did_join: bool = False
    join_type: Optional[str] = None
    join_row_count: int = 0
    explosion_detected: bool = False

    # --- SCD (Legacy Phase-1) ---
    did_scd: bool = False
    scd_type: Optional[str] = None
    scd_inserts: int = 0
    scd_updates: int = 0

    # --- Schema evolution ---
    did_evolve_schema: bool = False

    # --- Silver / Quarantine ---
    silver_row_count: int = 0
    quarantine_row_count: int = 0
    source_row_count: int = 0
    source_a_row_count: int = 0
    silver_row_count_at_day_start: int = 0  # For freshness check in _do_commit
    total_raw_rows: int = 0
    total_quarantine_rows: int = 0
    day28_quarantine_rows: int = 0

    # --- Grader ---
    current_contract_columns: List[str] = Field(default_factory=list)
    grader_passed: bool = False
    grader_report: str = ""

    # --- Observation context (v4.0 §6) ---
    last_action_result: str = ""
    last_block_reason: str = ""

    # --- Governance ---
    cumulative_reward: float = 0.0


class MedusaObservation(Observation):
    """Observation returned to the agent after every step.

    ``features`` is a 16-element normalised float vector suitable as
    direct RL input::

        [time_delta_a_norm, time_delta_b_norm, is_stale_a, is_stale_b,
         null_ratio_key_a, null_ratio_key_b, uniqueness_a, uniqueness_b,
         match_rate, new_cols_a_norm, new_cols_b_norm, schema_compat,
         did_prep_a, did_prep_b, did_dedup_b, step_frac]
    """

    message: str = ""
    features: List[float] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reward: Optional[float] = None
    done: bool = False
