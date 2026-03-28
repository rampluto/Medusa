from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class MedusaActionType(str, Enum):
    """Discrete action set for the MEDUSA controller."""

    SYNC_CHECK = "SYNC_CHECK"
    EVOLVE_SCHEMA = "EVOLVE_SCHEMA"
    PREP_KEYS_A = "PREP_KEYS_A"
    PREP_KEYS_B = "PREP_KEYS_B"
    DEDUPLICATE_B = "DEDUPLICATE_B"
    EXECUTE_JOIN_INNER = "EXECUTE_JOIN_INNER"
    EXECUTE_JOIN_LEFT = "EXECUTE_JOIN_LEFT"
    EXECUTE_JOIN_ANTI = "EXECUTE_JOIN_ANTI"
    APPLY_SCD_1 = "APPLY_SCD_1"
    APPLY_SCD_2 = "APPLY_SCD_2"
    COMMIT = "COMMIT"


class MedusaAction(Action):
    """One controller action (enum + optional params for future use)."""

    action: MedusaActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class MedusaState(State):
    """Full pipeline controller state.

    Tracks every book-keeping flag needed by the reward engine and grader.
    """

    run_id: Optional[str] = None
    seed: Optional[int] = None
    scenario_id: Optional[str] = None
    max_steps: int = 20

    step_idx: int = 0
    stage: str = "init"  # init | running | committed | failed

    # --- Freshness ---
    time_delta_a: float = 0.0  # Hours since Source A last updated
    time_delta_b: float = 0.0
    is_stale_a: bool = False
    is_stale_b: bool = False
    did_sync_check: bool = False

    # --- Schema ---
    did_evolve_schema: bool = False
    new_cols_a: int = 0   # Number of new columns in A not yet in Silver
    new_cols_b: int = 0
    schema_compat: float = 1.0  # 0-1 key-type compatibility score

    # --- Key Health ---
    null_ratio_key_a: float = 0.0
    null_ratio_key_b: float = 0.0
    uniqueness_a: float = 1.0   # 1.0 = fully unique
    uniqueness_b: float = 1.0
    did_prep_a: bool = False
    did_prep_b: bool = False
    did_dedup_b: bool = False

    # --- Referential Integrity ---
    match_rate: float = 0.0  # % of Key_A values found in Key_B

    # --- Join Result ---
    did_join: bool = False
    join_type: Optional[str] = None
    join_row_count: int = 0
    explosion_detected: bool = False

    # --- SCD ---
    did_scd: bool = False
    scd_type: Optional[str] = None
    scd_inserts: int = 0
    scd_updates: int = 0

    # --- Silver / Quarantine ---
    silver_row_count: int = 0
    quarantine_row_count: int = 0
    source_a_row_count: int = 0

    # --- Grader ---
    grader_passed: bool = False
    grader_report: str = ""

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
