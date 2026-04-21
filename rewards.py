"""MEDUSA reward engine — v4.0.

Reward table as defined in MEDUSA4_plan.md §7.  All reward logic lives in
``RewardEngine`` so it can be unit-tested independently of the environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Reward table (plan §7)
# ---------------------------------------------------------------------------

REWARD_TABLE: Dict[str, float] = {
    "step_cost":              -0.1,   # Any valid action
    "profile_2nd_call":       -1.0,   # profile_table 2nd call/day on same table
    "clean_checklist_hit":    +1.0,   # clean_column on a (col, op) on today's checklist
    "deduplicate_effective":  +1.0,   # deduplicate that removes > 0 rows (1st call/day)
    "evolve_schema_valid":    +1.0,   # evolve_silver_schema on a real drifted column
    "evolve_schema_invalid":  -1.0,   # evolve_silver_schema on non-existent column
    "quarantine_checklist":   +0.5,   # quarantine_rows on rows matching checklist
    "block_penalty":          -2.0,   # BLOCK triggered (action prevented)
    "execute_merge_success":  +3.0,   # execute_merge adds rows to Silver
    "completion_bonus":     +200.0,   # Day 30 completion + clean quarantine
}


# ---------------------------------------------------------------------------
# RewardEngine
# ---------------------------------------------------------------------------

class RewardEngine:
    """Compute per-step reward from action context per plan §7.

    The environment calls ``compute()`` for each tool call to get the
    scalar reward.  ``commit_reward()`` is called separately for
    COMMIT_DAY since that action has its own day-number reward.
    """
    REWARD_TABLE = REWARD_TABLE

    def compute(
        self,
        action_type: str,
        *,
        # PROFILE_TABLE
        profile_call_count: int = 0,
        # CLEAN_COLUMN
        col_op: Tuple[str, str] = ("", ""),
        today_anomalies: List[Tuple[str, str]] | None = None,
        # DEDUPLICATE
        dupes_removed: int = 0,
        # EVOLVE_SILVER_SCHEMA
        column_exists_in_raw: bool = True,
        # EXECUTE_MERGE
        merge_added_rows: bool = False,
        # QUARANTINE_ROWS
        quarantine_condition: str = "",
        # BLOCK
        is_blocked: bool = False,
    ) -> float:
        """Return the scalar reward for a single non-commit step.

        Args:
            action_type: The v4.0 action string (e.g. "PROFILE_TABLE").
            profile_call_count: How many times this table was profiled today
                (1 = first call, 2 = second, etc.).
            col_op: (column, operation) tuple for CLEAN_COLUMN.
            today_anomalies: Anomaly checklist entries for the current day.
            dupes_removed: Number of duplicate rows removed by DEDUPLICATE.
            column_exists_in_raw: Whether the column exists in today's raw
                data (for EVOLVE_SILVER_SCHEMA).
            quarantine_condition: The SQL-like condition used for QUARANTINE_ROWS.
            is_blocked: Whether the action was blocked.

        Returns:
            Scalar float reward.
        """
        if is_blocked:
            return REWARD_TABLE["block_penalty"]

        reward = REWARD_TABLE["step_cost"]  # always applied

        if action_type == "PROFILE_TABLE":
            if profile_call_count >= 2:
                reward = REWARD_TABLE["profile_2nd_call"]

        elif action_type == "CLEAN_COLUMN":
            if today_anomalies and col_op in today_anomalies:
                reward += REWARD_TABLE["clean_checklist_hit"]

        elif action_type == "DEDUPLICATE":
            if dupes_removed > 0:
                reward += REWARD_TABLE["deduplicate_effective"]

        elif action_type == "EVOLVE_SILVER_SCHEMA":
            if column_exists_in_raw:
                reward += REWARD_TABLE["evolve_schema_valid"]
            else:
                reward = REWARD_TABLE["evolve_schema_invalid"]

        elif action_type == "QUARANTINE_ROWS":
            if today_anomalies:
                for col, op in today_anomalies:
                    if op == "quarantine" and col.lower() in quarantine_condition.lower():
                        reward += REWARD_TABLE["quarantine_checklist"]
                        break

        elif action_type == "EXECUTE_MERGE":
            if merge_added_rows:
                reward += REWARD_TABLE["execute_merge_success"]

        return reward

    @staticmethod
    def commit_reward(current_day: int) -> float:
        """Reward for a successful COMMIT_DAY.  ``+day_number * 5.0``."""
        return float(current_day) * 5.0

    @staticmethod
    def crash_reward(current_day: int) -> float:
        """Terminal crash penalty. Proportional to remaining days."""
        remaining_days = 30 - current_day
        return -(remaining_days * 5.0)

    @staticmethod
    def completion_bonus() -> float:
        """Day 30 completion bonus (if quarantine ceiling met)."""
        return REWARD_TABLE["completion_bonus"]

    # ------------------------------------------------------------------
    # Legacy backward compat
    # ------------------------------------------------------------------

    def evaluate(
        self,
        action: str,
        metrics: dict | None = None,
        state: object | None = None,
    ) -> float:
        """Legacy API for ``_handle_legacy_action``.

        Maps the old Phase-1 ``(action, metrics, state)`` call signature
        to the v4.0 reward table.  Returns step_cost for most actions.
        """
        metrics = metrics or {}
        reward = REWARD_TABLE["step_cost"]

        if "match_rate" in metrics and metrics.get("match_rate", 0) > 0.80:
            reward += 25.0
        if metrics.get("explosion_detected"):
            reward = -100.0
        if "APPLY_SCD_2" in action.upper():
            reward += 5.0

        return reward
