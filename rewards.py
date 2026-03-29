"""MEDUSA reward engine.

Reward model as defined in the MEDUSA blueprint. All reward logic is in a
single ``RewardEngine`` class so it can be unit-tested in isolation from the
environment.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Reward table (blueprint §3)
# ---------------------------------------------------------------------------

REWARD_TABLE: Dict[str, float] = {
    "high_match_join": +25.0,      # match_rate > 0.90
    "correct_scd2": +5.0,          # SCD-2 used on a tracked column
    "quarantine_precision": +10.0, # Orphaned rows correctly moved to quarantine
    "row_explosion": -100.0,       # Cartesian product detected
    "dirty_join": -30.0,           # Join attempted without PREP_KEYS → 0-row result
    "stale_processing": -15.0,     # Action taken while source is stale (not synced first)
    "step_penalty": -0.2,          # Per-step efficiency penalty
}

HIGH_MATCH_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# RewardEngine
# ---------------------------------------------------------------------------

class RewardEngine:
    """Compute per-step reward from action context and operator metrics."""

    def evaluate(
        self,
        action_type: str,
        metrics: Dict[str, Any],
        state_before: Any,  # MedusaState snapshot before step
    ) -> float:
        """Return the scalar reward for a single step.

        Args:
            action_type: The ``MedusaActionType`` value string (e.g. "SYNC_CHECK").
            metrics: Dictionary returned by the corresponding operator.
            state_before: State object *before* this step was applied.

        Returns:
            Scalar float reward.
        """
        reward = REWARD_TABLE["step_penalty"]  # always applied

        if action_type == "SYNC_CHECK":
            # No positive/negative signal from sync_check itself
            pass

        elif action_type in ("PREP_KEYS_A", "PREP_KEYS_B"):
            # Neutral — prep is just a prerequisite
            pass

        elif action_type == "DEDUPLICATE_B":
            pass

        elif action_type == "EVOLVE_SCHEMA":
            pass

        elif action_type in ("EXECUTE_JOIN_INNER", "EXECUTE_JOIN_LEFT", "EXECUTE_JOIN_ANTI"):
            explosion = metrics.get("explosion_detected", False)
            if explosion:
                reward += REWARD_TABLE["row_explosion"]
            else:
                join_rows = metrics.get("join_rows", 0)
                fact_rows = metrics.get("fact_rows", 1)
                # "Dirty join" = join executed without PREP_KEYS and produced 0 rows
                # even though the source was non-empty
                if join_rows == 0 and fact_rows > 0:
                    if not state_before.did_prep_a or not state_before.did_prep_b:
                        reward += REWARD_TABLE["dirty_join"]
                else:
                    match_rate = metrics.get("match_rate", 0.0)
                    if match_rate >= HIGH_MATCH_THRESHOLD:
                        reward += REWARD_TABLE["high_match_join"]

                    # Quarantine precision: reward if orphans were quarantined
                    quarantine_rows = metrics.get("quarantine_rows", 0)
                    if quarantine_rows > 0 and action_type == "EXECUTE_JOIN_LEFT":
                        reward += REWARD_TABLE["quarantine_precision"]

            # Stale processing: ran join while a source was stale (never synced)
            if (state_before.is_stale_a or state_before.is_stale_b) and not state_before.did_sync_check:
                reward += REWARD_TABLE["stale_processing"]

        elif action_type in ("APPLY_SCD_1", "APPLY_SCD_2"):
            if action_type == "APPLY_SCD_2":
                # Reward if SCD-2 was the right choice (tracked col involved)
                reward += REWARD_TABLE["correct_scd2"]

            if (state_before.is_stale_a or state_before.is_stale_b) and not state_before.did_sync_check:
                reward += REWARD_TABLE["stale_processing"]

        elif action_type == "COMMIT":
            # Base commit — grader adds bonus/penalty separately
            pass

        return reward
