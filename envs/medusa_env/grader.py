"""MEDUSA deterministic post-commit grader.

Runs a four-check audit after the agent issues COMMIT and returns a
``GraderResult`` that feeds a bonus/penalty into the terminal reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import pandas as pd

if TYPE_CHECKING:
    from .scenarios import Scenario


# ---------------------------------------------------------------------------
# GraderResult
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    """Outcome of the post-commit audit."""

    passed: bool = False
    volume_ok: bool = False    # Silver rows ≤ Source A rows (no duplicates from join)
    integrity_ok: bool = False # Quarantine holds only true orphans
    schema_ok: bool = False    # Silver has union of required columns
    history_ok: bool = False   # SCD-2 timestamps non-overlapping
    failures: List[str] = field(default_factory=list)
    bonus_reward: float = 0.0
    report: str = ""


# Reward tuning
_BONUS_ALL_PASS = +15.0
_PENALTY_ALL_FAIL = -20.0
_BONUS_PER_CHECK = +3.0
_PENALTY_PER_FAIL = -5.0


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """Post-commit deterministic audit following MEDUSA spec §4."""

    def audit(
        self,
        silver: pd.DataFrame,
        quarantine: pd.DataFrame,
        bronze_a: pd.DataFrame,
        bronze_b: pd.DataFrame,
        join_key: str,
        join_type: str,
        scd_type: int,
        scenario: "Scenario",
    ) -> GraderResult:
        """Run all four grader checks and compute bonus reward.

        Args:
            silver: The final Silver DataFrame after SCD merge.
            quarantine: Rows from A that did not match B.
            bronze_a: Original fact source (pre-cleaning).
            bronze_b: Original dimension source (pre-cleaning).
            join_key: Column used for the join.
            join_type: "inner" | "left" | "anti"
            scd_type: 1 or 2
            scenario: The current episode's scenario (has tracked_cols etc.)

        Returns:
            GraderResult with individual check statuses and bonus_reward.
        """
        result = GraderResult()

        # ── 1. Volume Check ──────────────────────────────────────────────
        # For left joins, Silver should not exceed Source A row count.
        if join_type == "left":
            source_a_rows = len(bronze_a.dropna(subset=[join_key]))
            silver_rows = len(silver[silver.get("is_current", pd.Series(True, index=silver.index)) == True]) if "is_current" in silver.columns else len(silver)  # noqa: E712
            result.volume_ok = silver_rows <= source_a_rows * 1.05  # 5% tolerance
            if not result.volume_ok:
                result.failures.append(
                    f"VOLUME_FAIL: Silver {silver_rows} rows > Source A {source_a_rows} rows"
                )
        else:
            result.volume_ok = True  # Not applicable for inner/anti joins

        # ── 2. Integrity Check ───────────────────────────────────────────
        # Quarantine rows should be true orphans (no match in B even after cleaning).
        if not quarantine.empty and join_key in quarantine.columns:
            dim_keys = set(bronze_b[join_key].dropna().astype(str).str.strip())
            quarantine_keys = set(quarantine[join_key].dropna().astype(str).str.strip())
            # Orphan = quarantine key truly not in dim
            could_join = quarantine_keys & dim_keys
            if could_join:
                result.integrity_ok = False
                result.failures.append(
                    f"INTEGRITY_FAIL: {len(could_join)} quarantine row(s) could have "
                    f"been joined if keys were cleaned."
                )
            else:
                result.integrity_ok = True
        else:
            result.integrity_ok = True  # Empty quarantine is fine

        # ── 3. Schema Check ──────────────────────────────────────────────
        # Silver must contain all required columns from A and B.
        required_from_a = [c for c in bronze_a.columns if c != join_key]
        required_from_b = [c for c in bronze_b.columns if c != join_key]
        required = set(required_from_a + required_from_b + scenario.new_cols_a + scenario.new_cols_b)
        silver_cols = set(silver.columns)
        missing = required - silver_cols
        if missing:
            result.schema_ok = False
            result.failures.append(f"SCHEMA_FAIL: Missing columns in Silver: {sorted(missing)}")
        else:
            result.schema_ok = True

        # ── 4. History Check (SCD-2 only) ────────────────────────────────
        if scd_type == 2 and "valid_from" in silver.columns and "valid_to" in silver.columns:
            overlap_found = False
            for key_val, group in silver.groupby(join_key):
                if len(group) < 2:
                    continue
                closed = group[group["valid_to"].notna()].sort_values("valid_from")
                for i in range(len(closed) - 1):
                    vt_i = closed.iloc[i]["valid_to"]
                    vf_next = closed.iloc[i + 1]["valid_from"]
                    if pd.notna(vt_i) and pd.notna(vf_next) and vt_i > vf_next:
                        overlap_found = True
                        break
                if overlap_found:
                    break
            if overlap_found:
                result.history_ok = False
                result.failures.append("HISTORY_FAIL: SCD-2 timestamps overlap for some keys.")
            else:
                result.history_ok = True
        else:
            result.history_ok = True  # Not applicable for SCD-1

        # ── Compute bonus ────────────────────────────────────────────────
        checks = [result.volume_ok, result.integrity_ok, result.schema_ok, result.history_ok]
        passed_count = sum(checks)
        failed_count = len(checks) - passed_count

        result.passed = all(checks)

        if result.passed:
            result.bonus_reward = _BONUS_ALL_PASS
        elif failed_count == len(checks):
            result.bonus_reward = _PENALTY_ALL_FAIL
        else:
            result.bonus_reward = passed_count * _BONUS_PER_CHECK - failed_count * _PENALTY_PER_FAIL

        result.report = _build_report(result)
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_report(result: GraderResult) -> str:
    lines = ["=== MEDUSA Grader Audit ==="]
    lines.append(f"  Volume OK:    {'✓' if result.volume_ok else '✗'}")
    lines.append(f"  Integrity OK: {'✓' if result.integrity_ok else '✗'}")
    lines.append(f"  Schema OK:    {'✓' if result.schema_ok else '✗'}")
    lines.append(f"  History OK:   {'✓' if result.history_ok else '✗'}")
    lines.append(f"  Bonus Reward: {result.bonus_reward:+.1f}")
    if result.failures:
        lines.append("  Failures:")
        for f in result.failures:
            lines.append(f"    - {f}")
    lines.append(f"  {'PASS ✓' if result.passed else 'FAIL ✗'}")
    return "\n".join(lines)
