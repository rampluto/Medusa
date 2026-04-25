"""MEDUSA deterministic post-commit grader — v4.0.

Runs deterministic Python assertions after the agent issues COMMIT_DAY.
Returns a ``GraderResult`` with pass/fail + diagnostic report.

The grader has **no reward logic** — that lives in ``_do_commit`` which
uses the pass/fail flag to assign +Day or -100.

Column names are NEVER hardcoded here.  The ``audit()`` method receives
``pk_col`` (the primary-key column) and ``numeric_cols`` (list of numeric
columns) from the environment / MedusaState so it works with any dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Set

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# GraderResult
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    """Outcome of the post-commit deterministic audit."""

    passed: bool = False
    freshness_ok: bool = False      # Silver grew since day start
    schema_ok: bool = False         # Silver columns ⊇ contract
    type_integrity_ok: bool = False # All numeric-role columns are float/int dtype
    null_integrity_ok: bool = False # No NULL pk_col rows in Silver (Day 28+)
    failures: List[str] = field(default_factory=list)
    report: str = ""


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """Post-commit deterministic audit following MEDUSA v4.0 spec.

    Four checks run on every COMMIT_DAY:
      1. Freshness     — Silver must have grown since day start.
      2. Schema        — Silver columns must be a superset of the Data Contract.
      3. Type integrity— All detected numeric columns must have numeric dtype.
      4. Null integrity— No NULL pk_col rows in Silver (Day 28+).

    No column names are hardcoded.  Pass ``pk_col`` and ``numeric_cols``
    (populated by the generator into MedusaState) at call time.
    """

    def audit(
        self,
        silver: pd.DataFrame,
        silver_at_day_start: int,
        current_day: int,
        contract_columns: List[str],
        merged_today: bool = False,
        pk_col: str = "",
        numeric_cols: List[str] | None = None,
    ) -> GraderResult:
        """Run all four grader checks.

        Args:
            silver:              The cumulative Silver DataFrame at commit time.
            silver_at_day_start: Row count of Silver at the start of this day.
            current_day:         Which day (1–30) is being committed.
            contract_columns:    The current Data Contract column list.
            merged_today:        True if EXECUTE_MERGE ran this day.
            pk_col:              Primary-key column name (from MedusaState.pk_col).
            numeric_cols:        Numeric-role column names (from MedusaState.numeric_cols).

        Returns:
            GraderResult with individual check statuses and report.
        """
        result = GraderResult()
        numeric_cols = numeric_cols or []

        # ── 1. Freshness Check ───────────────────────────────────────
        silver_len = len(silver)
        if silver_len > silver_at_day_start or merged_today:
            result.freshness_ok = True
        else:
            result.failures.append(
                f"FRESHNESS_FAIL: Silver {silver_len} rows, "
                f"was {silver_at_day_start} at day start."
            )

        # ── 2. Schema Check ─────────────────────────────────────────
        if not silver.empty:
            actual_cols = set(silver.columns)
            expected_cols = set(contract_columns)
            if expected_cols.issubset(actual_cols):
                result.schema_ok = True
            else:
                missing = sorted(expected_cols - actual_cols)
                result.failures.append(
                    f"SCHEMA_FAIL: Missing columns: {missing}."
                )
        else:
            result.failures.append("SCHEMA_FAIL: Silver is empty.")

        # ── 3. Type Integrity (role-based, persistent) ───────────────
        # All numeric-role columns that exist in Silver must have numeric dtype.
        type_failures = []
        checked = False
        for col in numeric_cols:
            if col in silver.columns:
                checked = True
                if not pd.api.types.is_numeric_dtype(silver[col]):
                    type_failures.append(
                        f"TYPE_FAIL: column '{col}' dtype is {silver[col].dtype}, "
                        f"expected numeric."
                    )
        if not type_failures:
            result.type_integrity_ok = True
        else:
            result.failures.extend(type_failures)

        # ── 4. Null Integrity (Day 28+, pk_col-based) ────────────────
        if current_day >= 28 and pk_col and not silver.empty and pk_col in silver.columns:
            null_count = int(silver[pk_col].isnull().sum())
            if null_count > 0:
                result.failures.append(
                    f"NULL_FAIL: {null_count} NULL '{pk_col}' rows in Silver "
                    f"(should be quarantined)."
                )
            else:
                result.null_integrity_ok = True
        else:
            result.null_integrity_ok = True

        # ── Result ───────────────────────────────────────────────────
        result.passed = all([
            result.freshness_ok,
            result.schema_ok,
            result.type_integrity_ok,
            result.null_integrity_ok,
        ])
        result.report = _build_report(result)
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_report(result: GraderResult) -> str:
    lines = ["=== MEDUSA v4.0 Grader Audit ==="]
    lines.append(f"  Freshness OK:      {'✓' if result.freshness_ok else '✗'}")
    lines.append(f"  Schema OK:         {'✓' if result.schema_ok else '✗'}")
    lines.append(f"  Type Integrity OK: {'✓' if result.type_integrity_ok else '✗'}")
    lines.append(f"  Null Integrity OK: {'✓' if result.null_integrity_ok else '✗'}")
    if result.failures:
        lines.append("  Failures:")
        for f in result.failures:
            lines.append(f"    - {f}")
    lines.append(f"  {'PASS ✓' if result.passed else 'FAIL ✗'}")
    return "\n".join(lines)
