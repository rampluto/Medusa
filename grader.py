"""MEDUSA deterministic post-commit grader — v4.0.

Runs deterministic Python assertions after the agent issues COMMIT_DAY.
Returns a ``GraderResult`` with pass/fail + diagnostic report.

The grader has **no reward logic** — that lives in ``_do_commit`` which
uses the pass/fail flag to assign +Day or -100.
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
    type_integrity_ok: bool = False # revenue dtype == float64
    null_integrity_ok: bool = False # No NULL user_id in Silver (Day 28+)
    failures: List[str] = field(default_factory=list)
    report: str = ""


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class Grader:
    """Post-commit deterministic audit following MEDUSA v4.0 spec.

    Four checks run on every COMMIT_DAY:
      1. Freshness — Silver must have grown since day start.
      2. Schema — Silver columns must be a superset of the Data Contract.
      3. Type integrity — ``revenue`` must be float64 (persistent).
      4. Null integrity — No NULL ``user_id`` rows in Silver (Day 28+).
    """

    def audit(
        self,
        silver: pd.DataFrame,
        silver_at_day_start: int,
        current_day: int,
        contract_columns: List[str],
    ) -> GraderResult:
        """Run all four grader checks.

        Args:
            silver: The cumulative Silver DataFrame at commit time.
            silver_at_day_start: Row count of Silver at the start of this day.
            current_day: Which day (1–30) is being committed.
            contract_columns: The current Data Contract column list.

        Returns:
            GraderResult with individual check statuses and report.
        """
        result = GraderResult()

        # ── 1. Freshness Check ───────────────────────────────────────
        silver_len = len(silver)
        if silver_len > silver_at_day_start:
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

        # ── 3. Type Integrity (persistent) ───────────────────────────
        if not silver.empty and "price" in silver.columns:
            if silver["price"].dtype == np.float64:
                result.type_integrity_ok = True
            else:
                result.failures.append(
                    f"TYPE_FAIL: price dtype is {silver['price'].dtype}, "
                    f"expected float64."
                )
        else:
            # No price column = not applicable, passes by default
            result.type_integrity_ok = True

        # ── 4. Null Integrity (Day 28+) ──────────────────────────────
        if current_day >= 28 and not silver.empty and "customer_id" in silver.columns:
            null_count = int(silver["customer_id"].isnull().sum())
            if null_count > 0:
                result.failures.append(
                    f"NULL_FAIL: {null_count} NULL customer_id rows in Silver "
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
