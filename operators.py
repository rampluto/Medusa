"""MEDUSA ETL operators.

Each operator is a stateless function that takes DataFrame(s) and returns a
(result_df_or_None, metrics_dict) tuple. The environment calls these from
``step()`` and passes the metrics to the reward engine.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Metrics = Dict[str, Any]
OpResult = Tuple[Optional[pd.DataFrame], Metrics]


# ---------------------------------------------------------------------------
# Operator: sync_check
# ---------------------------------------------------------------------------

def sync_check(
    bronze_a: pd.DataFrame,
    bronze_b: pd.DataFrame,
    time_delta_a: float,
    time_delta_b: float,
    stale_threshold_hours: float = 6.0,
) -> OpResult:
    """Inspect freshness of both sources.

    Returns metrics about staleness without modifying any data.
    """
    is_stale_a = time_delta_a > stale_threshold_hours
    is_stale_b = time_delta_b > stale_threshold_hours
    metrics: Metrics = {
        "time_delta_a": time_delta_a,
        "time_delta_b": time_delta_b,
        "is_stale_a": is_stale_a,
        "is_stale_b": is_stale_b,
        "rows_a": len(bronze_a),
        "rows_b": len(bronze_b),
    }
    return None, metrics


# ---------------------------------------------------------------------------
# Operator: evolve_schema
# ---------------------------------------------------------------------------

def evolve_schema(
    silver: pd.DataFrame,
    bronze_a: pd.DataFrame,
    bronze_b: pd.DataFrame,
    new_cols_a: list[str],
    new_cols_b: list[str],
) -> OpResult:
    """Add new columns (from schema drift) to the Silver DataFrame.

    Fills missing historical rows with NaN.
    """
    added: list[str] = []
    result = silver.copy()

    for col in new_cols_a + new_cols_b:
        if col not in result.columns:
            result[col] = pd.NA
            added.append(col)

    metrics: Metrics = {
        "cols_added": added,
        "new_cols_count": len(added),
        "silver_col_count": len(result.columns),
    }
    return result, metrics


# ---------------------------------------------------------------------------
# Operator: prep_keys
# ---------------------------------------------------------------------------

def prep_keys(df: pd.DataFrame, key_col: str) -> OpResult:
    """Cast, strip whitespace, and null-fill the join key column.

    Returns a cleaned copy of ``df`` with metrics about how many rows were
    affected.
    """
    result = df.copy()
    original_nulls = result[key_col].isna().sum()
    original_len = len(result)

    # Strip whitespace (treat blank strings as nulls)
    result[key_col] = result[key_col].astype(str).str.strip()
    result[key_col] = result[key_col].replace({"None": pd.NA, "nan": pd.NA, "": pd.NA})

    # Cast to string (uniform type for join)
    result[key_col] = result[key_col].astype("string")

    after_nulls = result[key_col].isna().sum()
    null_ratio_before = original_nulls / max(original_len, 1)
    null_ratio_after = int(after_nulls) / max(original_len, 1)

    metrics: Metrics = {
        "null_ratio_before": null_ratio_before,
        "null_ratio_after": null_ratio_after,
        "rows_trimmed": original_len - int(after_nulls),
        "null_rows_dropped": 0,  # We do NOT drop nulls; grader catches orphans
    }
    return result, metrics


# ---------------------------------------------------------------------------
# Operator: deduplicate
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame, key_col: str) -> OpResult:
    """Ensure Dimension (Source B) is unique on ``key_col``.

    Keeps the last occurrence so the most-recent record wins.
    """
    original_len = len(df)
    result = df.drop_duplicates(subset=[key_col], keep="last").reset_index(drop=True)
    dupes_removed = original_len - len(result)

    non_null = result[key_col].notna().sum()
    uniqueness = non_null / max(len(result), 1)

    metrics: Metrics = {
        "dupes_removed": dupes_removed,
        "uniqueness": float(uniqueness),
        "rows_after": len(result),
    }
    return result, metrics


# ---------------------------------------------------------------------------
# Operator: execute_join
# ---------------------------------------------------------------------------

_EXPLOSION_MULTIPLIER = 1.05  # > 5% extra rows triggers explosion alert


def execute_join(
    fact: pd.DataFrame,
    dim: pd.DataFrame,
    key_col: str,
    join_type: str,  # "inner" | "left" | "anti"
) -> Tuple[pd.DataFrame, pd.DataFrame, Metrics]:
    """Join Fact (A) with Dimension (B).

    Returns (joined_df, quarantine_df, metrics).
    ``quarantine_df`` contains rows from A that did not match B (orphans).
    """
    # Drop null-keyed rows from both before joining
    fact_clean = fact.dropna(subset=[key_col])
    dim_clean = dim.dropna(subset=[key_col])

    # Compute match rate before join
    fact_keys = set(fact_clean[key_col].astype(str))
    dim_keys = set(dim_clean[key_col].astype(str))
    overlap = fact_keys & dim_keys
    match_rate = len(overlap) / max(len(fact_keys), 1)

    if join_type == "anti":
        # Anti-join: rows in A NOT in B → goes to quarantine
        mask = ~fact_clean[key_col].astype(str).isin(dim_keys)
        joined = pd.DataFrame(columns=list(fact_clean.columns) + [
            c for c in dim_clean.columns if c != key_col
        ])
        quarantine = fact_clean[mask].copy()
    elif join_type == "inner":
        merged = fact_clean.merge(dim_clean, on=key_col, how="inner",
                                   suffixes=("_a", "_b"))
        quarantine = fact_clean[~fact_clean[key_col].astype(str).isin(dim_keys)].copy()
        joined = merged
    else:  # left
        merged = fact_clean.merge(dim_clean, on=key_col, how="left",
                                   suffixes=("_a", "_b"))
        # Quarantine = rows where all dim columns are NaN (no match)
        dim_cols = [c for c in dim_clean.columns if c != key_col]
        if dim_cols:
            no_match_mask = merged[dim_cols[0]].isna() if dim_cols else pd.Series(False, index=merged.index)
        else:
            no_match_mask = pd.Series(False, index=merged.index)
        quarantine = merged[no_match_mask][[key_col]].copy()
        joined = merged

    # Explosion detection
    explosion = len(joined) > len(fact_clean) * _EXPLOSION_MULTIPLIER

    metrics: Metrics = {
        "join_type": join_type,
        "fact_rows": len(fact_clean),
        "dim_rows": len(dim_clean),
        "join_rows": len(joined),
        "quarantine_rows": len(quarantine),
        "match_rate": match_rate,
        "explosion_detected": explosion,
    }
    return joined, quarantine, metrics


# ---------------------------------------------------------------------------
# Operator: apply_scd
# ---------------------------------------------------------------------------

def apply_scd(
    silver: pd.DataFrame,
    joined: pd.DataFrame,
    key_col: str,
    tracked_col: str,
    scd_type: int,  # 1 or 2
) -> OpResult:
    """Merge ``joined`` result into Silver using SCD-1 or SCD-2.

    SCD-1: overwrite existing records.
    SCD-2: close old records (valid_to = now) and insert new ones with
           a new valid_from / valid_to = None (open record).
    """
    now = datetime.datetime.now(datetime.UTC)
    inserts = 0
    updates = 0

    if joined.empty:
        metrics: Metrics = {
            "scd_type": scd_type,
            "inserts": 0,
            "updates": 0,
            "silver_rows": len(silver),
        }
        return silver, metrics

    if silver.empty:
        # First load — treat everything as inserts
        result = joined.copy()
        if scd_type == 2:
            result["valid_from"] = now
            result["valid_to"] = pd.NaT
            result["is_current"] = True
        inserts = len(result)
        metrics = {
            "scd_type": scd_type,
            "inserts": inserts,
            "updates": 0,
            "silver_rows": len(result),
        }
        return result, metrics

    if scd_type == 1:
        # Upsert: overwrite matching records
        exists_mask = silver[key_col].isin(joined[key_col])
        new_keys_mask = ~joined[key_col].isin(silver[key_col])

        result = silver[~exists_mask].copy()
        result = pd.concat([result, joined], ignore_index=True)

        updates = int(exists_mask.sum())
        inserts = int(new_keys_mask.sum())

    else:  # SCD-2
        # Ensure Silver has timestamp columns
        if "valid_from" not in silver.columns:
            silver = silver.copy()
            silver["valid_from"] = now - datetime.timedelta(days=30)
            silver["valid_to"] = pd.NaT
            silver["is_current"] = True

        silver_result = silver.copy()
        new_rows: list[pd.DataFrame] = []

        for _, new_row in joined.iterrows():
            key_val = new_row[key_col]
            current_mask = (silver_result[key_col] == key_val) & (silver_result["is_current"] == True)  # noqa: E712
            current_rows = silver_result[current_mask]

            if current_rows.empty:
                # New record
                row_df = pd.DataFrame([new_row])
                row_df["valid_from"] = now
                row_df["valid_to"] = pd.NaT
                row_df["is_current"] = True
                new_rows.append(row_df)
                inserts += 1
            else:
                # Check if tracked column changed
                old_val = current_rows.iloc[0].get(tracked_col)
                new_val = new_row.get(tracked_col)
                if old_val != new_val:
                    # Close old record
                    silver_result.loc[current_mask, "valid_to"] = now
                    silver_result.loc[current_mask, "is_current"] = False
                    # Insert new record
                    row_df = pd.DataFrame([new_row])
                    row_df["valid_from"] = now
                    row_df["valid_to"] = pd.NaT
                    row_df["is_current"] = True
                    new_rows.append(row_df)
                    updates += 1

        if new_rows:
            silver_result = pd.concat([silver_result] + new_rows, ignore_index=True)
        result = silver_result

    metrics = {
        "scd_type": scd_type,
        "inserts": inserts,
        "updates": updates,
        "silver_rows": len(result),
    }
    return result, metrics
