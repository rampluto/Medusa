"""MEDUSA ETL operators — v4.0.

Each v4.0 operator is a stateless function that takes DataFrame(s) and returns
a result tuple.  The environment's ``_do_*`` methods delegate data
transformations here and keep book-keeping (state, rewards) at the env level.

Legacy Phase-1 operators (sync_check, prep_keys, execute_join, apply_scd) are
retained at the bottom for backward compatibility with existing tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Type aliases
Metrics = Dict[str, Any]
OpResult = Tuple[Optional[pd.DataFrame], Metrics]


# ═══════════════════════════════════════════════════════════════════════════
# v4.0 Operators (used by MedusaEnv._do_* methods)
# ═══════════════════════════════════════════════════════════════════════════


def profile_table(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Profile a DataFrame: return per-column dtype, null %, unique count, duplicates.

    Args:
        df: The daily cleaned DataFrame to profile.

    Returns:
        Dict mapping column name → {dtype, null_pct, n_unique, n_duplicates}.
    """
    profile: Dict[str, Dict[str, Any]] = {}
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
    return profile


def clean_column(
    df: pd.DataFrame, col: str, op: str
) -> Tuple[pd.DataFrame, int]:
    """Apply a cleaning operation to a column.

    Args:
        df: DataFrame to modify (modified in-place and returned).
        col: Column name.
        op: One of ``"strip"``, ``"cast"``, or ``"fill_zero"``.

    Returns:
        (modified_df, rows_affected)
    """
    rows_affected = 0

    if op == "strip":
        before = df[col].copy()
        df[col] = df[col].apply(
            lambda x: str(x).strip() if pd.notna(x) else x
        )
        rows_affected = int((before != df[col]).sum())

    elif op == "cast":
        # Handle "$50.50" → 50.50 style strings
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

    return df, rows_affected


def deduplicate_rows(
    df: pd.DataFrame, key: str
) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows from a DataFrame.

    Args:
        df: DataFrame to deduplicate.
        key: Column to use as the uniqueness key.

    Returns:
        (deduped_df, dupes_removed)
    """
    before_len = len(df)
    if key in df.columns:
        df = df.drop_duplicates(subset=[key], keep="last")
    else:
        df = df.drop_duplicates(keep="last")
    dupes_removed = before_len - len(df)
    return df, dupes_removed


def quarantine_rows(
    df: pd.DataFrame, condition: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into kept and quarantined rows.

    Args:
        df: DataFrame to filter.
        condition: SQL-like condition (e.g. "user_id IS NULL").

    Returns:
        (kept_df, quarantined_df)
    """
    quarantined = pd.DataFrame()
    kept = df.copy()

    if "IS NULL" in condition.upper() and "IS NOT NULL" not in condition.upper():
        col_name = condition.upper().replace("IS NULL", "").strip()
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

    return kept, quarantined


def merge_into_silver(
    silver: pd.DataFrame,
    daily: pd.DataFrame,
    key: str = "user_id",
) -> pd.DataFrame:
    """Upsert daily batch into cumulative Silver.

    - New keys → appended.
    - Existing keys → updated (SCD-1 overwrite).

    Args:
        silver: Current cumulative Silver DataFrame.
        daily: Today's cleaned/merged batch.
        key: Column used for key-based upsert.

    Returns:
        Updated Silver DataFrame.
    """
    # Align schemas
    for col in silver.columns:
        if col not in daily.columns:
            daily[col] = np.nan
    for col in daily.columns:
        if not silver.empty and col not in silver.columns:
            silver[col] = np.nan

    if silver.empty:
        return daily.copy()

    if key not in daily.columns or key not in silver.columns:
        return pd.concat([silver, daily], ignore_index=True)

    # Upsert: update existing, append new
    existing_keys = set(silver[key].dropna())
    new_mask = ~daily[key].isin(existing_keys) | daily[key].isna()
    new_rows = daily[new_mask]
    update_rows = daily[~new_mask]

    if not update_rows.empty:
        silver = silver.set_index(key)
        silver.update(update_rows.set_index(key))
        silver = silver.reset_index()

    if not new_rows.empty:
        silver = pd.concat([silver, new_rows], ignore_index=True)

    return silver


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Phase-1 Operators (backward compat)
# ═══════════════════════════════════════════════════════════════════════════


_STALE_THRESHOLD_HOURS = 6.0
_EXPLOSION_MULTIPLIER = 1.05  # > 5% extra rows triggers explosion alert


def sync_check(
    bronze_a: pd.DataFrame,
    bronze_b: pd.DataFrame,
    time_delta_a: float,
    time_delta_b: float,
    stale_threshold_hours: float = _STALE_THRESHOLD_HOURS,
) -> OpResult:
    """Inspect freshness of both sources.

    Returns metrics about staleness without modifying any data.
    """
    is_stale_a = time_delta_a > stale_threshold_hours
    is_stale_b = time_delta_b > stale_threshold_hours
    metrics = {
        "time_delta_a": time_delta_a,
        "time_delta_b": time_delta_b,
        "is_stale_a": is_stale_a,
        "is_stale_b": is_stale_b,
    }
    return None, metrics


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
    new_count = 0
    for col in new_cols_a:
        if col in bronze_a.columns and col not in silver.columns:
            silver[col] = np.nan
            new_count += 1
    for col in new_cols_b:
        if col in bronze_b.columns and col not in silver.columns:
            silver[col] = np.nan
            new_count += 1
    return silver, {"new_cols_count": new_count}


def prep_keys(df: pd.DataFrame, key_col: str) -> OpResult:
    """Cast, strip whitespace, and null-fill the join key column.

    Returns a cleaned copy of ``df`` with metrics about how many rows were
    affected.
    """
    result = df.copy()
    null_before = float(result[key_col].isna().mean())
    n_rows = len(result)

    # 1. Cast to str
    result[key_col] = result[key_col].astype(str)
    # 2. Strip whitespace
    result[key_col] = result[key_col].str.strip()
    # 3. Replace "None" / "nan" → actual NaN
    result[key_col] = result[key_col].replace({"None": np.nan, "nan": np.nan})
    null_after = float(result[key_col].isna().mean())

    metrics = {
        "null_ratio_before": null_before,
        "null_ratio_after": null_after,
        "uniqueness": float(result[key_col].nunique() / max(n_rows, 1)),
        "rows_processed": n_rows,
    }
    return result, metrics


def deduplicate(df: pd.DataFrame, key_col: str) -> OpResult:
    """Ensure Dimension (Source B) is unique on ``key_col``.

    Keeps the last occurrence so the most-recent record wins.
    """
    before = len(df)
    result = df.drop_duplicates(subset=[key_col], keep="last")
    return result, {"dupes_removed": before - len(result)}


def execute_join(
    fact: pd.DataFrame,
    dim: pd.DataFrame,
    key_col: str,
    join_type: str,  # "inner" | "left" | "anti"
) -> tuple[pd.DataFrame, pd.DataFrame, Metrics]:
    """Join Fact (A) with Dimension (B).

    Returns (joined_df, quarantine_df, metrics).
    ``quarantine_df`` contains rows from A that did not match B (orphans).
    """
    fact_rows = len(fact)

    if join_type == "anti":
        merged = fact.merge(dim[[key_col]], on=key_col, how="left", indicator=True)
        quarantine = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        joined = pd.DataFrame()
        metrics: Metrics = {
            "join_type": "anti",
            "fact_rows": fact_rows,
            "join_rows": 0,
            "quarantine_rows": len(quarantine),
            "match_rate": 0.0,
            "explosion_detected": False,
        }
        return joined, quarantine, metrics

    how = join_type
    joined = fact.merge(dim, on=key_col, how=how)
    join_rows = len(joined)

    explosion = join_rows > fact_rows * _EXPLOSION_MULTIPLIER

    quarantine = pd.DataFrame()
    if how == "left":
        dim_keys = set(dim[key_col].dropna())
        orphan_mask = ~fact[key_col].isin(dim_keys) & fact[key_col].notna()
        quarantine = fact[orphan_mask]

    match_keys = set(fact[key_col].dropna()) & set(dim[key_col].dropna())
    match_rate = len(match_keys) / max(len(fact[key_col].dropna()), 1)

    metrics = {
        "join_type": join_type,
        "fact_rows": fact_rows,
        "join_rows": join_rows,
        "quarantine_rows": len(quarantine),
        "match_rate": round(match_rate, 4),
        "explosion_detected": explosion,
    }

    return joined, quarantine, metrics


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
    now = datetime.datetime.now(datetime.timezone.utc)
    inserts = 0
    updates = 0

    if scd_type == 1:
        if silver.empty:
            result = joined.copy()
            inserts = len(result)
        else:
            result = silver.copy()
            for _, row in joined.iterrows():
                key_val = row[key_col]
                mask = result[key_col] == key_val
                if mask.any():
                    for col in joined.columns:
                        if col != key_col:
                            result.loc[mask, col] = row[col]
                    updates += 1
                else:
                    result = pd.concat(
                        [result, pd.DataFrame([row])], ignore_index=True
                    )
                    inserts += 1
        return result, {"scd_type": 1, "inserts": inserts, "updates": updates}

    # SCD-2
    if silver.empty:
        result = joined.copy()
        result["valid_from"] = now
        result["valid_to"] = pd.NaT
        result["is_current"] = True
        inserts = len(result)
    else:
        result = silver.copy()
        if "valid_from" not in result.columns:
            result["valid_from"] = now
            result["valid_to"] = pd.NaT
            result["is_current"] = True

        new_rows = []
        for _, row in joined.iterrows():
            key_val = row[key_col]
            current_mask = (result[key_col] == key_val) & (
                result.get("is_current", pd.Series(True, index=result.index))
                == True  # noqa: E712
            )
            if current_mask.any():
                old_tracked = result.loc[current_mask, tracked_col].iloc[0]
                if old_tracked != row.get(tracked_col):
                    result.loc[current_mask, "valid_to"] = now
                    result.loc[current_mask, "is_current"] = False
                    new_rec = row.to_dict()
                    new_rec["valid_from"] = now
                    new_rec["valid_to"] = pd.NaT
                    new_rec["is_current"] = True
                    new_rows.append(new_rec)
                    updates += 1
                else:
                    for col in joined.columns:
                        if col not in (key_col, tracked_col):
                            result.loc[current_mask, col] = row[col]
            else:
                new_rec = row.to_dict()
                new_rec["valid_from"] = now
                new_rec["valid_to"] = pd.NaT
                new_rec["is_current"] = True
                new_rows.append(new_rec)
                inserts += 1

        if new_rows:
            result = pd.concat(
                [result, pd.DataFrame(new_rows)], ignore_index=True
            )

    return result, {"scd_type": 2, "inserts": inserts, "updates": updates}
