"""MEDUSA scenario generator.

Produces randomised Bronze A (Fact) and Bronze B (Dimension) DataFrames to
drive each training episode. Four canonical scenarios cover the canonical
failure modes described in the MEDUSA blueprint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """One episode's worth of Bronze source data + configuration."""

    id: str
    bronze_a: pd.DataFrame          # Fact table (source of truth for volume)
    bronze_b: pd.DataFrame          # Dimension table (must be unique on key)
    join_key: str                   # Column name used to join A and B
    tracked_cols: List[str]         # Columns in B that require SCD-2 history
    is_stale_a: bool                # Whether Source A is past the freshness threshold
    is_stale_b: bool
    time_delta_a: float             # Hours since Source A was last refreshed
    time_delta_b: float
    new_cols_a: List[str]           # Extra columns in A not in Silver yet
    new_cols_b: List[str]           # Extra columns in B not in Silver yet
    description: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STALE_THRESHOLD_HOURS = 6.0


def _make_fact(
    rng: random.Random,
    n_rows: int,
    key_col: str,
    null_ratio: float = 0.0,
    extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create a synthetic Fact (Bronze A) DataFrame."""
    keys = [f"K{i:04d}" for i in rng.sample(range(1, n_rows * 2), n_rows)]

    # Inject nulls into the key
    null_mask = rng.sample(range(n_rows), int(n_rows * null_ratio))
    for idx in null_mask:
        keys[idx] = None  # type: ignore[call-overload]

    data = {
        key_col: keys,
        "fact_value": [rng.uniform(0, 1000) for _ in range(n_rows)],
        "fact_category": [rng.choice(["A", "B", "C"]) for _ in range(n_rows)],
        "created_at": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
    }
    for col in (extra_cols or []):
        data[col] = [rng.uniform(0, 100) for _ in range(n_rows)]

    return pd.DataFrame(data)


def _make_dim(
    rng: random.Random,
    n_rows: int,
    key_col: str,
    null_ratio: float = 0.0,
    uniqueness: float = 1.0,   # < 1.0 means some keys are duplicated
    match_keys: Optional[List[str]] = None,  # If given, use these as the key pool
    extra_cols: Optional[List[str]] = None,
    tracked_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create a synthetic Dimension (Bronze B) DataFrame."""
    if match_keys:
        # Choose from overlap pool to control referential integrity
        available = list(match_keys)
        keys = [rng.choice(available) for _ in range(n_rows)]
    else:
        keys = [f"K{i:04d}" for i in rng.sample(range(1, n_rows * 3), n_rows)]

    # Inject duplicates (lower uniqueness)
    if uniqueness < 1.0:
        n_dupes = int(n_rows * (1 - uniqueness))
        for i in rng.sample(range(n_rows), n_dupes):
            keys[i] = keys[rng.randint(0, i - 1)] if i > 0 else keys[0]

    # Inject nulls
    null_mask = rng.sample(range(n_rows), int(n_rows * null_ratio))
    for idx in null_mask:
        keys[idx] = None  # type: ignore[call-overload]

    data: dict = {key_col: keys, "dim_name": [f"Name_{k}" for k in keys]}
    for col in (tracked_cols or []):
        data[col] = [rng.choice(["x", "y", "z"]) for _ in range(n_rows)]
    for col in (extra_cols or []):
        data[col] = [rng.uniform(0, 100) for _ in range(n_rows)]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """Generates Bronze A/B DataFrames for MEDUSA episodes."""

    STALE_THRESHOLD = _STALE_THRESHOLD_HOURS
    JOIN_KEY = "entity_id"
    TRACKED_COLS = ["dim_status"]

    # Four canonical scenario types
    CANONICAL: List[str] = ["clean", "dirty_keys", "stale", "schema_drift"]

    def __init__(self, n_fact_rows: int = 200, n_dim_rows: int = 150):
        self.n_fact_rows = n_fact_rows
        self.n_dim_rows = n_dim_rows

    def generate(self, seed: Optional[int] = None) -> Scenario:
        """Generate a random scenario. Canonical scenarios cycle through seeds 0-3."""
        rng = random.Random(seed)
        if seed is not None and 0 <= seed < len(self.CANONICAL):
            return self._canonical(self.CANONICAL[seed], seed)
        variant = rng.choice(self.CANONICAL)
        return self._canonical(variant, seed)

    def _canonical(self, variant: str, seed: Optional[int]) -> Scenario:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        key = self.JOIN_KEY
        n_a = self.n_fact_rows
        n_b = self.n_dim_rows

        if variant == "clean":
            # Fresh, unique keys, ~100% match rate
            fact = _make_fact(rng, n_a, key, null_ratio=0.0)
            valid_keys = fact[key].dropna().tolist()
            dim = _make_dim(rng, n_b, key, null_ratio=0.0, uniqueness=1.0,
                            match_keys=valid_keys, tracked_cols=self.TRACKED_COLS)
            return Scenario(
                id=f"clean_{seed}",
                bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=False, is_stale_b=False,
                time_delta_a=1.0, time_delta_b=2.0,
                new_cols_a=[], new_cols_b=[],
                description="Clean scenario: fresh, unique keys, high match rate.",
            )

        elif variant == "dirty_keys":
            # High null ratio in keys, no trimming / type-casting yet
            fact = _make_fact(rng, n_a, key, null_ratio=0.25)
            fact[key] = fact[key].apply(
                lambda k: f"  {k}  " if k and rng.random() < 0.3 else k  # whitespace noise
            )
            dim = _make_dim(rng, n_b, key, null_ratio=0.15, uniqueness=0.85,
                            tracked_cols=self.TRACKED_COLS)
            return Scenario(
                id=f"dirty_keys_{seed}",
                bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=False, is_stale_b=False,
                time_delta_a=2.0, time_delta_b=3.0,
                new_cols_a=[], new_cols_b=[],
                description="Dirty keys: nulls + whitespace in join keys.",
            )

        elif variant == "stale":
            # One or both sources have not refreshed recently
            fact = _make_fact(rng, n_a, key, null_ratio=0.0)
            valid_keys = fact[key].dropna().tolist()
            dim = _make_dim(rng, n_b, key, null_ratio=0.0, uniqueness=1.0,
                            match_keys=valid_keys, tracked_cols=self.TRACKED_COLS)
            td_a = rng.uniform(8.0, 24.0)   # definitely stale
            td_b = rng.uniform(0.5, 4.0)
            return Scenario(
                id=f"stale_{seed}",
                bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=td_a > self.STALE_THRESHOLD,
                is_stale_b=td_b > self.STALE_THRESHOLD,
                time_delta_a=td_a, time_delta_b=td_b,
                new_cols_a=[], new_cols_b=[],
                description=f"Stale scenario: Source A is {td_a:.1f}h old.",
            )

        else:  # schema_drift
            # New columns in A and/or B not yet registered in Silver
            extra_a = ["new_metric_a"]
            extra_b = ["new_attr_b"]
            fact = _make_fact(rng, n_a, key, null_ratio=0.0, extra_cols=extra_a)
            valid_keys = fact[key].dropna().tolist()
            dim = _make_dim(rng, n_b, key, null_ratio=0.0, uniqueness=1.0,
                            match_keys=valid_keys,
                            tracked_cols=self.TRACKED_COLS, extra_cols=extra_b)
            return Scenario(
                id=f"schema_drift_{seed}",
                bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=False, is_stale_b=False,
                time_delta_a=1.0, time_delta_b=1.5,
                new_cols_a=extra_a, new_cols_b=extra_b,
                description="Schema drift: new columns in A and B.",
            )
