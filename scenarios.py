"""MEDUSA scenario generator — v4.0.

Produces:
  1. Legacy single-episode scenarios (4 canonical variants) — backward compat
  2. DayDataGenerator for the 30-day gauntlet: per-day Bronze batches with
     deterministic corruptions and trap-day injections.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scenario dataclass (legacy — single-episode mode)
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

# Column pool for the daily data generator
_COLUMNS = {
    "user_id": "str",
    "customer_name": "str",
    "product_name": "str",
    "category": "str",
    "revenue": "float",
    "discount_amount": "float",
    "quantity": "int",
    "price": "float",
    "order_date": "str",
    "region": "str",
}

# Corruption pool for gap days
_CORRUPTION_POOL: List[Tuple[str, str]] = [
    ("discount_amount", "fill_zero"),
    ("quantity", "fill_zero"),
    ("price", "fill_zero"),
    ("customer_name", "strip"),
    ("product_name", "strip"),
    ("category", "strip"),
    ("quantity", "cast"),
    ("price", "cast"),
    ("revenue", "cast"),
]


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
        if len(available) >= n_rows:
            keys = rng.sample(available, n_rows)
        else:
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
# Legacy Scenario Generator (single-episode mode, backward compat)
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


# ---------------------------------------------------------------------------
# v4.0: 30-Day Data Generator
# ---------------------------------------------------------------------------

@dataclass
class DayBatch:
    """One day's worth of Bronze data for the 30-day gauntlet."""

    day: int
    raw_data: pd.DataFrame          # The daily Bronze batch
    anomalies: List[Tuple[str, str]]  # List of (col, op) corruptions injected
    is_trap_day: bool
    trap_type: Optional[str] = None  # "type_trap" | "oom_trap" | "schema_drift" | "null_nuke"
    new_columns: List[str] = field(default_factory=list)  # Columns added by schema drift
    description: str = ""


class DayDataGenerator:
    """Generates daily Bronze batches for the 30-day gauntlet.

    Each day produces a fresh batch of raw data with deterministic
    corruptions injected. The anomaly checklist is the source of truth
    for both the grader and the reward gate.
    """

    # Major trap days with fixed, unique traps
    TRAP_DAYS: Dict[int, str] = {
        8: "type_trap",
        14: "oom_trap",
        21: "schema_drift",
        28: "null_nuke",
    }

    # Base columns present in every daily batch
    BASE_COLUMNS = ["user_id", "customer_name", "product_name",
                    "category", "revenue", "discount_amount",
                    "quantity", "price", "order_date", "region"]

    def __init__(self, episode_seed: int, n_rows: int = 100):
        self.episode_seed = episode_seed
        self.n_rows = n_rows
        self._day_anomalies: Dict[int, List[Tuple[str, str]]] = {}
        self._build_anomaly_schedule()

    def _build_anomaly_schedule(self) -> None:
        """Pre-compute the anomaly checklist for all 30 days."""
        rng = random.Random(self.episode_seed)

        for day in range(1, 31):
            if day in self.TRAP_DAYS:
                trap = self.TRAP_DAYS[day]
                if trap == "type_trap":
                    self._day_anomalies[day] = [("revenue", "strip"), ("revenue", "cast")]
                elif trap == "oom_trap":
                    self._day_anomalies[day] = [("user_id", "deduplicate")]
                elif trap == "schema_drift":
                    self._day_anomalies[day] = [("promo_code", "evolve")]
                elif trap == "null_nuke":
                    self._day_anomalies[day] = [("user_id", "quarantine")]
            else:
                # Gap day: pick 1-2 corruptions from the pool
                n_corruptions = rng.choice([1, 2])
                day_rng = random.Random(self.episode_seed * 1000 + day)
                pool = list(_CORRUPTION_POOL)
                day_rng.shuffle(pool)
                self._day_anomalies[day] = pool[:n_corruptions]

    @property
    def day_anomalies(self) -> Dict[int, List[Tuple[str, str]]]:
        return dict(self._day_anomalies)

    def generate_day(self, day: int) -> DayBatch:
        """Generate the Bronze batch for a specific day."""
        rng = random.Random(self.episode_seed * 1000 + day)
        n = self.n_rows

        # --- Build base data ---
        data: Dict[str, Any] = {
            "user_id": [f"U{rng.randint(1000, 9999)}" for _ in range(n)],
            "customer_name": [rng.choice(["Alice", "Bob", "Charlie", "Diana", "Eve",
                                          "Frank", "Grace", "Hank"]) for _ in range(n)],
            "product_name": [rng.choice(["Widget", "Gadget", "Doohickey", "Thingamajig",
                                         "Gizmo"]) for _ in range(n)],
            "category": [rng.choice(["Electronics", "Home", "Garden", "Sports",
                                     "Books"]) for _ in range(n)],
            "revenue": [round(rng.uniform(5.0, 500.0), 2) for _ in range(n)],
            "discount_amount": [round(rng.uniform(0.0, 50.0), 2) for _ in range(n)],
            "quantity": [rng.randint(1, 20) for _ in range(n)],
            "price": [round(rng.uniform(1.0, 200.0), 2) for _ in range(n)],
            "order_date": [f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
                           for _ in range(n)],
            "region": [rng.choice(["North", "South", "East", "West"]) for _ in range(n)],
        }

        anomalies = self._day_anomalies.get(day, [])
        is_trap = day in self.TRAP_DAYS
        trap_type = self.TRAP_DAYS.get(day)
        new_columns: List[str] = []
        description = f"Day {day}"

        # --- Inject corruptions ---
        if is_trap:
            if trap_type == "type_trap":
                # Day 8: revenue as "$50.50" strings
                data["revenue"] = [f"${v}" for v in data["revenue"]]
                description = "Day 8 — Type Trap: revenue stored as '$50.50' strings"

            elif trap_type == "oom_trap":
                # Day 14: massive duplicate user_ids
                oom_key = rng.choice(data["user_id"][:5])
                data["user_id"] = [oom_key] * n  # All same key
                description = "Day 14 — OOM Trap: all rows have identical user_id"

            elif trap_type == "schema_drift":
                # Day 21: promo_code column appears
                data["promo_code"] = [f"PROMO{rng.randint(100, 999)}" if rng.random() > 0.3
                                      else None for _ in range(n)]
                new_columns = ["promo_code"]
                description = "Day 21 — Schema Drift: promo_code column appears"

            elif trap_type == "null_nuke":
                # Day 28: 20% of user_id are NULL
                null_count = int(n * 0.20)
                null_indices = rng.sample(range(n), null_count)
                for idx in null_indices:
                    data["user_id"][idx] = None
                description = "Day 28 — Null Nuke: 20% of user_id are NULL"
        else:
            # Apply gap-day corruptions
            for col, op in anomalies:
                if col not in data:
                    continue
                if op == "fill_zero":
                    # Inject 5-15% nulls
                    null_pct = rng.uniform(0.05, 0.15)
                    null_count = int(n * null_pct)
                    null_indices = rng.sample(range(n), null_count)
                    for idx in null_indices:
                        data[col][idx] = None
                elif op == "strip":
                    # Add trailing whitespace to 10-30% of rows
                    ws_pct = rng.uniform(0.10, 0.30)
                    ws_count = int(n * ws_pct)
                    ws_indices = rng.sample(range(n), ws_count)
                    for idx in ws_indices:
                        if data[col][idx] is not None:
                            data[col][idx] = f"  {data[col][idx]}  "
                elif op == "cast":
                    # Convert numeric to string for 100% of rows
                    data[col] = [str(v) if v is not None else None for v in data[col]]

        df = pd.DataFrame(data)

        return DayBatch(
            day=day,
            raw_data=df,
            anomalies=anomalies,
            is_trap_day=is_trap,
            trap_type=trap_type,
            new_columns=new_columns,
            description=description,
        )

    def verify_batch_has_anomaly(self, batch: DayBatch) -> bool:
        """Verify that the raw data in a batch fails at least one assertion.

        Returns True if the batch has detectable corruption.
        """
        df = batch.raw_data

        # Check for nulls in any column
        if df.isnull().any().any():
            return True

        # Check for whitespace in string columns
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].dropna().astype(str).str.contains(r"^\s|\s$").any():
                return True

        # Check for type mismatches (e.g., revenue as string with $)
        if "revenue" in df.columns:
            try:
                pd.to_numeric(df["revenue"], errors="raise")
            except (ValueError, TypeError):
                return True

        # Check for schema drift (new columns)
        if batch.new_columns:
            return True

        # Check for duplicate keys
        if "user_id" in df.columns:
            if df["user_id"].duplicated().sum() > len(df) * 0.5:
                return True

        return False