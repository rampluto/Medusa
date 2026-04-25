"""MEDUSA scenario generator — v4.0.

Produces:
  1. Legacy single-episode scenarios (4 canonical variants) — backward compat
  2. DayDataGenerator for the 30-day gauntlet: per-day Bronze batches with
     deterministic corruptions and trap-day injections.

Column names are NEVER hardcoded in the corruption or verification logic.
All corruptions are expressed in terms of column *roles* (numeric, id, string,
categorical, date) detected at runtime from the data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column-role detection  (shared by both generators)
# ---------------------------------------------------------------------------

_DATE_LIKE_PATTERN = r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
_NUMERIC_THRESHOLD = 0.80
_DATE_THRESHOLD = 0.60
_ID_UNIQUE_THRESHOLD = 0.80
_CAT_UNIQUE_THRESHOLD = 0.20
_NON_NUMERIC_TOKENS = ["N/A", "ERR", "--", "?", "n.a.", "#NUM!"]


def detect_column_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into roles based purely on data statistics.

    Returns:
        dict with keys: 'numeric', 'date', 'id', 'categorical', 'string'
        Each value is a list of column names assigned that role.
    """
    roles: dict[str, list[str]] = {
        "numeric": [], "date": [], "id": [], "categorical": [], "string": [],
    }
    n = max(len(df), 1)
    for col in df.columns:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        # Numeric?
        if pd.to_numeric(series, errors="coerce").notna().mean() >= _NUMERIC_THRESHOLD:
            roles["numeric"].append(col)
            continue
        # Date-like?
        if series.str.match(_DATE_LIKE_PATTERN, na=False).mean() >= _DATE_THRESHOLD:
            roles["date"].append(col)
            continue
        # String: check cardinality
        unique_ratio = df[col].nunique(dropna=True) / n
        if unique_ratio >= _ID_UNIQUE_THRESHOLD:
            roles["id"].append(col)
        elif unique_ratio <= _CAT_UNIQUE_THRESHOLD:
            roles["categorical"].append(col)
        else:
            roles["string"].append(col)
    return roles


def _pick(roles: dict, *role_priority: str) -> tuple[str | None, str | None]:
    for role in role_priority:
        cols = roles.get(role, [])
        if cols:
            return cols[0], role
    return None, None


def _pick_other(roles: dict, exclude: str, *role_priority: str) -> tuple[str | None, str | None]:
    for role in role_priority:
        for col in roles.get(role, []):
            if col != exclude:
                return col, role
    return None, None


def _verify_has_anomaly(df: pd.DataFrame, roles: dict, new_columns: list[str]) -> bool:
    """Role-based anomaly check — no hardcoded column names."""
    if df.isnull().any().any():
        return True
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].dropna().astype(str).str.contains(r"^\s|\s$", regex=True).any():
            return True
    for col in roles.get("numeric", []):
        if col in df.columns:
            series = df[col].astype(str)
            if pd.to_numeric(series, errors="coerce").notna().mean() < 0.90:
                return True
            num = pd.to_numeric(df[col], errors="coerce")
            if (num < 0).any():
                return True
    if new_columns:
        return True
    for col in roles.get("id", []):
        if col in df.columns and df[col].notna().sum() > 0:
            top_ratio = df[col].value_counts(normalize=True, dropna=True).iloc[0]
            if top_ratio > 0.50:
                return True
    return False


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
# Internal helpers (legacy)
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
    uniqueness: float = 1.0,
    match_keys: Optional[List[str]] = None,
    extra_cols: Optional[List[str]] = None,
    tracked_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create a synthetic Dimension (Bronze B) DataFrame."""
    if match_keys:
        available = list(match_keys)
        if len(available) >= n_rows:
            keys = rng.sample(available, n_rows)
        else:
            keys = [rng.choice(available) for _ in range(n_rows)]
    else:
        keys = [f"K{i:04d}" for i in rng.sample(range(1, n_rows * 3), n_rows)]

    if uniqueness < 1.0:
        n_dupes = int(n_rows * (1 - uniqueness))
        for i in rng.sample(range(n_rows), n_dupes):
            keys[i] = keys[rng.randint(0, i - 1)] if i > 0 else keys[0]

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
    CANONICAL: List[str] = ["clean", "dirty_keys", "stale", "schema_drift"]

    def __init__(self, n_fact_rows: int = 200, n_dim_rows: int = 150):
        self.n_fact_rows = n_fact_rows
        self.n_dim_rows = n_dim_rows

    def generate(self, seed: Optional[int] = None) -> Scenario:
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
            fact = _make_fact(rng, n_a, key, null_ratio=0.0)
            valid_keys = fact[key].dropna().tolist()
            dim = _make_dim(rng, n_b, key, null_ratio=0.0, uniqueness=1.0,
                            match_keys=valid_keys, tracked_cols=self.TRACKED_COLS)
            return Scenario(
                id=f"clean_{seed}", bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=False, is_stale_b=False,
                time_delta_a=1.0, time_delta_b=2.0,
                new_cols_a=[], new_cols_b=[],
                description="Clean scenario: fresh, unique keys, high match rate.",
            )

        elif variant == "dirty_keys":
            fact = _make_fact(rng, n_a, key, null_ratio=0.25)
            fact[key] = fact[key].apply(
                lambda k: f"  {k}  " if k and rng.random() < 0.3 else k
            )
            dim = _make_dim(rng, n_b, key, null_ratio=0.15, uniqueness=0.85,
                            tracked_cols=self.TRACKED_COLS)
            return Scenario(
                id=f"dirty_keys_{seed}", bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=False, is_stale_b=False,
                time_delta_a=2.0, time_delta_b=3.0,
                new_cols_a=[], new_cols_b=[],
                description="Dirty keys: nulls + whitespace in join keys.",
            )

        elif variant == "stale":
            fact = _make_fact(rng, n_a, key, null_ratio=0.0)
            valid_keys = fact[key].dropna().tolist()
            dim = _make_dim(rng, n_b, key, null_ratio=0.0, uniqueness=1.0,
                            match_keys=valid_keys, tracked_cols=self.TRACKED_COLS)
            td_a = rng.uniform(8.0, 24.0)
            td_b = rng.uniform(0.5, 4.0)
            return Scenario(
                id=f"stale_{seed}", bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=td_a > self.STALE_THRESHOLD,
                is_stale_b=td_b > self.STALE_THRESHOLD,
                time_delta_a=td_a, time_delta_b=td_b,
                new_cols_a=[], new_cols_b=[],
                description=f"Stale scenario: Source A is {td_a:.1f}h old.",
            )

        else:  # schema_drift
            extra_a = ["new_metric_a"]
            extra_b = ["new_attr_b"]
            fact = _make_fact(rng, n_a, key, null_ratio=0.0, extra_cols=extra_a)
            valid_keys = fact[key].dropna().tolist()
            dim = _make_dim(rng, n_b, key, null_ratio=0.0, uniqueness=1.0,
                            match_keys=valid_keys,
                            tracked_cols=self.TRACKED_COLS, extra_cols=extra_b)
            return Scenario(
                id=f"schema_drift_{seed}", bronze_a=fact, bronze_b=dim,
                join_key=key, tracked_cols=self.TRACKED_COLS,
                is_stale_a=False, is_stale_b=False,
                time_delta_a=1.0, time_delta_b=1.5,
                new_cols_a=extra_a, new_cols_b=extra_b,
                description="Schema drift: new columns in A and B.",
            )


# ---------------------------------------------------------------------------
# v4.0: DayBatch dataclass
# ---------------------------------------------------------------------------

@dataclass
class DayBatch:
    """One day's worth of Bronze data for the 30-day gauntlet."""

    day: int
    raw_data: pd.DataFrame
    anomalies: List[Tuple[str, str]]
    is_trap_day: bool
    trap_type: Optional[str] = None
    new_columns: List[str] = field(default_factory=list)
    description: str = ""

    # Role metadata (populated after generation so callers can read them)
    pk_col: str = ""
    numeric_cols: List[str] = field(default_factory=list)
    string_cols: List[str] = field(default_factory=list)
    baseline_schema: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# v4.0: DayDataGenerator  (purely synthetic — zero hardcoded column names)
# ---------------------------------------------------------------------------

class DayDataGenerator:
    """Generates daily Bronze batches for the 30-day gauntlet.

    Builds a synthetic dataset whose schema is described by COLUMN_SPEC.
    All corruption logic is role-based — the code never references a column
    by name in the corruption or verification paths.
    """

    TRAP_DAYS: Dict[int, str] = {
        8: "type_trap",
        14: "oom_trap",
        21: "schema_drift",
        28: "null_nuke",
    }

    # Schema spec: col_name → role_hint
    # These are the only place column names appear; they define the dataset.
    COLUMN_SPEC: Dict[str, str] = {
        "row_id":       "id",
        "entity_name":  "categorical",
        "item_name":    "categorical",
        "segment":      "categorical",
        "amount":       "numeric",
        "discount":     "numeric",
        "quantity":     "numeric",
        "unit_price":   "numeric",
        "event_date":   "date",
        "region":       "categorical",
    }

    def __init__(self, episode_seed: int, n_rows: int = 100):
        self.episode_seed = episode_seed
        self.n_rows = n_rows
        self._day_anomalies: Dict[int, List[Tuple[str, str]]] = {}

        # Build a sample day-1 frame to detect roles
        self._sample_df = self._build_base_data(seed=episode_seed, n=n_rows)
        self._roles = detect_column_roles(self._sample_df)
        self._pk_col, _ = _pick(self._roles, "id")
        self._numeric_cols = list(self._roles.get("numeric", []))
        self._string_cols = list(self._roles.get("string", []) + self._roles.get("categorical", []))
        self._baseline_schema = list(self._sample_df.columns)

        self._build_anomaly_schedule()

    # ── Internal data builder ───────────────────────────────────────────────

    def _build_base_data(self, seed: int, n: int) -> pd.DataFrame:
        """Build a base data frame from COLUMN_SPEC without any corruption."""
        rng = random.Random(seed)
        data: Dict[str, Any] = {}
        for col, role in self.COLUMN_SPEC.items():
            if role == "id":
                data[col] = [f"ID{rng.randint(10000, 99999)}" for _ in range(n)]
            elif role == "numeric":
                data[col] = [round(rng.uniform(1.0, 500.0), 2) for _ in range(n)]
            elif role == "categorical":
                cats = [f"{col[:3].upper()}_{c}" for c in ["A", "B", "C", "D", "E"]]
                data[col] = [rng.choice(cats) for _ in range(n)]
            elif role == "date":
                data[col] = [
                    f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
                    for _ in range(n)
                ]
            else:
                data[col] = [f"val_{rng.randint(0, 999)}" for _ in range(n)]
        return pd.DataFrame(data)

    # ── Anomaly schedule ────────────────────────────────────────────────────

    def _build_anomaly_schedule(self) -> None:
        """Pre-compute the anomaly (col, op) checklist for all 30 days using roles."""
        rng = random.Random(self.episode_seed)
        pk = self._pk_col
        num_cols = self._numeric_cols
        str_cols = self._string_cols

        gap_pool: List[Tuple[str, str]] = []
        for col in num_cols:
            gap_pool.append((col, "fill_null"))
        for col in str_cols:
            gap_pool.append((col, "strip"))
        for col in num_cols:
            gap_pool.append((col, "cast"))

        for day in range(1, 31):
            if day in self.TRAP_DAYS:
                trap = self.TRAP_DAYS[day]
                if trap == "type_trap":
                    col = num_cols[0] if num_cols else None
                    self._day_anomalies[day] = [(col, "type_mixed")] if col else []
                elif trap == "oom_trap":
                    self._day_anomalies[day] = [(pk, "deduplicate")] if pk else []
                elif trap == "schema_drift":
                    self._day_anomalies[day] = [("extra_feature", "evolve")]
                elif trap == "null_nuke":
                    self._day_anomalies[day] = [(pk, "quarantine")] if pk else []
            else:
                day_rng = random.Random(self.episode_seed * 1000 + day)
                pool = list(gap_pool)
                day_rng.shuffle(pool)
                n_corruptions = day_rng.choice([1, 2])
                self._day_anomalies[day] = pool[:n_corruptions]

    @property
    def day_anomalies(self) -> Dict[int, List[Tuple[str, str]]]:
        return dict(self._day_anomalies)

    # ── Day generation ──────────────────────────────────────────────────────

    def generate_day(self, day: int) -> DayBatch:
        rng = random.Random(self.episode_seed * 1000 + day)
        n = self.n_rows
        data = self._build_base_data(seed=self.episode_seed * 1000 + day, n=n)

        anomalies = self._day_anomalies.get(day, [])
        is_trap = day in self.TRAP_DAYS
        trap_type = self.TRAP_DAYS.get(day)
        new_columns: List[str] = []
        description = f"Day {day}"

        if is_trap:
            if trap_type == "type_trap":
                # Mix non-numeric tokens into first numeric column → parseable_ratio ≈ 0.70
                num_col = self._numeric_cols[0] if self._numeric_cols else None
                if num_col:
                    bad_count = int(n * 0.30)
                    bad_indices = rng.sample(range(n), bad_count)
                    data[num_col] = data[num_col].astype(str)
                    for idx in bad_indices:
                        data.at[idx, num_col] = rng.choice(_NON_NUMERIC_TOKENS)
                description = f"Day {day} — Type Trap: numeric column has mixed-type tokens"

            elif trap_type == "oom_trap":
                # Clone pk to 80% rows + 30% nulls in second id/numeric col
                pk = self._pk_col
                if pk:
                    anchor = data[pk].iloc[0]
                    clone_indices = rng.sample(range(n), int(n * 0.80))
                    data.loc[clone_indices, pk] = anchor
                other_col, _ = _pick_other(self._roles, pk or "", "id", "numeric")
                if other_col:
                    null_indices = rng.sample(range(n), int(n * 0.30))
                    data.loc[null_indices, other_col] = None
                description = f"Day {day} — OOM Trap: key column massively duplicated"

            elif trap_type == "schema_drift":
                # Add a synthetic column with 40% nulls
                new_col = "extra_feature"
                suffix = 0
                while new_col in data.columns:
                    suffix += 1
                    new_col = f"extra_feature_{suffix}"
                data[new_col] = [
                    f"VAL{rng.randint(100, 999)}" if rng.random() > 0.40 else None
                    for _ in range(n)
                ]
                new_columns = [new_col]
                description = f"Day {day} — Schema Drift: new column '{new_col}' appears"

            elif trap_type == "null_nuke":
                # 50% nulls in pk column
                pk = self._pk_col
                if pk:
                    null_indices = rng.sample(range(n), int(n * 0.50))
                    data.loc[null_indices, pk] = None
                description = f"Day {day} — Null Nuke: 50% of pk column is NULL"

        else:
            for col, op in anomalies:
                if col not in data.columns:
                    continue
                if op == "fill_null":
                    null_pct = rng.uniform(0.15, 0.30)
                    null_indices = rng.sample(range(n), int(n * null_pct))
                    data.loc[null_indices, col] = None
                elif op == "strip":
                    ws_pct = rng.uniform(0.30, 0.50)
                    ws_indices = rng.sample(range(n), int(n * ws_pct))
                    data.loc[ws_indices, col] = data.loc[ws_indices, col].apply(
                        lambda x: f"  {x}  " if pd.notna(x) else x
                    )
                elif op == "cast":
                    data[col] = data[col].astype(str)
                elif op == "negative":
                    neg_pct = rng.uniform(0.10, 0.20)
                    neg_indices = rng.sample(range(n), int(n * neg_pct))
                    for idx in neg_indices:
                        try:
                            data.at[idx, col] = -abs(float(data.at[idx, col]))
                        except (ValueError, TypeError):
                            data.at[idx, col] = -1.0

        # Post-mutation role detection
        final_roles = detect_column_roles(data)
        if not _verify_has_anomaly(data, final_roles, new_columns):
            fallback, _ = _pick(final_roles, "numeric", "id", "categorical", "string")
            if fallback:
                data.loc[0, fallback] = None

        return DayBatch(
            day=day, raw_data=data, anomalies=anomalies,
            is_trap_day=is_trap, trap_type=trap_type,
            new_columns=new_columns, description=description,
            pk_col=self._pk_col or "",
            numeric_cols=list(self._numeric_cols),
            string_cols=list(self._string_cols),
            baseline_schema=list(self._baseline_schema),
        )

    def verify_batch_has_anomaly(self, batch: DayBatch) -> bool:
        roles = detect_column_roles(batch.raw_data)
        return _verify_has_anomaly(batch.raw_data, roles, batch.new_columns)


# ---------------------------------------------------------------------------
# v4.0: OlistDayGenerator  (Olist CSV-backed — role-based, no hardcoded cols)
# ---------------------------------------------------------------------------

class OlistDayGenerator:
    """Generates daily Bronze batches for the 30-day gauntlet using Olist dataset.

    Column roles are detected automatically from the day-1 CSV so that no
    column names are hardcoded in the corruption or verification logic.
    """

    TRAP_DAYS: Dict[int, str] = {
        8: "type_trap",
        14: "oom_trap",
        21: "schema_drift",
        28: "null_nuke",
    }

    def __init__(
        self,
        episode_seed: Optional[int] = None,
        n_rows: int = 100,
        data_dir: str = "data/olist",
        anomalies_file: str = "anomalies_map.json",
    ):
        import json
        from pathlib import Path
        base = Path(__file__).parent
        self.data_dir = base / data_dir
        self.episode_seed = episode_seed
        self.n_rows = n_rows

        with open(self.data_dir / anomalies_file) as f:
            raw_map = json.load(f)
            self._day_anomalies: Dict[int, List[Tuple[str, str]]] = {
                int(k): [(col, op) for col, op in v]
                for k, v in raw_map.items()
            }

        # Detect column roles from day 1 to avoid hardcoding
        day1_df = pd.read_csv(self.data_dir / "day_01.csv")
        self._roles = detect_column_roles(day1_df)
        self._pk_col, _ = _pick(self._roles, "id")
        self._numeric_cols = list(self._roles.get("numeric", []))
        self._string_cols = list(
            self._roles.get("string", []) + self._roles.get("categorical", [])
        )
        self._baseline_schema = list(day1_df.columns)

    # Expose detected metadata
    @property
    def pk_col(self) -> str:
        return self._pk_col or ""

    @property
    def numeric_cols(self) -> List[str]:
        return list(self._numeric_cols)

    @property
    def baseline_schema(self) -> List[str]:
        return list(self._baseline_schema)

    @property
    def day_anomalies(self) -> Dict[int, List[Tuple[str, str]]]:
        return dict(self._day_anomalies)

    def generate_day(self, day: int) -> DayBatch:
        csv_path = self.data_dir / f"day_{day:02d}.csv"
        df = pd.read_csv(csv_path)

        anomalies = self._day_anomalies.get(day, [])
        is_trap = day in self.TRAP_DAYS
        trap_type = self.TRAP_DAYS.get(day)

        # Detect new columns vs baseline schema (role-based, not name-based)
        new_columns = [c for c in df.columns if c not in self._baseline_schema]

        description = f"Day {day} (Olist)"
        if is_trap:
            description += f" — TRAP: {trap_type}"

        # Detect current roles (baseline roles + any new cols)
        current_roles = detect_column_roles(df)

        return DayBatch(
            day=day, raw_data=df, anomalies=anomalies,
            is_trap_day=is_trap, trap_type=trap_type,
            new_columns=new_columns, description=description,
            pk_col=self._pk_col or "",
            numeric_cols=list(self._numeric_cols),
            string_cols=list(self._string_cols),
            baseline_schema=list(self._baseline_schema),
        )

    def verify_batch_has_anomaly(self, batch: DayBatch) -> bool:
        roles = detect_column_roles(batch.raw_data)
        return _verify_has_anomaly(batch.raw_data, roles, batch.new_columns)