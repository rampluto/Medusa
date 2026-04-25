"""
generate_olist_days.py
----------------------
Deterministic corruption script for the MEDUSA 30-day Olist gauntlet.

Column names are NEVER hardcoded in the corruption logic.
Corruptions are applied by detecting column *roles* from the data at runtime:

  Role       | Detection rule
  -----------|----------------------------------------------------------
  numeric    | ≥80% of non-null values parse as float
  date       | ≥60% of non-null values match a date-like pattern
  id         | unique-ratio ≥80% among non-null, string dtype
  categorical| unique-ratio ≤20%, string dtype
  string     | everything else that is string dtype

Trap and gap-day corruptions are expressed in terms of roles, not names.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Optional
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "olist_dataset"
OUTPUT_DIR = BASE_DIR / "data" / "olist"

# ---------------------------------------------------------------------------
# Constants / patterns
# ---------------------------------------------------------------------------
DATE_LIKE_PATTERN = r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
NON_NUMERIC_TOKENS = ["N/A", "ERR", "--", "?", "n.a.", "#NUM!"]
NUMERIC_PRESENCE_THRESHOLD = 0.80   # fraction of non-nulls that must parse as float
DATE_PRESENCE_THRESHOLD = 0.60
ID_UNIQUE_THRESHOLD = 0.80          # fraction unique → treat as key/id column
CAT_UNIQUE_THRESHOLD = 0.20         # fraction unique → treat as categorical

# ---------------------------------------------------------------------------
# Column-role detection  (no column-name knowledge required)
# ---------------------------------------------------------------------------

def detect_column_roles(df: pd.DataFrame, primary_key: Optional[str] = None) -> dict[str, list[str]]:
    """
    Return a mapping of role → [column_names].
    Works on any DataFrame regardless of column names.
    Primary key must be explicitly provided.
    """
    roles: dict[str, list[str]] = {
        "numeric": [],
        "date": [],
        "id": [],
        "categorical": [],
        "string": [],
    }
    n = max(len(df), 1)

    if primary_key and primary_key in df.columns:
        roles["id"].append(primary_key)

    for col in df.columns:
        if primary_key and col == primary_key:
            continue
            
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        # --- numeric? ---
        numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
        if numeric_ratio >= NUMERIC_PRESENCE_THRESHOLD:
            roles["numeric"].append(col)
            continue

        # --- date-like? ---
        date_ratio = series.str.match(DATE_LIKE_PATTERN, na=False).mean()
        if date_ratio >= DATE_PRESENCE_THRESHOLD:
            roles["date"].append(col)
            continue

        # --- string: check cardinality ---
        unique_ratio = df[col].nunique(dropna=True) / n
        if unique_ratio <= CAT_UNIQUE_THRESHOLD:
            roles["categorical"].append(col)
        else:
            roles["string"].append(col)

    return roles


def pick_col(roles: dict, *role_priority: str) -> tuple[str | None, str | None]:
    """Pick the first available column matching any role in priority order."""
    for role in role_priority:
        cols = roles.get(role, [])
        if cols:
            return cols[0], role
    return None, None


def pick_other_col(
    roles: dict, exclude: str, *role_priority: str
) -> tuple[str | None, str | None]:
    """Pick first column that is NOT `exclude`, following priority order."""
    for role in role_priority:
        for col in roles.get(role, []):
            if col != exclude:
                return col, role
    return None, None


# ---------------------------------------------------------------------------
# Primitive corruption applicators  (column-name-agnostic)
# ---------------------------------------------------------------------------

def inject_nulls(chunk: pd.DataFrame, col: str, day_rng: random.Random, n: int,
                 pct_range: tuple[float, float] = (0.15, 0.30)) -> None:
    """Set a fraction of rows in `col` to None."""
    pct = day_rng.uniform(*pct_range)
    count = int(n * pct)
    if count == 0:
        return
    indices = day_rng.sample(range(n), count)
    chunk.loc[indices, col] = None


def inject_whitespace(chunk: pd.DataFrame, col: str, day_rng: random.Random, n: int,
                      pct_range: tuple[float, float] = (0.30, 0.50)) -> None:
    """Pad a fraction of string values with leading/trailing spaces."""
    pct = day_rng.uniform(*pct_range)
    count = int(n * pct)
    if count == 0:
        return
    indices = day_rng.sample(range(n), count)
    chunk.loc[indices, col] = chunk.loc[indices, col].apply(
        lambda x: f"  {x}  " if pd.notna(x) else x
    )


def inject_negatives(chunk: pd.DataFrame, col: str, day_rng: random.Random, n: int,
                     pct_range: tuple[float, float] = (0.10, 0.20)) -> None:
    """Replace a fraction of numeric values with their negative (domain-invalid)."""
    pct = day_rng.uniform(*pct_range)
    count = int(n * pct)
    if count == 0:
        return
    indices = day_rng.sample(range(n), count)
    for idx in indices:
        try:
            chunk.at[idx, col] = -abs(float(chunk.at[idx, col]))
        except (ValueError, TypeError):
            chunk.at[idx, col] = -1.0


def inject_mixed_type_tokens(chunk: pd.DataFrame, col: str, day_rng: random.Random,
                              n: int, bad_pct: float = 0.30) -> None:
    """
    Replace `bad_pct` of values with non-numeric tokens so that
    type_consistency parseable_ratio on a numeric column drops to ~(1 - bad_pct).
    Cast whole column to str first so all values become text.
    """
    count = int(n * bad_pct)
    indices = day_rng.sample(range(n), count)
    chunk[col] = chunk[col].astype(str)
    for idx in indices:
        chunk.at[idx, col] = day_rng.choice(NON_NUMERIC_TOKENS)


def clone_key_column(chunk: pd.DataFrame, col: str, day_rng: random.Random,
                     n: int, clone_pct: float = 0.80) -> None:
    """Stamp a single key value onto `clone_pct` of rows — destroys uniqueness."""
    anchor = chunk[col].dropna().iloc[0]
    indices = day_rng.sample(range(n), int(n * clone_pct))
    chunk.loc[indices, col] = anchor


def add_synthetic_column(chunk: pd.DataFrame, day_rng: random.Random,
                          n: int, null_pct: float = 0.40,
                          col_name: str = "extra_feature") -> str:
    """
    Add a new column (schema drift) with `null_pct` nulls.
    Returns the actual column name used (disambiguated if already present).
    """
    base = col_name
    suffix = 0
    while col_name in chunk.columns:
        suffix += 1
        col_name = f"{base}_{suffix}"

    values = []
    for _ in range(n):
        if day_rng.random() < null_pct:
            values.append(None)
        else:
            values.append(f"VAL{day_rng.randint(100, 999)}")
    chunk[col_name] = values
    return col_name


# ---------------------------------------------------------------------------
# Anomaly verifier  (role-based, no hardcoded column names)
# ---------------------------------------------------------------------------

def verify_batch_has_anomaly(
    df: pd.DataFrame, roles: dict, new_columns: list[str]
) -> bool:
    """Return True if any detectable anomaly is present in the raw DataFrame."""

    # Null in any column
    if df.isnull().any().any():
        return True

    # Whitespace in any string-typed column
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].dropna().astype(str).str.contains(r"^\s|\s$", regex=True).any():
            return True

    # Mixed-type: column expected to be numeric but has non-numeric tokens
    for col in roles.get("numeric", []):
        if col in df.columns:
            series = df[col].astype(str)
            numeric_ratio = pd.to_numeric(series, errors="coerce").notna().mean()
            if numeric_ratio < 0.90:   # significant non-numeric leakage
                return True

    # Domain-invalid negatives in any detected numeric role column
    for col in roles.get("numeric", []):
        if col in df.columns:
            num = pd.to_numeric(df[col], errors="coerce")
            if (num < 0).any():
                return True

    # Schema drift
    if new_columns:
        return True

    # Key-column over-duplication (>50% rows share same value)
    for col in roles.get("id", []):
        if col in df.columns and df[col].notna().sum() > 0:
            top_ratio = df[col].value_counts(normalize=True, dropna=True).iloc[0]
            if top_ratio > 0.50:
                return True

    return False


# ---------------------------------------------------------------------------
# Data loading  (Olist-specific — only this section references fixed names)
# ---------------------------------------------------------------------------

def load_raw_dataframe() -> pd.DataFrame:
    """Load and merge Olist CSVs into a single Bronze DataFrame.
    If CSVs are missing, automatically generate synthetic datasets with the right schemas.
    """
    print("Loading Olist datasets...")
    
    items_path = DATASET_DIR / "olist_order_items_dataset.csv"
    orders_path = DATASET_DIR / "olist_orders_dataset.csv"
    customs_path = DATASET_DIR / "olist_customers_dataset.csv"
    
    if items_path.exists() and orders_path.exists() and customs_path.exists():
        df_items = pd.read_csv(items_path)
        df_orders = pd.read_csv(orders_path)
        df_customs = pd.read_csv(customs_path)
    else:
        print("Kaggle CSVs not found in olist_dataset/. Auto-generating a 5,000-row synthetic substitute...")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        # Create synthetic datasets
        rng = np.random.default_rng(42)
        n = 5000
        customer_ids = [f"cust_{i}" for i in range(n)]
        order_ids = [f"order_{i}" for i in range(n)]
        
        df_customs = pd.DataFrame({
            "customer_id": customer_ids,
            "customer_city": rng.choice(["sao paulo", "rio de janeiro", "curitiba"], n),
            "customer_state": rng.choice(["SP", "RJ", "PR"], n),
        })
        df_orders = pd.DataFrame({
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_purchase_timestamp": pd.date_range("2018-01-01", periods=n, freq="min").astype(str),
        })
        df_items = pd.DataFrame({
            "order_id": order_ids,
            "product_id": [f"prod_{rng.integers(1, 100)}" for _ in range(n)],
            "seller_id": [f"seller_{rng.integers(1, 20)}" for _ in range(n)],
            "price": rng.uniform(10.0, 500.0, n).round(2),
            "freight_value": rng.uniform(5.0, 50.0, n).round(2),
        })

    print("Merging DataFrames...")
    df = df_items.merge(df_orders, on="order_id", how="inner")
    df = df.merge(df_customs, on="customer_id", how="inner")

    keep_cols = [
        "customer_id", "order_id", "product_id", "seller_id",
        "price", "freight_value", "customer_city", "customer_state",
        "order_purchase_timestamp",
    ]
    df = df[keep_cols].rename(columns={"order_purchase_timestamp": "order_date"})
    return df.sort_values("order_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRAP_DAYS: dict[int, str] = {
    8: "type_trap",
    14: "oom_trap",
    21: "schema_drift",
    28: "null_nuke",
}

# Gap-day corruption menu: each entry is (role_preference, op_tag, applicator)
# The actual column is chosen at runtime via detect_column_roles().
_GAP_DAY_POOL = [
    ("numeric",               "fill_null",  inject_nulls),
    ("string", "categorical", "whitespace", inject_whitespace),  # fallback to categorical
    ("numeric",               "negative",   inject_negatives),
]


def _resolve_gap_entry(entry: tuple) -> tuple:
    """Unpack variable-length pool entry into (roles_tuple, op_tag, fn)."""
    *roles, op_tag, fn = entry
    return tuple(roles), op_tag, fn


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataframe()

    print("Chunking data into 30 days...")
    indices = np.array_split(range(len(df)), 30)
    chunks = [df.iloc[idx].copy().reset_index(drop=True) for idx in indices]

    anomalies_map: dict[int, list] = {}
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    for i in range(30):
        day = i + 1
        chunk = chunks[i].copy()
        n = len(chunk)

        # Synthesise a numeric column (discount) from existing numeric data
        # We pick the first numeric column dynamically and derive a discount from it.
        pre_roles = detect_column_roles(chunk, primary_key="order_id")
        base_numeric_col, _ = pick_col(pre_roles, "numeric")
        if base_numeric_col is not None:
            chunk["discount_amount"] = (
                np_rng.uniform(0, 0.2, n) * pd.to_numeric(chunk[base_numeric_col], errors="coerce").fillna(0)
            ).round(2)

        # Re-detect roles AFTER adding discount_amount
        roles = detect_column_roles(chunk, primary_key="order_id")

        anomalies: list = []
        is_trap = day in TRAP_DAYS
        trap_type = TRAP_DAYS.get(day)
        new_columns: list[str] = []
        day_rng = random.Random(42 * 1000 + day)

        if is_trap:

            if trap_type == "type_trap":
                # Pick the first numeric column; inject 30% non-numeric tokens
                # → type_consistency parseable_ratio ≈ 0.70
                col, _ = pick_col(roles, "numeric")
                if col:
                    inject_mixed_type_tokens(chunk, col, day_rng, n, bad_pct=0.30)
                    anomalies = [(col, "type_mixed")]

            elif trap_type == "oom_trap":
                # Clone an id column to 80% of rows → breaks row-level deduplication attempts
                # + inject 30% nulls in a second column → hits completeness
                id_col, _ = pick_col(roles, "id")
                other_col, _ = pick_other_col(roles, id_col or "", "id", "numeric", "string")
                if id_col:
                    clone_key_column(chunk, id_col, day_rng, n, clone_pct=0.80)
                    anomalies.append((id_col, "deduplicate"))
                if other_col:
                    inject_nulls(chunk, other_col, day_rng, n, pct_range=(0.30, 0.30))
                    anomalies.append((other_col, "quarantine"))

            elif trap_type == "schema_drift":
                # Add a new column with 40% nulls → hits completeness + schema compat
                new_col = add_synthetic_column(chunk, day_rng, n, null_pct=0.40)
                new_columns = [new_col]
                anomalies = [(new_col, "evolve")]

            elif trap_type == "null_nuke":
                # Inject 50% nulls into an id column → completeness tanks
                id_col, _ = pick_col(roles, "id")
                if id_col:
                    inject_nulls(chunk, id_col, day_rng, n, pct_range=(0.50, 0.50))
                    anomalies = [(id_col, "quarantine")]

        else:
            # Gap day: pick 1–2 corruptions from the pool
            pool = list(_GAP_DAY_POOL)
            day_rng.shuffle(pool)
            n_corruptions = day_rng.choice([1, 2])
            selected = pool[:n_corruptions]

            for entry in selected:
                role_priority, op_tag, fn = _resolve_gap_entry(entry)
                col, _ = pick_col(roles, *role_priority)
                if col:
                    fn(chunk, col, day_rng, n)
                    anomalies.append((col, op_tag))

        # Re-detect after mutations so verifier uses updated schema
        final_roles = detect_column_roles(chunk)
        anomalies_map[day] = anomalies

        if not verify_batch_has_anomaly(chunk, final_roles, new_columns):
            print(f"WARNING: Day {day} anomaly verification failed — forcing null.")
            fallback_col, _ = pick_col(final_roles, "numeric", "id", "string")
            if fallback_col:
                chunk.loc[0, fallback_col] = None

        out_path = OUTPUT_DIR / f"day_{day:02d}.csv"
        chunk.to_csv(out_path, index=False)
        print(f"  Day {day:02d} written → {out_path.name}")

    with open(OUTPUT_DIR / "anomalies_map.json", "w") as f:
        json.dump(anomalies_map, f, indent=2)

    print(f"\nGenerated 30 days of data at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
