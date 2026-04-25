import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "olist_dataset"
OUTPUT_DIR = BASE_DIR / "data" / "olist"

def verify_batch_has_anomaly(df, anomalies, is_trap, trap_type, new_columns):
    """Verify that the raw data in a batch fails at least one assertion."""
    
    # Check for nulls in any column
    if df.isnull().any().any():
        return True

    # Check for whitespace in string columns
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].dropna().astype(str).str.contains(r"^\s|\s$").any():
            return True

    # Check for type mismatches (e.g., price as string with $ or non-numeric text)
    if "price" in df.columns:
        try:
            pd.to_numeric(df["price"], errors="raise")
        except (ValueError, TypeError):
            return True

    # Check for schema drift (new columns)
    if new_columns:
        return True

    # Check for duplicate keys
    if "customer_id" in df.columns:
        if df["customer_id"].duplicated().sum() > len(df) * 0.5:
            return True

    # Check for negative price values (domain-invalid)
    if "price" in df.columns:
        try:
            numeric_price = pd.to_numeric(df["price"], errors="coerce")
            if (numeric_price < 0).any():
                return True
        except Exception:
            pass

    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading Olist datasets...")
    # Load DFs
    df_items = pd.read_csv(DATASET_DIR / "olist_order_items_dataset.csv")
    df_orders = pd.read_csv(DATASET_DIR / "olist_orders_dataset.csv")
    df_customers = pd.read_csv(DATASET_DIR / "olist_customers_dataset.csv")

    print("Merging DataFrames...")
    # Phase A: Merge
    df = df_items.merge(df_orders, on="order_id", how="inner")
    df = df.merge(df_customers, on="customer_id", how="inner")
    
    # Filter columns to create Bronze base
    keep_cols = [
        "customer_id", "order_id", "product_id", "seller_id", 
        "price", "freight_value", "customer_city", "customer_state", 
        "order_purchase_timestamp"
    ]
    df = df[keep_cols]
    
    # Rename timestamp to order_date
    df = df.rename(columns={"order_purchase_timestamp": "order_date"})
    
    # Sort
    print("Sorting by order_date...")
    df = df.sort_values(by="order_date").reset_index(drop=True)
    
    # Create chunks
    print("Chunking data into 30 days...")
    indices = np.array_split(range(len(df)), 30)
    chunks = [df.iloc[idx].copy().reset_index(drop=True) for idx in indices]

    # Trap Days 
    TRAP_DAYS = {
        8: "type_trap",
        14: "oom_trap",
        21: "schema_drift",
        28: "null_nuke",
    }
    
    # --- Gap-day corruption pool ---
    # Each tuple: (col, op, severity_description)
    # Pool now has 3 entries so all three corruptions can be selected
    _CORRUPTION_POOL = [
        ("discount_amount", "fill_null"),   # hits completeness / numeric_sanity
        ("customer_city", "strip"),          # hits string_cleanliness
        ("price", "negative"),               # hits numeric_sanity (domain-invalid floats)
    ]
    
    anomalies_map = {}
    rng = random.Random(42)  # Seed for deterministic generation
    np_rng = np.random.default_rng(42)

    for i in range(30):
        day = i + 1
        chunk = chunks[i]
        n = len(chunk)
        
        # Phase A: Synthesize discount_amount (as numeric)
        chunk["discount_amount"] = np_rng.uniform(0, 0.2, n) * chunk["price"]
        chunk["discount_amount"] = chunk["discount_amount"].round(2)
        
        anomalies = []
        is_trap = day in TRAP_DAYS
        trap_type = TRAP_DAYS.get(day)
        new_columns = []
        
        if is_trap:
            if trap_type == "type_trap":
                # --- Day 8: partial non-numeric strings so type_consistency drops to ~0.70 ---
                # Strategy: mix "$123.45"-style with non-numeric tokens "N/A" and "ERR"
                # so that the parseable_ratio on price falls to roughly 0.70.
                # We inject ~30% corrupted values.
                day_rng = random.Random(42 * 1000 + day)
                corrupt_pct = 0.30  # 30% non-numeric → parseable_ratio ≈ 0.70
                corrupt_count = int(n * corrupt_pct)
                corrupt_indices = day_rng.sample(range(n), corrupt_count)
                bad_tokens = ["N/A", "ERR", "--", "?", "n.a.", "#NUM!"]
                # Keep remaining rows as proper dollar strings (no $ - just numeric text)
                # so parseable_ratio for the other 70% stays numeric
                chunk["price"] = chunk["price"].astype(str)  # ensure string dtype
                for idx in corrupt_indices:
                    chunk.at[idx, "price"] = day_rng.choice(bad_tokens)
                anomalies = [("price", "type_mixed")]

            elif trap_type == "oom_trap":
                # --- Day 14: clone customer_id to 80% of rows + 30% nulls in order_id ---
                # Defeats uniqueness AND completeness simultaneously.
                day_rng = random.Random(42 * 1000 + day)
                oom_key = chunk["customer_id"].dropna().iloc[0]
                # Stamp oom_key on 80% of rows
                oom_indices = day_rng.sample(range(n), int(n * 0.80))
                chunk.loc[oom_indices, "customer_id"] = oom_key
                # Also inject 30% nulls in order_id
                null_order_count = int(n * 0.30)
                null_order_indices = day_rng.sample(range(n), null_order_count)
                chunk.loc[null_order_indices, "order_id"] = None
                anomalies = [("customer_id", "deduplicate"), ("order_id", "quarantine")]

            elif trap_type == "schema_drift":
                # --- Day 21: new promo_code column with 40% nulls ---
                # Hits both schema completeness and column_quality (low fill rate).
                day_rng = random.Random(42 * 1000 + day)
                promo_values = []
                for _ in range(n):
                    if day_rng.random() < 0.40:   # 40% null
                        promo_values.append(None)
                    else:
                        promo_values.append(f"PROMO{day_rng.randint(100, 999)}")
                chunk["promo_code"] = promo_values
                new_columns = ["promo_code"]
                anomalies = [("promo_code", "evolve")]

            elif trap_type == "null_nuke":
                # --- Day 28: 50% customer_id nulls (raised from 20%) ---
                # At 50%, completeness contribution drops ~0.90 → enough to fail 0.80 threshold
                # combined with the base gap-day noise.
                day_rng = random.Random(42 * 1000 + day)
                null_count = int(n * 0.50)
                null_indices = day_rng.sample(range(n), null_count)
                chunk.loc[null_indices, "customer_id"] = None
                anomalies = [("customer_id", "quarantine")]
                
        else:
            # --- Random Gap Day mutations (raised severity) ---
            # Select 1–2 corruptions from pool (now 3-entry pool)
            day_rng = random.Random(42 * 1000 + day)
            pool = list(_CORRUPTION_POOL)
            day_rng.shuffle(pool)
            n_corruptions = day_rng.choice([1, 2])
            anomalies = pool[:n_corruptions]
            
            for col, op in anomalies:
                if op == "fill_null" and col == "discount_amount":
                    # Raised: 15–30% null ratio (was 2–5%)
                    null_pct = day_rng.uniform(0.15, 0.30)
                    null_count = int(n * null_pct)
                    null_indices = day_rng.sample(range(n), null_count)
                    chunk.loc[null_indices, col] = None

                elif op == "strip" and col == "customer_city":
                    # Raised: 30–50% whitespace ratio (was 5–10%)
                    ws_pct = day_rng.uniform(0.30, 0.50)
                    ws_count = int(n * ws_pct)
                    ws_indices = day_rng.sample(range(n), ws_count)
                    chunk.loc[ws_indices, col] = chunk.loc[ws_indices, col].apply(
                        lambda x: f"  {x}  " if pd.notna(x) else x
                    )

                elif op == "negative" and col == "price":
                    # New: inject 10–20% of price values as negative floats
                    # Valid floats but domain-invalid → hits numeric_sanity
                    neg_pct = day_rng.uniform(0.10, 0.20)
                    neg_count = int(n * neg_pct)
                    neg_indices = day_rng.sample(range(n), neg_count)
                    for idx in neg_indices:
                        original = chunk.at[idx, "price"]
                        try:
                            chunk.at[idx, "price"] = -abs(float(original))
                        except (ValueError, TypeError):
                            chunk.at[idx, "price"] = -1.0
        
        anomalies_map[day] = anomalies
        
        # Verify
        if not verify_batch_has_anomaly(chunk, anomalies, is_trap, trap_type, new_columns):
            print(f"WARNING: Day {day} raw data failed anomaly verification! Forcing null...")
            if not is_trap:
                # Force a null discount amount to make sure it trips grader
                chunk.loc[0, "discount_amount"] = None
        
        # Save output
        out_path = OUTPUT_DIR / f"day_{day:02d}.csv"
        chunk.to_csv(out_path, index=False)
        
    # Write anomalies map
    with open(OUTPUT_DIR / "anomalies_map.json", "w") as f:
        json.dump(anomalies_map, f, indent=2)
        
    print(f"Generated 30 days of data at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
