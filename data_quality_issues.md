# `data_quality_score.py` — Structural Issues & Blind Spots

These apply to **any dataset**, not just Olist.

---

## 1. `type_consistency` is blind to "valid-but-wrong-domain" text

**Where:** `infer_column_type()` → `FLOAT_PATTERN` match

When a numeric column is entirely formatted as a non-numeric string (e.g. all values are `"$X.XX"`, `"USD 45"`, or any other domain-specific prefix), the result is:

- `numeric_ratio ≈ 0.0`
- Falls through to `inferred_type = "text"`, `parseable_ratio = 1.0 - 0.0 = 1.0`
- DQS scores the corrupted column as perfectly **clean text** → full marks

**Fix at corruption side:** Mix valid numeric values with non-parseable tokens (`"N/A"`, `"ERR"`) so that `numeric_ratio` settles around 0.70. DQS then infers `numeric` with `parseable_ratio = 0.70` and penalises `type_consistency`.

**Underlying DQS design gap:** No "expected column type" contract is enforced. DQS infers type from data, so a fully-corrupted type column looks like a legitimate text column.

---

## 2. `numeric_sanity` has a hard 90% numeric-presence gate

**Where:** `numeric_sanity_score()` line:

```python
if len(numeric_values) / max(len(values), 1) < 0.90:
    continue
```

**Effect:** Any numeric column with more than 10% non-numeric values is **silently skipped** by the outlier check. A column that is 85% numeric after corruption escapes `numeric_sanity` entirely.

**Secondary issue:** The outlier detector uses IQR (`±1.5 × IQR`). Domain-invalid values (e.g. negative prices, negative counts) are only flagged if they statistically fall outside this range — which depends on the data distribution, not domain rules. Tight distributions flag more; wide distributions flag fewer.

**Impact on corruption design:** Negative-value injections may or may not trigger `numeric_sanity`. Always verify empirically per dataset.

---

## 3. `uniqueness_score` is whole-row de-duplication only

**Where:** `uniqueness_score()`:

```python
duplicate_rows = int(frame.duplicated(keep="first").sum())
```

**Effect:** DQS only detects rows that are **entirely identical**. Cloning a single key column (e.g. stamping one ID onto 80% of rows) while leaving all other columns intact produces **zero full-row duplicates** → `uniqueness_score` is unaffected.

**There is no column-level or primary-key uniqueness check.** Key duplication is invisible to DQS unless accompanied by null injections in other columns that hurt `completeness`.

**Impact on corruption design:** To make `uniqueness` fail, rows must be fully cloned. To make key-duplication *detectable*, pair it with `completeness` hits (null injection in other columns).

---

## 4. `column_quality_score` only grades column *names*, not column *presence*

**Where:** `column_quality_score()` penalises:
- Blank names
- Duplicate names
- Whitespace-padded names
- Unnamed columns

**Effect:** Adding a new unexpected column (schema drift) has **zero impact** on `column_quality` as long as the name is well-formed. Schema drift is completely invisible to DQS from a structural standpoint, only impacting `completeness` if the new column contains nulls.

**Impact:** To make schema drift scorable, the new column must have significant null density — the penalty routes through `completeness`, not `column_quality`.

---

## 5. `completeness_score` depends on NULL_MARKERS and CSV round-trip

**Where:** `NULL_MARKERS = {"", "na", "n/a", "null", "none", "nan", ...}`

DQS reads all CSVs with `dtype=str, keep_default_na=False`. Python `None` → written to CSV as empty string `""` → re-read as `""`, which is in `NULL_MARKERS`.

**This works correctly** but is an implicit dependency on CSV encoding behavior. If a dataset is passed in-memory (bypassing the CSV round-trip), `None` must still serialize to `""` or an explicit null marker for `completeness_score` to count it.

**Risk:** Datasets using custom null markers (e.g. `"NULL"`, `"-1"`, `"#N/A"`) are only partially covered by `NULL_MARKERS`. Values like `"-1"` as a sentinel are **not detected** as null — they pass through as valid numerics.

---

## 6. `string_cleanliness` is dtype-agnostic

**Where:** `string_cleanliness_score()`:

```python
dirty_cells = int(((frame != stripped) & ~missing).to_numpy().sum())
```

DQS reads everything as `str`. So even numeric columns with whitespace artifacts (e.g. `" 123.45 "` from a CSV generator) count as dirty cells.

**Effect:** Whitespace corruption on string columns correctly hits `string_cleanliness`. But any numeric column with whitespace side-effects also leaks into the score. This means `string_cleanliness` can double-penalize a row where both a text field is whitespace-padded and a numeric field carries an incidental space.

---

## Summary

| Issue | Affected Metric | Severity |
|---|---|---|
| Currency/prefix-formatted numeric → treated as clean text | `type_consistency` | **High** — score stays at 1.0 |
| `<90%` numeric columns skipped entirely | `numeric_sanity` | **High** — complete blind spot |
| Key-column cloning not detected | `uniqueness` | **High** — only row-level dup check |
| Schema drift not penalized structurally | `column_quality` | **Medium** — routes through completeness only |
| Custom/sentinel null markers missed | `completeness` | **Medium** — depends on dataset convention |
| Numeric columns leak into string cleanliness | `string_cleanliness` | **Low** — minor double-counting |

---

## What This Means for Corruption Design

| Corruption Goal | Reliable DQS Signal | Unreliable / Absent Signal |
|---|---|---|
| Inject mixed type tokens (30%) | `type_consistency` ↓ ~0.70 | `numeric_sanity` (col < 90% numeric — skipped) |
| Inject domain-invalid negatives | `numeric_sanity` (if dist is tight) | `numeric_sanity` (if dist is wide) |
| Clone key column only | *(nothing)* | `uniqueness` (not triggered) |
| Clone key + null other column | `completeness` ↓ | `uniqueness` (still not triggered) |
| Add new column (40% null) | `completeness` ↓ | `column_quality` (not triggered) |
| 50% null on key column | `completeness` ↓ strongly | — |
| 30–50% whitespace on text column | `string_cleanliness` ↓ | — |
