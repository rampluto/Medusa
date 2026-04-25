#!/usr/bin/env python3
"""Score CSV data quality on a 0-1 scale using only the standard library."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = {
    "readability": 0.07,
    "completeness": 0.18,
    "uniqueness": 0.10,
    "type_consistency": 0.16,
    "date_format_sanity": 0.12,
    "column_quality": 0.09,
    "string_cleanliness": 0.06,
    "numeric_sanity": 0.22,
}

NULL_MARKERS = {"", "na", "n/a", "null", "none", "nan", "nat", "-"}
DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
)
DATE_COLUMN_HINTS = ("date", "time", "timestamp", "created", "updated", "dt")
BOOL_VALUES = {"true", "false", "yes", "no", "y", "n", "0", "1"}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def rounded(value: float, digits: int = 4) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, digits)


def is_null(value: str) -> bool:
    return value.strip().lower() in NULL_MARKERS


def can_parse_float(value: str) -> bool:
    try:
        number = float(value.strip())
    except ValueError:
        return False
    return math.isfinite(number)


def can_parse_date(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text)
        return True
    except ValueError:
        pass

    return any(_matches_date_format(text, fmt) for fmt in DATE_FORMATS)


def detect_date_format(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    try:
        datetime.fromisoformat(text)
        return "iso"
    except ValueError:
        pass

    for fmt in DATE_FORMATS:
        if _matches_date_format(text, fmt):
            return fmt
    return None


def can_parse_bool(value: str) -> bool:
    return value.strip().lower() in BOOL_VALUES


def _matches_date_format(text: str, fmt: str) -> bool:
    try:
        datetime.strptime(text, fmt)
    except ValueError:
        return False
    return True


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def load_csv(path: Path) -> tuple[list[str], list[list[str]], str | None]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001 - CLI should report any read/parser failure.
        return [], [], str(exc)

    if not rows:
        return [], [], "CSV is empty"

    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows, None


def normalize_row(row: list[str], column_count: int) -> list[str]:
    if len(row) >= column_count:
        return row[:column_count]
    return row + [""] * (column_count - len(row))


def column_values(rows: list[list[str]], index: int) -> list[str]:
    return [row[index] if index < len(row) else "" for row in rows]


def completeness_score(rows: list[list[str]], column_count: int) -> tuple[float, dict[str, Any]]:
    total_cells = max(len(rows) * column_count, 1)
    missing_cells = sum(is_null(row[index]) for row in rows for index in range(column_count))
    score = 1.0 - (missing_cells / total_cells)
    return score, {
        "missing_cells": missing_cells,
        "total_cells": total_cells,
        "missing_ratio": rounded(missing_cells / total_cells),
    }


def uniqueness_score(rows: list[list[str]]) -> tuple[float, dict[str, Any]]:
    if not rows:
        return 0.0, {"duplicate_rows": 0, "duplicate_ratio": 1.0}

    counts = Counter(tuple(row) for row in rows)
    duplicate_rows = sum(count - 1 for count in counts.values() if count > 1)
    duplicate_ratio = duplicate_rows / len(rows)
    return 1.0 - duplicate_ratio, {
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": rounded(duplicate_ratio),
    }


def infer_column_type(values: list[str]) -> tuple[str, float, dict[str, Any]]:
    non_null = [value for value in values if not is_null(value)]
    if not non_null:
        return "empty", 1.0, {
            "non_null_values": 0,
            "numeric_ratio": 0.0,
            "date_ratio": 0.0,
            "boolean_ratio": 0.0,
            "invalid_values": 0,
        }

    numeric_ratio = sum(can_parse_float(value) for value in non_null) / len(non_null)
    date_ratio = sum(can_parse_date(value) for value in non_null) / len(non_null)
    boolean_ratio = sum(can_parse_bool(value) for value in non_null) / len(non_null)
    candidates = {
        "numeric": numeric_ratio,
        "datetime": date_ratio,
        "boolean": boolean_ratio,
    }
    inferred_type, best_ratio = max(candidates.items(), key=lambda item: item[1])

    if best_ratio >= 0.60:
        parseable_ratio = best_ratio
    else:
        inferred_type = "text"
        parseable_ratio = 1.0 - max(numeric_ratio, date_ratio, boolean_ratio)

    return inferred_type, parseable_ratio, {
        "non_null_values": len(non_null),
        "numeric_ratio": rounded(numeric_ratio),
        "date_ratio": rounded(date_ratio),
        "boolean_ratio": rounded(boolean_ratio),
        "invalid_values": int(round(len(non_null) * (1.0 - parseable_ratio))),
    }


def type_consistency_score(header: list[str], rows: list[list[str]]) -> tuple[float, dict[str, Any]]:
    if not header:
        return 0.0, {"columns": {}}

    scores = []
    details = {}
    for index, name in enumerate(header):
        inferred_type, parseable_ratio, profile = infer_column_type(column_values(rows, index))
        scores.append(parseable_ratio)
        details[name or f"column_{index + 1}"] = {
            "inferred_type": inferred_type,
            "parseable_ratio": rounded(parseable_ratio),
            **profile,
        }

    return statistics.fmean(scores), {"columns": details}


def looks_like_date_column(column_name: str) -> bool:
    normalized = column_name.strip().lower().replace("-", "_")
    return any(hint in normalized for hint in DATE_COLUMN_HINTS)


def date_format_sanity_score(header: list[str], rows: list[list[str]]) -> tuple[float, dict[str, Any]]:
    date_columns = []
    scores = []
    details = {}

    for index, name in enumerate(header):
        values = [value for value in column_values(rows, index) if not is_null(value)]
        if not values:
            continue

        formats = [detect_date_format(value) for value in values]
        parseable_count = sum(fmt is not None for fmt in formats)
        parseable_ratio = parseable_count / len(values)
        is_date_column = looks_like_date_column(name) or parseable_ratio >= 0.60
        if not is_date_column:
            continue

        date_columns.append(name or f"column_{index + 1}")
        valid_formats = [fmt for fmt in formats if fmt is not None]
        dominant_count = Counter(valid_formats).most_common(1)[0][1] if valid_formats else 0
        dominant_format_ratio = dominant_count / max(parseable_count, 1)
        score = parseable_ratio * dominant_format_ratio
        scores.append(score)
        details[name or f"column_{index + 1}"] = {
            "parseable_date_ratio": rounded(parseable_ratio),
            "dominant_format_ratio": rounded(dominant_format_ratio),
            "invalid_date_values": len(values) - parseable_count,
            "formats_seen": dict(Counter(valid_formats)),
        }

    if not date_columns:
        return 1.0, {"date_columns": [], "checked_date_columns": 0}

    return statistics.fmean(scores), {
        "date_columns": date_columns,
        "checked_date_columns": len(date_columns),
        "columns": details,
    }


def column_quality_score(header: list[str]) -> tuple[float, dict[str, Any]]:
    if not header:
        return 0.0, {"issues": ["no_columns"]}

    counts = Counter(header)
    blank_count = sum(not column.strip() for column in header)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    whitespace_count = sum(column != column.strip() for column in header)
    unnamed_count = sum(column.lower().startswith("unnamed:") for column in header)

    duplicate_penalty = duplicate_count * 2
    score = 1.0 - ((blank_count + duplicate_penalty + whitespace_count + unnamed_count) / len(header))
    return clamp(score), {
        "blank_column_names": blank_count,
        "duplicate_column_names": duplicate_count,
        "names_with_outer_whitespace": whitespace_count,
        "unnamed_columns": unnamed_count,
    }


def string_cleanliness_score(rows: list[list[str]], column_count: int) -> tuple[float, dict[str, Any]]:
    checked_cells = 0
    dirty_cells = 0
    for row in rows:
        for index in range(column_count):
            value = row[index]
            if is_null(value):
                continue
            checked_cells += 1
            dirty_cells += value != value.strip()

    dirty_ratio = dirty_cells / max(checked_cells, 1)
    return 1.0 - dirty_ratio, {
        "dirty_string_cells": dirty_cells,
        "checked_non_null_cells": checked_cells,
        "dirty_string_ratio": rounded(dirty_ratio),
    }


def numeric_sanity_score(header: list[str], rows: list[list[str]]) -> tuple[float, dict[str, Any]]:
    numeric_columns = []
    checked_cells = 0
    outlier_cells = 0
    constant_numeric_columns = 0

    for index, name in enumerate(header):
        values = [value.strip() for value in column_values(rows, index) if not is_null(value)]
        numeric_values = [float(value) for value in values if can_parse_float(value)]
        if len(numeric_values) < 3 or len(numeric_values) / max(len(values), 1) < 0.90:
            continue

        numeric_columns.append(name or f"column_{index + 1}")
        checked_cells += len(numeric_values)
        unique_values = set(numeric_values)
        if len(unique_values) <= 1:
            constant_numeric_columns += 1
            continue

        sorted_values = sorted(numeric_values)
        q1 = quantile(sorted_values, 0.25)
        q3 = quantile(sorted_values, 0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        outlier_cells += sum(value < lower or value > upper for value in numeric_values)

    if not numeric_columns:
        return 1.0, {
            "numeric_columns": [],
            "outlier_cells": 0,
            "outlier_ratio": 0.0,
            "constant_numeric_columns": 0,
        }

    outlier_ratio = outlier_cells / max(checked_cells, 1)
    constant_ratio = constant_numeric_columns / len(numeric_columns)
    score = 1.0 - min(1.0, outlier_ratio + (0.5 * constant_ratio))
    return score, {
        "numeric_columns": numeric_columns,
        "outlier_cells": outlier_cells,
        "checked_numeric_cells": checked_cells,
        "outlier_ratio": rounded(outlier_ratio),
        "constant_numeric_columns": constant_numeric_columns,
    }


def row_width_score(rows: list[list[str]], column_count: int) -> tuple[float, dict[str, Any]]:
    if not rows:
        return 0.0, {"bad_width_rows": 0, "bad_width_ratio": 1.0}

    bad_width_rows = sum(len(row) != column_count for row in rows)
    bad_width_ratio = bad_width_rows / len(rows)
    return 1.0 - bad_width_ratio, {
        "bad_width_rows": bad_width_rows,
        "bad_width_ratio": rounded(bad_width_ratio),
    }


def score_csv(path: Path) -> dict[str, Any]:
    header, raw_rows, error = load_csv(path)
    if error is not None:
        return {
            "file": str(path),
            "score": 0.0,
            "passed": False,
            "error": f"Could not read CSV: {error}",
            "component_scores": {"readability": 0.0},
            "details": {},
        }

    column_count = len(header)
    rows = [normalize_row(row, column_count) for row in raw_rows]
    row_width, row_width_details = row_width_score(raw_rows, column_count)
    readability = row_width if header else 0.0

    component_scores: dict[str, float] = {"readability": clamp(readability)}
    details: dict[str, Any] = {
        "rows": len(rows),
        "columns": column_count,
        "column_names": header,
        "row_width": row_width_details,
    }

    component_functions = {
        "completeness": lambda: completeness_score(rows, column_count),
        "uniqueness": lambda: uniqueness_score(rows),
        "type_consistency": lambda: type_consistency_score(header, rows),
        "date_format_sanity": lambda: date_format_sanity_score(header, rows),
        "column_quality": lambda: column_quality_score(header),
        "string_cleanliness": lambda: string_cleanliness_score(rows, column_count),
        "numeric_sanity": lambda: numeric_sanity_score(header, rows),
    }

    for name, scorer in component_functions.items():
        score, detail = scorer()
        component_scores[name] = clamp(score)
        details[name] = detail

    overall = sum(component_scores[name] * DEFAULT_WEIGHTS[name] for name in DEFAULT_WEIGHTS)
    return {
        "file": str(path),
        "score": rounded(clamp(overall)),
        "passed": clamp(overall) >= 0.80,
        "component_scores": {name: rounded(score) for name, score in component_scores.items()},
        "weights": DEFAULT_WEIGHTS,
        "details": details,
    }


def print_text_report(result: dict[str, Any]) -> None:
    print(f"File: {result['file']}")
    print(f"Data quality score: {result['score']:.4f} / 1.0000")
    print(f"Passed threshold 0.80: {result['passed']}")

    error = result.get("error")
    if error:
        print(f"Error: {error}")
        return

    details = result["details"]
    print(f"Rows: {details['rows']}")
    print(f"Columns: {details['columns']}")
    print("\nComponent scores:")
    for name, score in result["component_scores"].items():
        weight = result["weights"].get(name, 0.0)
        print(f"  - {name}: {score:.4f} (weight {weight:.2f})")

    print("\nKey issues:")
    print(f"  - Bad-width rows: {details['row_width']['bad_width_rows']}")
    print(f"  - Missing cells: {details['completeness']['missing_cells']}")
    print(f"  - Duplicate rows: {details['uniqueness']['duplicate_rows']}")
    print(f"  - Duplicate columns: {details['column_quality']['duplicate_column_names']}")
    print(f"  - Checked date columns: {details['date_format_sanity']['checked_date_columns']}")
    print(f"  - Dirty string cells: {details['string_cleanliness']['dirty_string_cells']}")
    print(f"  - Numeric outlier cells: {details['numeric_sanity']['outlier_cells']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate a 0-1 data quality score for a CSV file.",
    )
    parser.add_argument("csv_file", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a short text report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score_csv(args.csv_file)

    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_text_report(result)

    return 0 if result.get("score", 0.0) >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
