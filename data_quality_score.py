#!/usr/bin/env python3
"""Score CSV data quality on a 0-1 scale."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


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
FLOAT_PATTERN = r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
DATE_LIKE_PATTERN = r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$"
PROFILE_SAMPLE_ROWS = 50000

class ProgressBar:
    def __init__(self, enabled: bool, width: int = 28) -> None:
        self.enabled = enabled
        self.width = width
        self.label = ""
        self.total = 0
        self.current = 0
        self.last_percent = -1

    def start(self, label: str, total: int) -> None:
        if not self.enabled:
            return
        self.label = label
        self.total = max(total, 1)
        self.current = 0
        self.last_percent = -1
        self._render()

    def advance(self, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + amount)
        percent = int((self.current / self.total) * 100)
        if percent != self.last_percent or self.current == self.total:
            self._render()

    def finish(self) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self._render()
        print(file=sys.stderr)

    def _render(self) -> None:
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(ratio * 100)
        self.last_percent = percent
        print(f"\r{self.label}: [{bar}] {percent:3d}%", end="", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class FrameProfile:
    frame: pd.DataFrame
    row_count: int
    stripped: pd.DataFrame
    lowered: pd.DataFrame
    missing: pd.DataFrame
    sample: pd.DataFrame
    sample_stripped: pd.DataFrame
    sample_lowered: pd.DataFrame
    sample_missing: pd.DataFrame
    sampled: bool


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def rounded(value: float, digits: int = 4) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, digits)


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


def _matches_date_format(text: str, fmt: str) -> bool:
    try:
        datetime.strptime(text, fmt)
    except ValueError:
        return False
    return True


def count_file_lines(path: Path) -> int:
    try:
        with path.open("rb") as file:
            line_count = 0
            last_byte = b""
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                line_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
            return line_count + (1 if last_byte and last_byte != b"\n" else 0)
    except OSError:
        return 0


def load_csv(
    path: Path,
    progress: ProgressBar | None = None,
    sample_rows: int = PROFILE_SAMPLE_ROWS,
) -> tuple[list[str], pd.DataFrame, int, str | None]:
    if progress is not None and progress.enabled:
        progress.start("Reading CSV", 2)

    try:
        line_count = count_file_lines(path)
        if progress is not None:
            progress.advance()
        max_rows = sample_rows + 1 if line_count > sample_rows + 1 else None
        frame = pd.read_csv(
            path,
            header=None,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            nrows=max_rows,
        )
        if progress is not None:
            progress.advance()
    except Exception as exc:  # noqa: BLE001 - CLI should report any read/parser failure.
        return [], pd.DataFrame(), 0, str(exc)
    finally:
        if progress is not None:
            progress.finish()

    if frame.empty:
        return [], pd.DataFrame(), 0, "CSV is empty"

    header = frame.iloc[0].astype(str).tolist()
    data = frame.iloc[1:].reset_index(drop=True)
    data.columns = range(data.shape[1])
    total_rows = max(line_count - 1, len(data))
    return header, data, total_rows, None


def normalize_frame(frame: pd.DataFrame, column_count: int) -> pd.DataFrame:
    rows = len(frame)
    normalized = frame.iloc[:, :column_count].copy()
    for index in range(normalized.shape[1], column_count):
        normalized[index] = ""
    if normalized.shape[1] == 0:
        return pd.DataFrame(index=range(rows))
    normalized = normalized.iloc[:, :column_count]
    normalized.columns = range(column_count)
    return normalized.fillna("").astype(str)


def strip_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda column: column.str.strip())


def build_frame_profile(
    frame: pd.DataFrame,
    row_count: int,
    sample_rows: int = PROFILE_SAMPLE_ROWS,
) -> FrameProfile:
    sampled = row_count > len(frame) or len(frame) > sample_rows
    sample = frame.head(sample_rows) if len(frame) > sample_rows else frame
    sample_stripped = strip_frame(sample)
    sample_lowered = sample_stripped.apply(lambda column: column.str.lower())
    sample_missing = sample_lowered.isin(NULL_MARKERS)
    if sampled:
        stripped = sample_stripped
        lowered = sample_lowered
        missing = sample_missing
    else:
        stripped = sample_stripped
        lowered = sample_lowered
        missing = sample_missing
    return FrameProfile(
        frame=frame,
        row_count=row_count,
        stripped=stripped,
        lowered=lowered,
        missing=missing,
        sample=sample,
        sample_stripped=sample_stripped,
        sample_lowered=sample_lowered,
        sample_missing=sample_missing,
        sampled=sampled,
    )


def completeness_score(profile: FrameProfile, column_count: int) -> tuple[float, dict[str, Any]]:
    total_cells = max(profile.row_count * column_count, 1)
    checked_cells = max(len(profile.sample) * column_count, 1) if profile.sampled else total_cells
    missing_cells = int(profile.sample_missing.to_numpy().sum()) if column_count else 0
    missing_ratio = missing_cells / checked_cells
    reported_missing_cells = int(round(missing_ratio * total_cells)) if profile.sampled else missing_cells
    score = 1.0 - missing_ratio
    detail: dict[str, Any] = {
        "missing_cells": reported_missing_cells,
        "checked_cells": checked_cells,
        "total_cells": total_cells,
        "missing_ratio": rounded(missing_ratio),
    }
    if profile.sampled:
        detail["sampled_rows"] = len(profile.sample)
        detail["sample_missing_cells"] = missing_cells
        detail["estimated_missing_cells"] = reported_missing_cells
    return score, detail


def uniqueness_score(profile: FrameProfile) -> tuple[float, dict[str, Any]]:
    frame = profile.sample if profile.sampled else profile.frame
    if frame.empty:
        return 0.0, {"duplicate_rows": 0, "duplicate_ratio": 1.0}

    duplicate_rows = int(frame.duplicated(keep="first").sum())
    duplicate_ratio = duplicate_rows / len(frame)
    detail: dict[str, Any] = {
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": rounded(duplicate_ratio),
    }
    if profile.sampled:
        detail["sampled_rows"] = len(profile.sample)
    return 1.0 - duplicate_ratio, detail


def infer_column_type(
    column_name: str,
    stripped: pd.Series,
    lowered: pd.Series,
    missing: pd.Series,
) -> tuple[str, float, dict[str, Any]]:
    non_null_count = int((~missing).sum())
    if non_null_count == 0:
        return "empty", 1.0, {
            "non_null_values": 0,
            "numeric_ratio": 0.0,
            "date_ratio": 0.0,
            "boolean_ratio": 0.0,
            "invalid_values": 0,
        }

    values = stripped[~missing]
    lowered_values = lowered[~missing]
    numeric_ratio = float(values.str.fullmatch(FLOAT_PATTERN, na=False).mean())
    cheap_date_ratio = date_like_ratio(values)
    should_parse_dates = looks_like_date_column(column_name) or cheap_date_ratio >= 0.60
    date_ratio = float(values.map(can_parse_date).mean()) if should_parse_dates else 0.0
    boolean_ratio = float(lowered_values.isin(BOOL_VALUES).mean())
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
        "non_null_values": non_null_count,
        "numeric_ratio": rounded(numeric_ratio),
        "date_ratio": rounded(date_ratio),
        "boolean_ratio": rounded(boolean_ratio),
        "invalid_values": int(round(non_null_count * (1.0 - parseable_ratio))),
    }


def type_consistency_score(header: list[str], profile: FrameProfile) -> tuple[float, dict[str, Any]]:
    if not header:
        return 0.0, {"columns": {}}

    scores = []
    details = {}
    for index, name in enumerate(header):
        inferred_type, parseable_ratio, column_profile = infer_column_type(
            name,
            profile.sample_stripped.iloc[:, index],
            profile.sample_lowered.iloc[:, index],
            profile.sample_missing.iloc[:, index],
        )
        scores.append(parseable_ratio)
        details[name or f"column_{index + 1}"] = {
            "inferred_type": inferred_type,
            "parseable_ratio": rounded(parseable_ratio),
            **column_profile,
        }

    result: dict[str, Any] = {"columns": details}
    if profile.sampled:
        result["sampled_rows"] = len(profile.sample)
    return statistics.fmean(scores), result


def looks_like_date_column(column_name: str) -> bool:
    normalized = column_name.strip().lower().replace("-", "_")
    return any(hint in normalized for hint in DATE_COLUMN_HINTS)


def date_like_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.str.match(DATE_LIKE_PATTERN, na=False).mean())


def date_format_sanity_score(header: list[str], profile: FrameProfile) -> tuple[float, dict[str, Any]]:
    date_columns = []
    scores = []
    details = {}

    for index, name in enumerate(header):
        values = profile.sample_stripped.iloc[:, index][~profile.sample_missing.iloc[:, index]]
        if values.empty:
            continue

        is_named_date_column = looks_like_date_column(name)
        if not is_named_date_column and date_like_ratio(values) < 0.60:
            continue

        formats = values.map(detect_date_format).tolist()
        parseable_count = sum(fmt is not None for fmt in formats)
        parseable_ratio = parseable_count / len(values)
        is_date_column = is_named_date_column or parseable_ratio >= 0.60
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
        detail: dict[str, Any] = {"date_columns": [], "checked_date_columns": 0}
        if profile.sampled:
            detail["sampled_rows"] = len(profile.sample)
        return 1.0, detail

    detail = {
        "date_columns": date_columns,
        "checked_date_columns": len(date_columns),
        "columns": details,
    }
    if profile.sampled:
        detail["sampled_rows"] = len(profile.sample)
    return statistics.fmean(scores), detail


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


def string_cleanliness_score(profile: FrameProfile, column_count: int) -> tuple[float, dict[str, Any]]:
    frame = profile.sample if profile.sampled else profile.frame
    stripped = profile.sample_stripped if profile.sampled else profile.stripped
    missing = profile.sample_missing if profile.sampled else profile.missing
    if frame.empty or column_count == 0:
        return 1.0, {
            "dirty_string_cells": 0,
            "checked_non_null_cells": 0,
            "dirty_string_ratio": 0.0,
        }

    checked_cells = int((~missing).to_numpy().sum())
    dirty_cells = int(((frame != stripped) & ~missing).to_numpy().sum())

    dirty_ratio = dirty_cells / max(checked_cells, 1)
    detail: dict[str, Any] = {
        "dirty_string_cells": dirty_cells,
        "checked_non_null_cells": checked_cells,
        "dirty_string_ratio": rounded(dirty_ratio),
    }
    if profile.sampled:
        detail["sampled_rows"] = len(profile.sample)
    return 1.0 - dirty_ratio, detail


def numeric_sanity_score(header: list[str], profile: FrameProfile) -> tuple[float, dict[str, Any]]:
    numeric_columns = []
    checked_cells = 0
    outlier_cells = 0
    constant_numeric_columns = 0

    for index, name in enumerate(header):
        values = profile.sample_stripped.iloc[:, index][~profile.sample_missing.iloc[:, index]]
        numeric_text = values[values.str.fullmatch(FLOAT_PATTERN, na=False)]
        numeric_values = numeric_text.astype("float64")
        numeric_values = numeric_values[numeric_values.between(-math.inf, math.inf)]
        if len(numeric_values) < 3 or len(numeric_values) / max(len(values), 1) < 0.90:
            continue

        numeric_columns.append(name or f"column_{index + 1}")
        checked_cells += len(numeric_values)
        if numeric_values.nunique(dropna=True) <= 1:
            constant_numeric_columns += 1
            continue

        q1 = float(numeric_values.quantile(0.25))
        q3 = float(numeric_values.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        outlier_cells += int(((numeric_values < lower) | (numeric_values > upper)).sum())

    if not numeric_columns:
        detail: dict[str, Any] = {
            "numeric_columns": [],
            "outlier_cells": 0,
            "outlier_ratio": 0.0,
            "constant_numeric_columns": 0,
        }
        if profile.sampled:
            detail["sampled_rows"] = len(profile.sample)
        return 1.0, detail

    outlier_ratio = outlier_cells / max(checked_cells, 1)
    constant_ratio = constant_numeric_columns / len(numeric_columns)
    score = 1.0 - min(1.0, outlier_ratio + (0.5 * constant_ratio))
    detail: dict[str, Any] = {
        "numeric_columns": numeric_columns,
        "outlier_cells": outlier_cells,
        "checked_numeric_cells": checked_cells,
        "outlier_ratio": rounded(outlier_ratio),
        "constant_numeric_columns": constant_numeric_columns,
    }
    if profile.sampled:
        detail["sampled_rows"] = len(profile.sample)
    return score, detail


def row_width_score(profile: FrameProfile, column_count: int) -> tuple[float, dict[str, Any]]:
    if profile.frame.empty:
        return 0.0, {"bad_width_rows": 0, "bad_width_ratio": 1.0}

    missing = profile.sample_missing if profile.sampled else profile.missing
    populated_widths = (~missing).sum(axis=1)
    bad_width_rows = int((populated_widths > column_count).sum())
    bad_width_ratio = bad_width_rows / len(missing)
    detail: dict[str, Any] = {
        "bad_width_rows": bad_width_rows,
        "bad_width_ratio": rounded(bad_width_ratio),
    }
    if profile.sampled:
        detail["sampled_rows"] = len(profile.sample)
    return 1.0 - bad_width_ratio, detail


def score_csv(path: Path, show_progress: bool = False) -> dict[str, Any]:
    progress = ProgressBar(show_progress)
    header, raw_frame, row_count, error = load_csv(path, progress)
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
    component_count = 7
    progress.start("Scoring", component_count + 3)
    rows = normalize_frame(raw_frame, column_count)
    progress.advance()
    profile = build_frame_profile(rows, row_count)
    progress.advance()
    row_width, row_width_details = row_width_score(profile, column_count)
    progress.advance()
    readability = row_width if header else 0.0

    component_scores: dict[str, float] = {"readability": clamp(readability)}
    details: dict[str, Any] = {
        "rows": row_count,
        "columns": column_count,
        "column_names": header,
        "row_width": row_width_details,
    }

    component_functions = {
        "completeness": lambda: completeness_score(profile, column_count),
        "uniqueness": lambda: uniqueness_score(profile),
        "type_consistency": lambda: type_consistency_score(header, profile),
        "date_format_sanity": lambda: date_format_sanity_score(header, profile),
        "column_quality": lambda: column_quality_score(header),
        "string_cleanliness": lambda: string_cleanliness_score(profile, column_count),
        "numeric_sanity": lambda: numeric_sanity_score(header, profile),
    }

    for name, scorer in component_functions.items():
        score, detail = scorer()
        component_scores[name] = clamp(score)
        details[name] = detail
        progress.advance()

    progress.finish()

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
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars on stderr. Enabled by default for text output in a terminal.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    show_progress = args.progress or (not args.json and not args.no_progress and sys.stderr.isatty())
    result = score_csv(args.csv_file, show_progress=show_progress)

    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_text_report(result)

    return 0 if result.get("score", 0.0) >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
