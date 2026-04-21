"""MEDUSA Task Definitions — v4.0.

Six tasks aligned with the 30-day gauntlet architecture.  Each task scores a
completed episode using v4.0 state flags (current_day, silver_row_count,
total_quarantine_rows, did_evolve_schema, etc.).

Usage::

    from medusa_env.tasks import TASKS, score_episode

    task = TASKS["basic_pipeline"]   # easy
    env = MedusaEnv(n_fact_rows=50)
    obs = env.reset(seed=task.seed)

    # ... agent takes actions ...

    result = score_episode(task.id, env.state)
    print(f"Score: {result.score:.2f}  ({result.grade})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional


if TYPE_CHECKING:
    from server.medusa_env import _EpisodeTables
    from .models import MedusaState


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A MEDUSA task definition."""

    id: str
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    seed: int                # Controls ScenarioGenerator / DayDataGenerator seed
    description: str
    success_criteria: List[str]
    scoring_rubric: Dict[str, float]


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Outcome of scoring a completed episode against a task."""

    task_id: str
    score: float             # 0.0 – 1.0
    grade: str               # "S" | "A" | "B" | "C" | "F"
    breakdown: Dict[str, float]   # per-criterion scores
    passed: bool
    notes: List[str] = field(default_factory=list)


def _grade(score: float) -> str:
    if score >= 0.90:
        return "S"
    if score >= 0.75:
        return "A"
    if score >= 0.55:
        return "B"
    if score >= 0.35:
        return "C"
    return "F"


# ---------------------------------------------------------------------------
# Task catalogue — v4.0 (30-day gauntlet native)
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {

    # ── EASY: Basic Pipeline ───────────────────────────────────────────────
    "basic_pipeline": Task(
        id="basic_pipeline",
        name="Basic Pipeline",
        difficulty="easy",
        seed=42,
        description=(
            "The agent must survive at least 5 days of the 30-day gauntlet, "
            "cleaning anomalies, merging daily batches into Silver, and "
            "committing each day without crashing."
        ),
        success_criteria=[
            "Survive at least 5 days (current_day >= 6)",
            "Silver table is non-empty",
            "No terminal crash",
            "Cumulative reward > 0",
        ],
        scoring_rubric={
            "survived_5_days":   0.30,
            "silver_built":      0.20,
            "no_crash":          0.20,
            "positive_reward":   0.15,
            "grader_passed":     0.15,
        },
    ),

    # ── MEDIUM: Survive Day 8 (Type Trap) ──────────────────────────────────
    "survive_day8": Task(
        id="survive_day8",
        name="Survive Day 8 — Type Trap",
        difficulty="medium",
        seed=42,
        description=(
            "Day 8 injects revenue as '$50.50' strings. The agent must "
            "strip the '$' and cast to float64 before committing. "
            "The grader checks Silver.dtypes['revenue'] == float64."
        ),
        success_criteria=[
            "Survive past Day 8 (current_day >= 9)",
            "Silver revenue column is float64",
            "Silver table is non-empty",
        ],
        scoring_rubric={
            "survived_day8":     0.35,
            "revenue_is_float":  0.30,
            "silver_built":      0.15,
            "grader_passed":     0.20,
        },
    ),

    # ── MEDIUM: Survive Day 14 (OOM Trap) ──────────────────────────────────
    "survive_day14": Task(
        id="survive_day14",
        name="Survive Day 14 — OOM Trap",
        difficulty="medium",
        seed=42,
        description=(
            "Day 14 injects massive duplicate user_ids. The agent must "
            "profile → deduplicate → merge to avoid exceeding the memory "
            "limit. Without dedup, EXECUTE_MERGE will BLOCK."
        ),
        success_criteria=[
            "Survive past Day 14 (current_day >= 15)",
            "Deduplication was performed on Day 14",
            "Silver table is non-empty",
        ],
        scoring_rubric={
            "survived_day14":    0.35,
            "dedup_used":        0.25,
            "silver_built":      0.15,
            "grader_passed":     0.25,
        },
    ),

    # ── MEDIUM: Survive Day 21 (Schema Drift) ──────────────────────────────
    "survive_day21": Task(
        id="survive_day21",
        name="Survive Day 21 — Schema Drift",
        difficulty="medium",
        seed=42,
        description=(
            "Day 21 introduces a new 'promo_code' column. The agent must "
            "call EVOLVE_SILVER_SCHEMA to add the column to the Data "
            "Contract before committing."
        ),
        success_criteria=[
            "Survive past Day 21 (current_day >= 22)",
            "EVOLVE_SILVER_SCHEMA was called",
            "Silver contains promo_code column",
        ],
        scoring_rubric={
            "survived_day21":    0.30,
            "schema_evolved":    0.30,
            "promo_in_silver":   0.20,
            "grader_passed":     0.20,
        },
    ),

    # ── MEDIUM: Survive Day 28 (Null Nuke) ─────────────────────────────────
    "survive_day28": Task(
        id="survive_day28",
        name="Survive Day 28 — Null Nuke",
        difficulty="medium",
        seed=42,
        description=(
            "Day 28 has 20% of user_id as NULL. The agent must "
            "QUARANTINE_ROWS where user_id IS NULL before merging. "
            "The grader checks that no NULL user_id rows are in Silver."
        ),
        success_criteria=[
            "Survive past Day 28 (current_day >= 29)",
            "Quarantine contains rows",
            "No NULL user_id in Silver",
        ],
        scoring_rubric={
            "survived_day28":      0.30,
            "quarantine_used":     0.25,
            "no_null_in_silver":   0.25,
            "grader_passed":       0.20,
        },
    ),

    # ── HARD: Full 30-Day Gauntlet ─────────────────────────────────────────
    "gauntlet_30day": Task(
        id="gauntlet_30day",
        name="Full 30-Day Gauntlet",
        difficulty="hard",
        seed=42,
        description=(
            "Complete all 30 days of the data pipeline gauntlet. "
            "The agent must handle all 4 trap days (8, 14, 21, 28), "
            "maintain the quarantine ceiling ≤ 5% (excluding Day 28), "
            "and earn the +100 completion bonus."
        ),
        success_criteria=[
            "Complete all 30 days (stage == 'committed')",
            "All grader checks passed",
            "Quarantine ceiling ≤ 5% (excluding Day 28)",
            "Cumulative reward > 0",
        ],
        scoring_rubric={
            "completed_30_days":   0.25,
            "all_graders_passed":  0.20,
            "quarantine_ceiling":  0.20,
            "positive_reward":     0.15,
            "high_reward":         0.20,
        },
    ),
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _build_checks(
    state: "MedusaState",
    tables: "Optional[_EpisodeTables]",
) -> Dict[str, bool]:
    """Build reusable boolean checks used across v4.0 task rubrics."""
    silver_cols = set()
    if tables is not None and not tables.silver.empty:
        silver_cols = set(tables.silver.columns)

    import numpy as np
    revenue_is_float = False
    if tables is not None and not tables.silver.empty and "revenue" in tables.silver.columns:
        revenue_is_float = tables.silver["revenue"].dtype == np.float64

    null_in_silver = True  # default: ok
    if tables is not None and not tables.silver.empty and "user_id" in tables.silver.columns:
        null_in_silver = int(tables.silver["user_id"].isnull().sum()) == 0

    return {
        # Day survival
        "survived_5_days": state.current_day >= 6 or state.stage == "committed",
        "survived_day8": state.current_day >= 9 or state.stage == "committed",
        "survived_day14": state.current_day >= 15 or state.stage == "committed",
        "survived_day21": state.current_day >= 22 or state.stage == "committed",
        "survived_day28": state.current_day >= 29 or state.stage == "committed",
        "completed_30_days": state.stage == "committed",

        # Pipeline basics
        "silver_built": state.silver_row_count > 0,
        "no_crash": state.stage != "failed",
        "positive_reward": state.cumulative_reward > 0,
        "high_reward": state.cumulative_reward >= 500,
        "grader_passed": state.grader_passed,
        "all_graders_passed": state.grader_passed and state.stage == "committed",

        # Trap-specific
        "revenue_is_float": revenue_is_float,
        "dedup_used": state.did_dedup_today or state.did_dedup_b,
        "schema_evolved": state.did_evolve_schema,
        "promo_in_silver": "promo_code" in silver_cols,
        "quarantine_used": state.total_quarantine_rows > 0,
        "no_null_in_silver": null_in_silver,

        # Quarantine ceiling (Day 28 excluded)
        "quarantine_ceiling": (
            (state.total_quarantine_rows - state.day28_quarantine_rows)
            / max(state.total_raw_rows, 1)
        ) <= 0.05 if state.total_raw_rows > 0 else True,
    }


def _apply_check(
    breakdown: Dict[str, float],
    rubric: Dict[str, float],
    checks: Dict[str, bool],
    notes: List[str],
    key: str,
    failure_note: str,
) -> None:
    """Apply a boolean check to the rubric and attach a note on failure."""
    if key not in rubric:
        return
    ok = checks.get(key, False)
    breakdown[key] = rubric[key] if ok else 0.0
    if not ok:
        notes.append(failure_note)


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def score_episode(
    task_id: str,
    state: "MedusaState",
    tables: "Optional[_EpisodeTables]" = None,
) -> TaskResult:
    """Score a completed MEDUSA v4.0 episode against the named task.

    Args:
        task_id: Any task id defined in ``TASKS``.
        state: Final ``MedusaState`` after the episode ended.
        tables: Episode tables (used for Silver inspection). Optional.

    Returns:
        TaskResult with score in [0.0, 1.0].
    """
    task = TASKS.get(task_id)
    if task is None:
        raise ValueError(f"Unknown task_id={task_id!r}. Valid: {list(TASKS)}")

    if state.stage not in ("committed", "failed"):
        return TaskResult(
            task_id=task_id, score=0.0, grade="F",
            breakdown={}, passed=False,
            notes=["Episode not finished — COMMIT_DAY was never issued."],
        )

    breakdown: Dict[str, float] = {}
    notes: List[str] = []
    rubric = task.scoring_rubric
    checks = _build_checks(state, tables)

    # Apply all rubric checks generically
    for key in rubric:
        failure_note = f"{key} check failed."
        if key in checks:
            _apply_check(breakdown, rubric, checks, notes, key, failure_note)
        else:
            # Unknown check key — award 0
            breakdown[key] = 0.0
            notes.append(f"Unknown rubric key: {key}")

    # ── Final score ───────────────────────────────────────────────────────
    total = sum(breakdown.values())
    score = max(0.1, min(0.95, total))
    passed = score >= 0.10

    return TaskResult(
        task_id=task_id,
        score=round(score, 4),
        grade=_grade(score),
        breakdown=breakdown,
        passed=passed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Convenience: score all tasks
# ---------------------------------------------------------------------------

def score_all_tasks(
    results: Dict[str, tuple],  # task_id → (state, tables)
) -> Dict[str, TaskResult]:
    """Score multiple completed episodes, one per task."""
    return {
        task_id: score_episode(task_id, state, tables)
        for task_id, (state, tables) in results.items()
    }
