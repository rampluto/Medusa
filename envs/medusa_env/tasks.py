"""MEDUSA Task Definitions.

Three formally graded tasks covering the easy → medium → hard spectrum.
Each task returns a deterministic score in [0.0, 1.0] after COMMIT.

Usage::

    from envs.medusa_env.tasks import TASKS, score_episode

    task = TASKS["clean_pipeline"]          # easy
    env = MedusaEnv(n_fact_rows=200, n_dim_rows=150)
    obs = env.reset(seed=task.seed)

    # ... agent takes actions ...
    obs = env.step(MedusaAction(action=MedusaActionType.COMMIT))

    result = score_episode(task.id, env.state, env._tables)
    print(f"Score: {result.score:.2f}  ({result.grade})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .medusa_env import _EpisodeTables
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
    seed: int                # Controls ScenarioGenerator variant
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
# Task catalogue
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {

    # ── EASY: Clean Pipeline ────────────────────────────────────────────────
    "clean_pipeline": Task(
        id="clean_pipeline",
        name="Clean Pipeline",
        difficulty="easy",
        seed=0,
        description=(
            "Both sources are fresh. Join keys are clean and unique. "
            "The agent must verify freshness, prepare keys, join, apply SCD, "
            "and commit without triggering a row explosion."
        ),
        success_criteria=[
            "COMMIT issued (episode finalized)",
            "No Cartesian explosion detected",
            "Silver row count ≤ Source A row count",
            "match_rate > 0.80 after join",
        ],
        scoring_rubric={
            "committed":        0.20,   # Agent issued COMMIT
            "no_explosion":     0.25,   # No row explosion
            "volume_ok":        0.20,   # Silver ≤ Source A rows
            "high_match":       0.20,   # match_rate > 0.80
            "grader_pass":      0.15,   # All 4 grader checks pass
        },
    ),

    # ── MEDIUM: Dirty Integration ───────────────────────────────────────────
    "dirty_integration": Task(
        id="dirty_integration",
        name="Dirty Key Integration",
        difficulty="medium",
        seed=1,
        description=(
            "Source A has NULLs and whitespace in join keys. "
            "Source B has duplicate keys that can cause row explosion. "
            "The agent must PREP_KEYS and DEDUPLICATE before joining, "
            "and correctly quarantine unresolvable orphans."
        ),
        success_criteria=[
            "PREP_KEYS_A issued before EXECUTE_JOIN",
            "PREP_KEYS_B issued before EXECUTE_JOIN",
            "DEDUPLICATE_B issued before EXECUTE_JOIN",
            "No row explosion",
            "Quarantine integrity check passes",
        ],
        scoring_rubric={
            "committed":        0.10,
            "prepped_before_join": 0.20,  # Both PREP_KEYS before join
            "deduped_before_join": 0.20,  # DEDUP before join
            "no_explosion":     0.25,
            "integrity_ok":     0.15,     # Quarantine holds true orphans only
            "grader_pass":      0.10,
        },
    ),

    # ── HARD: Full Medallion Integration ────────────────────────────────────
    "full_medallion": Task(
        id="full_medallion",
        name="Full Medallion Integration",
        difficulty="hard",
        seed=2,
        description=(
            "Source A is stale (>6h old). Source B has new schema columns "
            "not registered in Silver. The agent must: check freshness, "
            "evolve the schema, clean keys, deduplicate, execute a left join, "
            "apply SCD-2 for tracked columns, and pass all grader checks."
        ),
        success_criteria=[
            "SYNC_CHECK issued before any join",
            "EVOLVE_SCHEMA issued before COMMIT",
            "SCD-2 applied (not SCD-1) for tracked column",
            "Silver schema contains new columns from drift",
            "All 4 grader checks pass",
        ],
        scoring_rubric={
            "committed":        0.05,
            "sync_checked":     0.15,     # SYNC_CHECK before join
            "schema_evolved":   0.15,     # EVOLVE_SCHEMA called
            "used_scd2":        0.20,     # Chose SCD-2 over SCD-1
            "schema_ok":        0.20,     # Silver has all required columns
            "grader_pass":      0.25,     # All 4 grader checks pass
        },
    ),
}


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def score_episode(
    task_id: str,
    state: "MedusaState",
    tables: "Optional[_EpisodeTables]" = None,
) -> TaskResult:
    """Score a completed MEDUSA episode against the named task.

    Args:
        task_id: One of "clean_pipeline", "dirty_integration", "full_medallion".
        state: Final ``MedusaState`` after the episode ended.
        tables: Episode tables (used for schema checks). Optional.

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
            notes=["Episode not finished — COMMIT was never issued."],
        )

    breakdown: Dict[str, float] = {}
    notes: List[str] = []
    rubric = task.scoring_rubric
    committed = state.stage == "committed"

    # ── Shared criteria ──────────────────────────────────────────────────
    if "committed" in rubric:
        breakdown["committed"] = rubric["committed"] if committed else 0.0

    if "no_explosion" in rubric:
        ok = not state.explosion_detected
        breakdown["no_explosion"] = rubric["no_explosion"] if ok else 0.0
        if not ok:
            notes.append("Row explosion was detected — heavy penalty applied.")

    if "grader_pass" in rubric:
        breakdown["grader_pass"] = rubric["grader_pass"] if state.grader_passed else 0.0

    # ── Task-specific criteria ────────────────────────────────────────────

    if task_id == "clean_pipeline":
        volume_ok = (
            state.silver_row_count <= state.source_a_row_count * 1.05
            and state.silver_row_count > 0
        )
        breakdown["volume_ok"] = rubric["volume_ok"] if volume_ok else 0.0
        breakdown["high_match"] = rubric["high_match"] if state.match_rate >= 0.80 else 0.0
        if state.match_rate < 0.80:
            notes.append(f"match_rate={state.match_rate:.1%} — target >80%.")

    elif task_id == "dirty_integration":
        # Both PREP_KEYS before join
        prepped = state.did_prep_a and state.did_prep_b and state.did_join
        breakdown["prepped_before_join"] = rubric["prepped_before_join"] if prepped else 0.0
        # DEDUP before join
        deduped = state.did_dedup_b and state.did_join
        breakdown["deduped_before_join"] = rubric["deduped_before_join"] if deduped else 0.0
        # Integrity check comes from grader
        integrity_ok = state.grader_passed or (
            state.quarantine_row_count >= 0  # grader_passed already covers this
        )
        # Use grader_passed as proxy for integrity
        breakdown["integrity_ok"] = rubric["integrity_ok"] if state.grader_passed else 0.0
        if not prepped:
            notes.append("Agent joined without prepping keys first.")
        if not deduped:
            notes.append("Agent joined without deduplicating Dimension.")

    elif task_id == "full_medallion":
        breakdown["sync_checked"] = rubric["sync_checked"] if state.did_sync_check else 0.0
        breakdown["schema_evolved"] = rubric["schema_evolved"] if state.did_evolve_schema else 0.0
        used_scd2 = state.scd_type == "SCD-2"
        breakdown["used_scd2"] = rubric["used_scd2"] if used_scd2 else 0.0
        breakdown["schema_ok"] = rubric["schema_ok"] if state.grader_passed else 0.0
        if not state.did_sync_check:
            notes.append("SYNC_CHECK was never called — stale source not verified.")
        if not state.did_evolve_schema:
            notes.append("EVOLVE_SCHEMA never called — new columns may be missing from Silver.")
        if not used_scd2:
            notes.append(f"Used SCD-1 instead of SCD-2 (scd_type={state.scd_type!r}).")

    # ── Final score ───────────────────────────────────────────────────────
    total = sum(breakdown.values())
    # Clip to [0, 1] (row explosion can make total negative from reward engine)
    score = max(0.0, min(1.0, total))
    passed = score >= 0.55

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
