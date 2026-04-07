"""MEDUSA Task Definitions.

Twelve formally graded tasks covering multiple easy, medium, and hard
pathways. Each task returns a deterministic score in [0.0, 1.0] after
COMMIT.

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

    # ── EASY: Schema Bootstrap ─────────────────────────────────────────────
    "schema_bootstrap": Task(
        id="schema_bootstrap",
        name="Schema Bootstrap",
        difficulty="easy",
        seed=3,
        description=(
            "Fresh sources arrive with new columns in both Bronze tables. "
            "The agent must evolve the Silver schema, execute a clean join, "
            "land a non-empty Silver table, and commit without row explosion."
        ),
        success_criteria=[
            "EVOLVE_SCHEMA issued before COMMIT",
            "No row explosion",
            "Silver contains the joined columns after drift",
            "Silver table is non-empty",
        ],
        scoring_rubric={
            "committed":           0.15,
            "no_explosion":        0.20,
            "schema_evolved":      0.25,
            "schema_materialized": 0.20,
            "silver_built":        0.20,
        },
    ),

    # ── MEDIUM: Dedup Guardrail ────────────────────────────────────────────
    "dedup_guardrail": Task(
        id="dedup_guardrail",
        name="Dedup Guardrail",
        difficulty="medium",
        seed=4,
        description=(
            "Dirty join keys and duplicate Dimension rows increase the risk of "
            "row explosion. The agent must prep keys, deduplicate Source B, "
            "produce a non-empty Silver table, and commit cleanly."
        ),
        success_criteria=[
            "PREP_KEYS_A and PREP_KEYS_B issued before join",
            "DEDUPLICATE_B issued before join",
            "No row explosion",
            "Silver table is non-empty",
            "Grader passes",
        ],
        scoring_rubric={
            "committed":           0.10,
            "prepped_before_join": 0.15,
            "deduped_before_join": 0.25,
            "no_explosion":        0.25,
            "silver_built":        0.10,
            "grader_pass":         0.15,
        },
    ),

    # ── HARD: Stale Sync Recovery ──────────────────────────────────────────
    "stale_sync_recovery": Task(
        id="stale_sync_recovery",
        name="Stale Sync Recovery",
        difficulty="hard",
        seed=5,
        description=(
            "Source A is stale and the pipeline must not proceed blindly. "
            "The agent must verify freshness, recover a high-match join, "
            "build Silver, and still pass the final audit."
        ),
        success_criteria=[
            "SYNC_CHECK issued before any join",
            "No row explosion",
            "match_rate > 0.80 after join",
            "Silver table is non-empty",
            "Grader passes",
        ],
        scoring_rubric={
            "committed":     0.05,
            "sync_checked":  0.30,
            "no_explosion":  0.20,
            "high_match":    0.15,
            "silver_built":  0.15,
            "grader_pass":   0.15,
        },
    ),

    # ── EASY: Fresh Join Baseline ──────────────────────────────────────────
    "fresh_join_baseline": Task(
        id="fresh_join_baseline",
        name="Fresh Join Baseline",
        difficulty="easy",
        seed=6,
        description=(
            "A clean baseline task that rewards a simple, efficient Bronze→Silver "
            "run. The agent should avoid unnecessary actions while producing a "
            "high-match, non-exploding join and a usable Silver table."
        ),
        success_criteria=[
            "COMMIT issued",
            "No row explosion",
            "match_rate > 0.80 after join",
            "Silver table is non-empty",
            "Episode completed efficiently",
        ],
        scoring_rubric={
            "committed":     0.15,
            "no_explosion":  0.25,
            "high_match":    0.25,
            "silver_built":  0.20,
            "efficient_run": 0.15,
        },
    ),

    # ── HARD: Stale History Guard ──────────────────────────────────────────
    "stale_history_guard": Task(
        id="stale_history_guard",
        name="Stale History Guard",
        difficulty="hard",
        seed=7,
        description=(
            "A stale-source episode where the agent must both verify freshness "
            "and preserve historical correctness. The task emphasizes SCD-2 "
            "usage and proper history columns in Silver."
        ),
        success_criteria=[
            "SYNC_CHECK issued before any join",
            "SCD-2 used instead of SCD-1",
            "Silver table is non-empty",
            "Silver contains history columns",
            "Grader passes",
        ],
        scoring_rubric={
            "committed":       0.05,
            "sync_checked":    0.20,
            "used_scd2":       0.25,
            "silver_built":    0.15,
            "history_columns": 0.15,
            "grader_pass":     0.20,
        },
    ),

    # ── MEDIUM: Orphan Quarantine ──────────────────────────────────────────
    "orphan_quarantine": Task(
        id="orphan_quarantine",
        name="Orphan Quarantine",
        difficulty="medium",
        seed=8,
        description=(
            "Dirty keys create unmatched Fact rows that should not be silently "
            "dropped. The agent must prep keys, choose a left join, preserve "
            "a meaningful quarantine set, and keep audit integrity intact."
        ),
        success_criteria=[
            "PREP_KEYS_A and PREP_KEYS_B issued before join",
            "Left join used",
            "Quarantine contains rows",
            "No row explosion",
            "Integrity checks pass",
        ],
        scoring_rubric={
            "committed":           0.10,
            "prepped_before_join": 0.15,
            "left_join_used":      0.20,
            "quarantine_nonempty": 0.20,
            "integrity_ok":        0.20,
            "no_explosion":        0.15,
        },
    ),

    # ── MEDIUM: Drift Alignment ────────────────────────────────────────────
    "drift_alignment": Task(
        id="drift_alignment",
        name="Drift Alignment",
        difficulty="medium",
        seed=9,
        description=(
            "Schema drift introduces new columns, but the pipeline is otherwise "
            "clean. The agent must evolve the schema, use the audited left-join "
            "path, materialize the new shape in Silver, and commit successfully."
        ),
        success_criteria=[
            "EVOLVE_SCHEMA issued before COMMIT",
            "Left join used",
            "Silver contains the joined columns after drift",
            "Silver table is non-empty",
            "Grader passes",
        ],
        scoring_rubric={
            "committed":           0.10,
            "schema_evolved":      0.25,
            "left_join_used":      0.15,
            "schema_materialized": 0.25,
            "silver_built":        0.10,
            "grader_pass":         0.15,
        },
    ),

    # ── EASY: Snapshot Upsert ──────────────────────────────────────────────
    "snapshot_upsert": Task(
        id="snapshot_upsert",
        name="Snapshot Upsert",
        difficulty="easy",
        seed=10,
        description=(
            "A clean snapshot-style load where SCD-1 is sufficient. The agent "
            "should choose overwrite semantics, maintain safe volume, and land "
            "a non-empty Silver table without introducing join problems."
        ),
        success_criteria=[
            "SCD-1 used instead of SCD-2",
            "No row explosion",
            "Silver row count ≤ Source A row count",
            "Silver table is non-empty",
            "Grader passes",
        ],
        scoring_rubric={
            "committed":     0.10,
            "no_explosion":  0.20,
            "used_scd1":     0.25,
            "volume_ok":     0.20,
            "silver_built":  0.15,
            "grader_pass":   0.10,
        },
    ),

    # ── HARD: Schema History Guard ─────────────────────────────────────────
    "schema_history_guard": Task(
        id="schema_history_guard",
        name="Schema History Guard",
        difficulty="hard",
        seed=11,
        description=(
            "Schema drift and historical tracking requirements arrive together. "
            "The agent must evolve schema, materialize the merged columns in "
            "Silver, use SCD-2, and preserve history metadata through commit."
        ),
        success_criteria=[
            "EVOLVE_SCHEMA issued before COMMIT",
            "SCD-2 used instead of SCD-1",
            "Silver contains the joined columns after drift",
            "Silver contains history columns",
            "Grader passes",
        ],
        scoring_rubric={
            "committed":           0.05,
            "schema_evolved":      0.20,
            "used_scd2":           0.20,
            "schema_materialized": 0.20,
            "history_columns":     0.15,
            "grader_pass":         0.20,
        },
    ),
}


def _build_checks(
    state: "MedusaState",
    tables: "Optional[_EpisodeTables]",
) -> Dict[str, bool]:
    """Build reusable boolean checks used across task rubrics."""
    silver_cols = set(tables.silver.columns) if tables is not None else set()
    joined_cols = set(tables.joined.columns) if tables is not None else set()

    return {
        "volume_ok": (
            state.silver_row_count <= state.source_a_row_count * 1.05
            and state.silver_row_count > 0
        ),
        "silver_built": state.silver_row_count > 0,
        "high_match": state.match_rate >= 0.80,
        "prepped_before_join": state.did_prep_a and state.did_prep_b and state.did_join,
        "deduped_before_join": state.did_dedup_b and state.did_join,
        "integrity_ok": state.grader_passed,
        "sync_checked": state.did_sync_check,
        "schema_evolved": state.did_evolve_schema,
        "used_scd2": state.scd_type == "SCD-2",
        "used_scd1": state.scd_type == "SCD-1",
        "schema_materialized": bool(joined_cols) and joined_cols.issubset(silver_cols),
        "history_columns": {"valid_from", "valid_to", "is_current"}.issubset(silver_cols),
        "left_join_used": state.join_type == "left",
        "quarantine_nonempty": state.quarantine_row_count > 0,
        "efficient_run": state.step_idx <= 8,
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
    ok = checks[key]
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
    """Score a completed MEDUSA episode against the named task.

    Args:
        task_id: Any task id defined in ``TASKS``.
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
    checks = _build_checks(state, tables)

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
        _apply_check(
            breakdown, rubric, checks, notes, "volume_ok",
            "Silver volume was empty or exceeded the allowed source volume.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "high_match",
            f"match_rate={state.match_rate:.1%} — target >80%.",
        )

    elif task_id == "dirty_integration":
        _apply_check(
            breakdown, rubric, checks, notes, "prepped_before_join",
            "Agent joined without prepping keys first.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "deduped_before_join",
            "Agent joined without deduplicating Dimension.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "integrity_ok",
            "Quarantine integrity did not pass the final audit.",
        )

    elif task_id == "full_medallion":
        _apply_check(
            breakdown, rubric, checks, notes, "sync_checked",
            "SYNC_CHECK was never called — stale source not verified.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "schema_evolved",
            "EVOLVE_SCHEMA never called — new columns may be missing from Silver.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "used_scd2",
            f"Used SCD-1 instead of SCD-2 (scd_type={state.scd_type!r}).",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "schema_ok",
            "Schema requirements did not pass the final audit.",
        )

    elif task_id == "schema_bootstrap":
        _apply_check(
            breakdown, rubric, checks, notes, "schema_evolved",
            "EVOLVE_SCHEMA was required before the schema-drift commit.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "schema_materialized",
            "Silver did not materialize the joined schema after drift.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )

    elif task_id == "dedup_guardrail":
        _apply_check(
            breakdown, rubric, checks, notes, "prepped_before_join",
            "Join executed before key preparation completed.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "deduped_before_join",
            "Dimension duplicates were not removed before join.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )

    elif task_id == "stale_sync_recovery":
        _apply_check(
            breakdown, rubric, checks, notes, "sync_checked",
            "SYNC_CHECK was required for the stale-source recovery task.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "high_match",
            f"match_rate={state.match_rate:.1%} — target >80% after stale recovery.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )

    elif task_id == "fresh_join_baseline":
        _apply_check(
            breakdown, rubric, checks, notes, "high_match",
            f"match_rate={state.match_rate:.1%} — target >80% for the clean baseline.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "efficient_run",
            f"Episode used {state.step_idx} steps — target is 8 or fewer.",
        )

    elif task_id == "stale_history_guard":
        _apply_check(
            breakdown, rubric, checks, notes, "sync_checked",
            "SYNC_CHECK was required before processing stale data.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "used_scd2",
            f"Used {state.scd_type!r} instead of SCD-2 for history preservation.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "history_columns",
            "Silver is missing the SCD-2 history columns.",
        )

    elif task_id == "orphan_quarantine":
        _apply_check(
            breakdown, rubric, checks, notes, "prepped_before_join",
            "Join executed before key preparation completed.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "left_join_used",
            f"Expected LEFT join for orphan quarantine, got {state.join_type!r}.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "quarantine_nonempty",
            "Expected unresolved rows to appear in quarantine.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "integrity_ok",
            "Quarantine integrity did not pass the final audit.",
        )

    elif task_id == "drift_alignment":
        _apply_check(
            breakdown, rubric, checks, notes, "schema_evolved",
            "EVOLVE_SCHEMA was required before the schema-drift commit.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "left_join_used",
            f"Expected LEFT join for the audited drift path, got {state.join_type!r}.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "schema_materialized",
            "Silver did not materialize the joined schema after drift.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )

    elif task_id == "snapshot_upsert":
        _apply_check(
            breakdown, rubric, checks, notes, "used_scd1",
            f"Used {state.scd_type!r} instead of SCD-1 for the snapshot upsert task.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "volume_ok",
            "Silver volume was empty or exceeded the allowed source volume.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "silver_built",
            "Silver table was never populated.",
        )

    elif task_id == "schema_history_guard":
        _apply_check(
            breakdown, rubric, checks, notes, "schema_evolved",
            "EVOLVE_SCHEMA was required before the schema-history commit.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "used_scd2",
            f"Used {state.scd_type!r} instead of SCD-2 for history preservation.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "schema_materialized",
            "Silver did not materialize the joined schema after drift.",
        )
        _apply_check(
            breakdown, rubric, checks, notes, "history_columns",
            "Silver is missing the SCD-2 history columns.",
        )

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
