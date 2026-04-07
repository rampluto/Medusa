"""MEDUSA — full environment implementation.

Replaces the Phase-1 skeleton with a complete reset/step pipeline that:
  • Generates Bronze A/B data from ``ScenarioGenerator``
  • Dispatches each action to the appropriate operator
  • Computes per-step rewards via ``RewardEngine``
  • Runs the deterministic grader on COMMIT
  • Builds a 16-float normalized feature vector for the RL agent
  • Maintains a governance log of every decision
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from .grader import Grader
    from .models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
    from .operators import (
        apply_scd,
        deduplicate,
        evolve_schema,
        execute_join,
        prep_keys,
        sync_check,
    )
    from .rewards import RewardEngine
    from .scenarios import Scenario, ScenarioGenerator
    from .tasks import TASKS, score_episode
except ImportError:
    from grader import Grader
    from models import MedusaAction, MedusaActionType, MedusaObservation, MedusaState
    from operators import (
        apply_scd,
        deduplicate,
        evolve_schema,
        execute_join,
        prep_keys,
        sync_check,
    )
    from rewards import RewardEngine
    from scenarios import Scenario, ScenarioGenerator
    from tasks import TASKS, score_episode


# ---------------------------------------------------------------------------
# Internal episode tables
# ---------------------------------------------------------------------------

@dataclass
class _EpisodeTables:
    """In-memory tables for one episode."""

    bronze_a: pd.DataFrame = field(default_factory=pd.DataFrame)
    bronze_a_prepped: pd.DataFrame = field(default_factory=pd.DataFrame)
    bronze_b: pd.DataFrame = field(default_factory=pd.DataFrame)
    bronze_b_prepped: pd.DataFrame = field(default_factory=pd.DataFrame)
    joined: pd.DataFrame = field(default_factory=pd.DataFrame)
    silver: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarantine: pd.DataFrame = field(default_factory=pd.DataFrame)
    governance_log: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

_MAX_TIME_DELTA = 48.0   # Normalisation ceiling (hours)
_MAX_COLS = 10.0          # Normalisation ceiling (new columns)


def _build_features(state: MedusaState) -> List[float]:
    """Build the 16-float normalised observation vector."""
    return [
        min(state.time_delta_a / _MAX_TIME_DELTA, 1.0),
        min(state.time_delta_b / _MAX_TIME_DELTA, 1.0),
        float(state.is_stale_a),
        float(state.is_stale_b),
        state.null_ratio_key_a,
        state.null_ratio_key_b,
        state.uniqueness_a,
        state.uniqueness_b,
        state.match_rate,
        min(state.new_cols_a / _MAX_COLS, 1.0),
        min(state.new_cols_b / _MAX_COLS, 1.0),
        state.schema_compat,
        float(state.did_prep_a),
        float(state.did_prep_b),
        float(state.did_dedup_b),
        min(state.step_idx / max(state.max_steps, 1), 1.0),
    ]


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class MedusaEnv(Environment[MedusaAction, MedusaObservation, MedusaState]):
    """MEDUSA: Medallion-Engineered Deterministic Unified Storage Agent.

    Simulates a Bronze→Silver data integration pipeline. The agent observes
    data quality signals and chooses ETL actions to produce a correct,
    historically consistent Silver entity.

    Args:
        scenario_seed: Fixed seed for deterministic episodes. ``None`` = random.
        max_steps: Maximum steps per episode before forced termination.
        stale_threshold_hours: Age (hours) at which a source is deemed stale.
        n_fact_rows: Size of the Fact / Source A table.
        n_dim_rows: Size of the Dimension / Source B table.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        scenario_seed: Optional[int] = None,
        max_steps: int = 20,
        stale_threshold_hours: float = 6.0,
        n_fact_rows: int = 200,
        n_dim_rows: int = 150,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._scenario_seed = scenario_seed
        self._max_steps = max_steps
        self._stale_threshold = stale_threshold_hours

        self._generator = ScenarioGenerator(
            n_fact_rows=n_fact_rows, n_dim_rows=n_dim_rows
        )
        self._reward_engine = RewardEngine()
        self._grader = Grader()

        self._state = MedusaState()
        self._tables = _EpisodeTables()
        self._scenario: Optional[Scenario] = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="medusa_env",
            description=(
                "MEDUSA: simulated Bronze→Silver integration controller for "
                "multi-source joins, schema drift, and SCD merges."
            ),
            version="0.2.0",
            documentation="envs/medusa_env/README.md",
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> MedusaState:
        return self._state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MedusaObservation:
        self._reset_rubric()

        effective_seed = seed if seed is not None else self._scenario_seed
        run_id = episode_id or str(uuid.uuid4())

        # Generate scenario
        self._scenario = self._generator.generate(seed=effective_seed)
        scen = self._scenario

        # Initialise tables
        self._tables = _EpisodeTables(
            bronze_a=scen.bronze_a.copy(),
            bronze_a_prepped=scen.bronze_a.copy(),
            bronze_b=scen.bronze_b.copy(),
            bronze_b_prepped=scen.bronze_b.copy(),
        )

        # Compute initial key health metrics from raw Bronze
        na_a = scen.bronze_a[scen.join_key].isna().sum()
        na_b = scen.bronze_b[scen.join_key].isna().sum()
        null_ratio_a = na_a / max(len(scen.bronze_a), 1)
        null_ratio_b = na_b / max(len(scen.bronze_b), 1)

        # Uniqueness of raw keys
        nna_a = scen.bronze_a[scen.join_key].dropna()
        nna_b = scen.bronze_b[scen.join_key].dropna()
        uniq_a = nna_a.nunique() / max(len(nna_a), 1)
        uniq_b = nna_b.nunique() / max(len(nna_b), 1)

        # Match rate on raw keys
        keys_a = set(nna_a.astype(str))
        keys_b = set(nna_b.astype(str))
        match_rate = len(keys_a & keys_b) / max(len(keys_a), 1)

        self._state = MedusaState(
            run_id=run_id,
            seed=effective_seed,
            scenario_id=scen.id,
            max_steps=self._max_steps,
            step_idx=0,
            stage="running",
            time_delta_a=scen.time_delta_a,
            time_delta_b=scen.time_delta_b,
            is_stale_a=scen.is_stale_a,
            is_stale_b=scen.is_stale_b,
            null_ratio_key_a=float(null_ratio_a),
            null_ratio_key_b=float(null_ratio_b),
            uniqueness_a=float(uniq_a),
            uniqueness_b=float(uniq_b),
            match_rate=float(match_rate),
            new_cols_a=len(scen.new_cols_a),
            new_cols_b=len(scen.new_cols_b),
            source_a_row_count=len(scen.bronze_a),
        )

        features = _build_features(self._state)
        obs = MedusaObservation(
            message=(
                f"MEDUSA episode started. Scenario: {scen.id}. "
                f"{scen.description} "
                f"Source A: {len(scen.bronze_a)} rows | "
                f"Source B: {len(scen.bronze_b)} rows."
            ),
            features=features,
            metrics={
                "scenario_id": scen.id,
                "null_ratio_key_a": null_ratio_a,
                "null_ratio_key_b": null_ratio_b,
                "match_rate": match_rate,
                "is_stale_a": scen.is_stale_a,
                "is_stale_b": scen.is_stale_b,
                "new_cols_a": scen.new_cols_a,
                "new_cols_b": scen.new_cols_b,
            },
            metadata={"run_id": run_id, "seed": effective_seed},
            reward=None,
            done=False,
        )
        return self._apply_transform(obs)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        action: MedusaAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MedusaObservation:
        if self._state.stage != "running":
            return self._apply_transform(MedusaObservation(
                message=f"Episode not running (stage={self._state.stage}). Call reset().",
                done=True,
                reward=0.0,
                features=_build_features(self._state),
                metadata={"run_id": self._state.run_id},
            ))

        # Snapshot state *before* applying action (for reward evaluation)
        state_before = copy.copy(self._state)
        self._state.step_idx += 1

        action_type = action.action
        metrics: dict = {}
        step_message = ""

        scen = self._scenario
        assert scen is not None, "reset() must be called before step()"

        # ── Dispatch ──────────────────────────────────────────────────
        try:
            if action_type == MedusaActionType.SYNC_CHECK:
                _, metrics = sync_check(
                    self._tables.bronze_a,
                    self._tables.bronze_b,
                    scen.time_delta_a,
                    scen.time_delta_b,
                    self._stale_threshold,
                )
                self._state.did_sync_check = True
                step_message = (
                    f"SYNC_CHECK: A={scen.time_delta_a:.1f}h "
                    f"{'[STALE]' if scen.is_stale_a else '[FRESH]'} | "
                    f"B={scen.time_delta_b:.1f}h "
                    f"{'[STALE]' if scen.is_stale_b else '[FRESH]'}"
                )

            elif action_type == MedusaActionType.EVOLVE_SCHEMA:
                result_df, metrics = evolve_schema(
                    self._tables.silver,
                    self._tables.bronze_a,
                    self._tables.bronze_b,
                    scen.new_cols_a,
                    scen.new_cols_b,
                )
                if result_df is not None:
                    self._tables.silver = result_df
                self._state.did_evolve_schema = True
                step_message = f"EVOLVE_SCHEMA: added {metrics.get('new_cols_count', 0)} column(s)."

            elif action_type == MedusaActionType.PREP_KEYS_A:
                result_df, metrics = prep_keys(
                    self._tables.bronze_a_prepped, scen.join_key
                )
                if result_df is not None:
                    self._tables.bronze_a_prepped = result_df
                self._state.did_prep_a = True
                self._state.null_ratio_key_a = float(metrics.get("null_ratio_after", 0.0))
                step_message = (
                    f"PREP_KEYS_A: null ratio {metrics.get('null_ratio_before', 0):.2%}"
                    f"→{metrics.get('null_ratio_after', 0):.2%}."
                )

            elif action_type == MedusaActionType.PREP_KEYS_B:
                result_df, metrics = prep_keys(
                    self._tables.bronze_b_prepped, scen.join_key
                )
                if result_df is not None:
                    self._tables.bronze_b_prepped = result_df
                self._state.did_prep_b = True
                self._state.null_ratio_key_b = float(metrics.get("null_ratio_after", 0.0))
                step_message = (
                    f"PREP_KEYS_B: null ratio {metrics.get('null_ratio_before', 0):.2%}"
                    f"→{metrics.get('null_ratio_after', 0):.2%}."
                )

            elif action_type == MedusaActionType.DEDUPLICATE_B:
                result_df, metrics = deduplicate(
                    self._tables.bronze_b_prepped, scen.join_key
                )
                if result_df is not None:
                    self._tables.bronze_b_prepped = result_df
                self._state.did_dedup_b = True
                self._state.uniqueness_b = float(metrics.get("uniqueness", 1.0))
                step_message = f"DEDUPLICATE_B: removed {metrics.get('dupes_removed', 0)} duplicate(s)."

            elif action_type in {
                MedusaActionType.EXECUTE_JOIN_INNER,
                MedusaActionType.EXECUTE_JOIN_LEFT,
                MedusaActionType.EXECUTE_JOIN_ANTI,
            }:
                join_map = {
                    MedusaActionType.EXECUTE_JOIN_INNER: "inner",
                    MedusaActionType.EXECUTE_JOIN_LEFT: "left",
                    MedusaActionType.EXECUTE_JOIN_ANTI: "anti",
                }
                join_type_str = join_map[action_type]
                joined, quarantine, metrics = execute_join(
                    self._tables.bronze_a_prepped,
                    self._tables.bronze_b_prepped,
                    scen.join_key,
                    join_type_str,
                )
                self._tables.joined = joined
                self._tables.quarantine = quarantine
                self._state.did_join = True
                self._state.join_type = join_type_str
                self._state.join_row_count = int(metrics.get("join_rows", 0))
                self._state.explosion_detected = bool(metrics.get("explosion_detected", False))
                self._state.match_rate = float(metrics.get("match_rate", 0.0))
                self._state.quarantine_row_count = len(quarantine)
                step_message = (
                    f"EXECUTE_JOIN ({join_type_str.upper()}): "
                    f"{self._state.join_row_count} rows | "
                    f"match_rate={self._state.match_rate:.1%} | "
                    f"quarantine={self._state.quarantine_row_count} | "
                    f"{'⚠ EXPLOSION' if self._state.explosion_detected else 'OK'}"
                )

            elif action_type in {MedusaActionType.APPLY_SCD_1, MedusaActionType.APPLY_SCD_2}:
                scd_type_int = 1 if action_type == MedusaActionType.APPLY_SCD_1 else 2
                tracked_col = scen.tracked_cols[0] if scen.tracked_cols else scen.join_key
                result_df, metrics = apply_scd(
                    self._tables.silver,
                    self._tables.joined,
                    scen.join_key,
                    tracked_col,
                    scd_type_int,
                )
                if result_df is not None:
                    self._tables.silver = result_df
                self._state.did_scd = True
                self._state.scd_type = f"SCD-{scd_type_int}"
                self._state.scd_inserts = int(metrics.get("inserts", 0))
                self._state.scd_updates = int(metrics.get("updates", 0))
                self._state.silver_row_count = int(metrics.get("silver_rows", 0))
                step_message = (
                    f"APPLY_SCD-{scd_type_int}: "
                    f"{self._state.scd_inserts} inserts, "
                    f"{self._state.scd_updates} updates → "
                    f"Silver {self._state.silver_row_count} rows."
                )

            elif action_type == MedusaActionType.COMMIT:
                return self._do_commit(state_before)

        except Exception as exc:  # noqa: BLE001
            step_message = f"ERROR in {action_type}: {exc}"
            metrics = {"error": str(exc)}

        # ── Reward ────────────────────────────────────────────────────
        reward = self._reward_engine.evaluate(
            action_type=action_type.value,
            metrics=metrics,
            state_before=state_before,
        )
        self._state.cumulative_reward += reward

        # ── Governance log ────────────────────────────────────────────
        self._tables.governance_log.append({
            "step": self._state.step_idx,
            "action": action_type.value,
            "reward": reward,
            "cumulative_reward": self._state.cumulative_reward,
            "metrics": metrics,
            "timestamp": time.time(),
        })

        # Check step limit
        done = self._state.step_idx >= self._state.max_steps
        if done:
            self._state.stage = "failed"
            step_message += " [MAX STEPS REACHED]"

        features = _build_features(self._state)
        obs = MedusaObservation(
            message=step_message,
            features=features,
            metrics=metrics,
            metadata={
                "run_id": self._state.run_id,
                "step": self._state.step_idx,
                "cumulative_reward": self._state.cumulative_reward,
            },
            reward=reward,
            done=done,
        )
        return self._apply_transform(obs)

    # ------------------------------------------------------------------
    # Commit (terminal step)
    # ------------------------------------------------------------------

    def _do_commit(self, state_before: MedusaState) -> MedusaObservation:
        """Run grader then finalise the episode."""
        scen = self._scenario
        assert scen is not None

        # Base step reward
        reward = self._reward_engine.evaluate(
            action_type=MedusaActionType.COMMIT.value,
            metrics={},
            state_before=state_before,
        )

        # Grader audit
        grader_result = self._grader.audit(
            silver=self._tables.silver,
            quarantine=self._tables.quarantine,
            bronze_a=scen.bronze_a,
            bronze_b=scen.bronze_b,
            join_key=scen.join_key,
            join_type=self._state.join_type or "left",
            scd_type=int(self._state.scd_type[-1]) if self._state.scd_type else 1,
            scenario=scen,
        )
        reward += grader_result.bonus_reward
        self._state.grader_passed = grader_result.passed
        self._state.grader_report = grader_result.report
        self._state.cumulative_reward += reward
        self._state.silver_row_count = len(self._tables.silver)
        self._state.quarantine_row_count = len(self._tables.quarantine)
        self._state.stage = "committed"

        self._tables.governance_log.append({
            "step": self._state.step_idx,
            "action": "COMMIT",
            "reward": reward,
            "cumulative_reward": self._state.cumulative_reward,
            "grader_passed": grader_result.passed,
            "grader_report": grader_result.report,
            "timestamp": time.time(),
        })

        # Map the current episode seed to the task definitions to get the explicit task_id
        task_id = next((tid for tid, t in TASKS.items() if t.seed == self._state.seed), "clean_pipeline")
        
        # Calculate the final [0, 1] evaluation score for this episode
        final_result = score_episode(task_id, self._state, self._tables)
        final_score = final_result.score

        features = _build_features(self._state)
        obs = MedusaObservation(
            message=(
                f"COMMIT: episode finalized. "
                f"{'Grader: PASS ✓' if grader_result.passed else 'Grader: FAIL ✗'} "
                f"Bonus: {grader_result.bonus_reward:+.1f} | "
                f"Total reward: {self._state.cumulative_reward:.1f} | "
                f"Final Score: {final_score:.3f}"
            ),
            features=features,
            metrics={
                "grader_passed": grader_result.passed,
                "grader_report": grader_result.report,
                "silver_rows": self._state.silver_row_count,
                "quarantine_rows": self._state.quarantine_row_count,
                "governance_log_entries": len(self._tables.governance_log),
                "score": final_score,
            },
            metadata={
                "run_id": self._state.run_id,
                "steps": self._state.step_idx,
                "cumulative_reward": self._state.cumulative_reward,
                "score": final_score,
            },
            reward=reward,
            done=True,
        )
        return self._apply_transform(obs)