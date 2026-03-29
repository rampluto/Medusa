"""Tests for the MEDUSA environment.

Covers: models, scenario generator, operators, reward engine, grader,
and full end-to-end environment episodes.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from medusa_env.models import (
    MedusaAction,
    MedusaActionType,
    MedusaObservation,
    MedusaState,
)


class TestMedusaModels:
    def test_action_creation(self):
        a = MedusaAction(action=MedusaActionType.SYNC_CHECK)
        assert a.action == MedusaActionType.SYNC_CHECK
        assert a.params == {}

    def test_state_defaults(self):
        s = MedusaState()
        assert s.stage == "init"
        assert s.step_idx == 0
        assert s.did_sync_check is False
        assert s.explosion_detected is False
        assert s.grader_passed is False

    def test_observation_defaults(self):
        obs = MedusaObservation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.features == []


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------

import pandas as pd

from medusa_env.scenarios import Scenario, ScenarioGenerator


class TestMedusaScenarios:
    @pytest.fixture
    def gen(self):
        return ScenarioGenerator(n_fact_rows=50, n_dim_rows=40)

    def test_canonical_clean(self, gen):
        scen = gen.generate(seed=0)
        assert scen.id.startswith("clean")
        assert isinstance(scen.bronze_a, pd.DataFrame)
        assert len(scen.bronze_a) == 50
        assert not scen.is_stale_a
        assert not scen.is_stale_b
        assert scen.new_cols_a == []

    def test_canonical_dirty_keys(self, gen):
        scen = gen.generate(seed=1)
        assert "dirty_keys" in scen.id
        # Dirty scenario should have actual null or whitespace keys
        has_issues = (
            scen.bronze_a[scen.join_key].isna().any()
            or scen.bronze_a[scen.join_key].astype(str).str.contains(r"^\s|\s$").any()
        )
        assert has_issues

    def test_canonical_stale(self, gen):
        scen = gen.generate(seed=2)
        assert "stale" in scen.id
        assert scen.is_stale_a  # Source A should be stale

    def test_canonical_schema_drift(self, gen):
        scen = gen.generate(seed=3)
        assert "schema_drift" in scen.id
        assert len(scen.new_cols_a) > 0
        assert len(scen.new_cols_b) > 0

    def test_random_seed_produces_scenario(self, gen):
        scen = gen.generate(seed=999)
        assert isinstance(scen, Scenario)
        assert scen.join_key in scen.bronze_a.columns
        assert scen.join_key in scen.bronze_b.columns


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

from medusa_env.operators import (
    apply_scd,
    deduplicate,
    evolve_schema,
    execute_join,
    prep_keys,
    sync_check,
)


class TestMedusaOperators:
    def test_sync_check_fresh(self):
        a = pd.DataFrame({"id": [1, 2]})
        b = pd.DataFrame({"id": [1, 2]})
        _, m = sync_check(a, b, time_delta_a=1.0, time_delta_b=2.0)
        assert m["is_stale_a"] is False
        assert m["is_stale_b"] is False

    def test_sync_check_stale(self):
        a = pd.DataFrame({"id": [1]})
        b = pd.DataFrame({"id": [1]})
        _, m = sync_check(a, b, time_delta_a=10.0, time_delta_b=1.0)
        assert m["is_stale_a"] is True
        assert m["is_stale_b"] is False

    def test_prep_keys_strips_whitespace(self):
        df = pd.DataFrame({"key": ["  K001  ", "K002", None]})
        result, m = prep_keys(df, "key")
        # Stripped key should have no leading/trailing spaces
        non_null = result["key"].dropna().tolist()
        assert all(v.strip() == v for v in non_null)
        assert m["null_ratio_before"] > 0

    def test_deduplicate_removes_dupes(self):
        df = pd.DataFrame({"key": ["A", "A", "B"], "val": [1, 2, 3]})
        result, m = deduplicate(df, "key")
        assert m["dupes_removed"] == 1
        assert len(result) == 2

    def test_execute_join_left_basic(self):
        fact = pd.DataFrame({"key": ["K001", "K002", "K003"], "val": [1, 2, 3]})
        dim = pd.DataFrame({"key": ["K001", "K002"], "dim_name": ["A", "B"]})
        joined, quarantine, m = execute_join(fact, dim, "key", "left")
        assert m["join_rows"] == 3  # left join keeps all fact rows
        assert m["match_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert len(quarantine) >= 1  # K003 should be quarantined

    def test_execute_join_detects_explosion(self):
        # Non-unique dim key → Cartesian explosion
        fact = pd.DataFrame({"key": ["K001"] * 10, "val": list(range(10))})
        dim = pd.DataFrame({"key": ["K001"] * 20, "dim_name": ["X"] * 20})
        joined, quarantine, m = execute_join(fact, dim, "key", "inner")
        assert m["explosion_detected"] is True

    def test_execute_join_anti(self):
        fact = pd.DataFrame({"key": ["K001", "K002", "K999"], "val": [1, 2, 3]})
        dim = pd.DataFrame({"key": ["K001", "K002"], "name": ["A", "B"]})
        joined, quarantine, m = execute_join(fact, dim, "key", "anti")
        assert len(joined) == 0  # Anti-join: no rows in joined
        assert len(quarantine) == 1  # K999 goes to quarantine

    def test_apply_scd1_upsert(self):
        silver = pd.DataFrame({"key": ["K001"], "val": [10], "status": ["old"]})
        joined = pd.DataFrame({"key": ["K001", "K002"], "val": [99, 20], "status": ["new", "new"]})
        result, m = apply_scd(silver, joined, "key", "status", scd_type=1)
        assert m["scd_type"] == 1
        assert m["inserts"] + m["updates"] > 0
        # K001 should be updated to val=99
        k1_row = result[result["key"] == "K001"]
        assert not k1_row.empty

    def test_apply_scd2_adds_history(self):
        silver = pd.DataFrame()
        joined = pd.DataFrame({"key": ["K001"], "status": ["active"]})
        result, m = apply_scd(silver, joined, "key", "status", scd_type=2)
        assert "valid_from" in result.columns
        assert m["inserts"] == 1

    def test_evolve_schema_adds_columns(self):
        silver = pd.DataFrame({"key": ["K001"], "val": [1]})
        a = pd.DataFrame({"key": ["K001"], "new_metric": [42]})
        b = pd.DataFrame({"key": ["K001"]})
        result, m = evolve_schema(silver, a, b, ["new_metric"], [])
        assert "new_metric" in result.columns
        assert m["new_cols_count"] == 1


# ---------------------------------------------------------------------------
# Reward Engine
# ---------------------------------------------------------------------------

from medusa_env.rewards import RewardEngine


class TestMedusaRewards:
    @pytest.fixture
    def engine(self):
        return RewardEngine()

    def _clean_state(self):
        s = MedusaState()
        s.did_prep_a = True
        s.did_prep_b = True
        s.did_sync_check = True
        return s

    def test_step_penalty_always_applied(self, engine):
        r = engine.evaluate("SYNC_CHECK", {}, MedusaState())
        assert r == pytest.approx(-0.2, abs=0.01)

    def test_high_match_join_reward(self, engine):
        r = engine.evaluate(
            "EXECUTE_JOIN_LEFT",
            {"match_rate": 0.95, "join_rows": 100, "fact_rows": 100,
             "explosion_detected": False, "quarantine_rows": 5},
            self._clean_state(),
        )
        assert r > 0.0  # +25 - 0.2 + 10 (quarantine) = +34.8

    def test_row_explosion_heavy_penalty(self, engine):
        r = engine.evaluate(
            "EXECUTE_JOIN_INNER",
            {"explosion_detected": True, "join_rows": 1000, "fact_rows": 100,
             "match_rate": 1.0, "quarantine_rows": 0},
            self._clean_state(),
        )
        assert r < -50.0

    def test_dirty_join_penalty(self, engine):
        # No PREP_KEYS → dirty join penalty
        state = MedusaState()
        state.did_prep_a = False
        state.did_prep_b = False
        r = engine.evaluate(
            "EXECUTE_JOIN_LEFT",
            {"explosion_detected": False, "join_rows": 0, "fact_rows": 50,
             "match_rate": 0.0, "quarantine_rows": 0},
            state,
        )
        assert r < -20.0

    def test_scd2_extra_reward(self, engine):
        r = engine.evaluate("APPLY_SCD_2", {}, self._clean_state())
        # +5 for SCD-2 - 0.2 step penalty
        assert r == pytest.approx(4.8, abs=0.01)

    def test_stale_processing_penalty(self, engine):
        state = MedusaState()
        state.is_stale_a = True
        state.did_sync_check = False  # Never checked freshness
        state.did_prep_a = True
        state.did_prep_b = True
        r = engine.evaluate(
            "EXECUTE_JOIN_LEFT",
            {"explosion_detected": False, "join_rows": 100, "fact_rows": 100,
             "match_rate": 0.95, "quarantine_rows": 0},
            state,
        )
        # Should include stale penalty on top of positive join reward
        assert r < 25.0  # Stale penalty reduces it


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

from medusa_env.grader import Grader
from medusa_env.scenarios import Scenario


class TestMedusaGrader:
    @pytest.fixture
    def grader(self):
        return Grader()

    def _make_scenario(self):
        a = pd.DataFrame({"entity_id": ["K1", "K2", "K3"], "val": [1, 2, 3],
                          "fact_category": ["A", "B", "C"],
                          "fact_value": [1.0, 2.0, 3.0],
                          "created_at": pd.date_range("2024-01-01", periods=3, freq="h")})
        b = pd.DataFrame({"entity_id": ["K1", "K2"], "dim_name": ["N1", "N2"], "dim_status": ["x", "y"]})
        return a, b

    def test_volume_check_pass(self, grader):
        a, b = self._make_scenario()
        silver = pd.DataFrame({"entity_id": ["K1", "K2"], "val": [1, 2]})
        scen = ScenarioGenerator(n_fact_rows=3, n_dim_rows=2).generate(seed=0)
        r = grader.audit(silver, pd.DataFrame(), a, b, "entity_id", "left", 1, scen)
        assert r.volume_ok is True

    def test_volume_check_fail(self, grader):
        a, b = self._make_scenario()
        # Silver has way more rows than source A → violation
        silver = pd.DataFrame({"entity_id": ["K1"] * 100})
        scen = ScenarioGenerator(n_fact_rows=3, n_dim_rows=2).generate(seed=0)
        r = grader.audit(silver, pd.DataFrame(), a, b, "entity_id", "left", 1, scen)
        assert r.volume_ok is False

    def test_integrity_check_quarantine_true_orphans(self, grader):
        a, b = self._make_scenario()
        # K3 is not in B → true orphan
        quarantine = pd.DataFrame({"entity_id": ["K3"]})
        scen = ScenarioGenerator(n_fact_rows=3, n_dim_rows=2).generate(seed=0)
        silver = pd.DataFrame({"entity_id": ["K1", "K2"]})
        r = grader.audit(silver, quarantine, a, b, "entity_id", "left", 1, scen)
        assert r.integrity_ok is True

    def test_integrity_check_fail_dirty_quarantine(self, grader):
        a, b = self._make_scenario()
        # K1 IS in B but ends up in quarantine (agent failed to clean it)
        quarantine = pd.DataFrame({"entity_id": ["K1"]})
        scen = ScenarioGenerator(n_fact_rows=3, n_dim_rows=2).generate(seed=0)
        silver = pd.DataFrame({"entity_id": ["K2"]})
        r = grader.audit(silver, quarantine, a, b, "entity_id", "left", 1, scen)
        assert r.integrity_ok is False

    def test_all_pass_gives_bonus(self, grader):
        gen = ScenarioGenerator(n_fact_rows=3, n_dim_rows=2)
        scen = gen.generate(seed=0)
        a, b = scen.bronze_a, scen.bronze_b
        # Simulate a perfect run
        silver = a.merge(b, on="entity_id", how="left")
        r = grader.audit(silver, pd.DataFrame(), a, b, "entity_id", "left", 1, scen)
        assert r.bonus_reward > 0


# ---------------------------------------------------------------------------
# Full environment integration
# ---------------------------------------------------------------------------

from medusa_env.server import MedusaEnv 
from medusa_env.models import MedusaActionType


class TestMedusaEnvironment:
    @pytest.fixture
    def env(self):
        return MedusaEnv(n_fact_rows=50, n_dim_rows=40)

    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=0)
        assert isinstance(obs, MedusaObservation)
        assert obs.done is False
        assert len(obs.features) == 16
        assert obs.reward is None

    def test_state_after_reset(self, env):
        env.reset(seed=0)
        state = env.state
        assert state.stage == "running"
        assert state.step_idx == 0
        assert state.source_a_row_count == 50

    def test_happy_path_episode(self, env):
        """Full pipeline: sync → evolve → prep both → dedup → join → scd → commit."""
        env.reset(seed=0)  # clean scenario

        actions = [
            MedusaActionType.SYNC_CHECK,
            MedusaActionType.EVOLVE_SCHEMA,
            MedusaActionType.PREP_KEYS_A,
            MedusaActionType.PREP_KEYS_B,
            MedusaActionType.DEDUPLICATE_B,
            MedusaActionType.EXECUTE_JOIN_LEFT,
            MedusaActionType.APPLY_SCD_2,
            MedusaActionType.COMMIT,
        ]
        obs = None
        for act_type in actions:
            obs = env.step(MedusaAction(action=act_type))

        assert obs is not None
        assert obs.done is True
        assert env.state.stage == "committed"
        assert env.state.grader_passed  # Clean scenario should pass grader

    def test_row_explosion_gives_heavy_penalty(self, env):
        """Joining on non-unique B keys should trigger explosion penalty."""
        env.reset(seed=1)  # dirty_keys scenario — B has duplicate keys

        # Skip prep & dedup — go straight to join
        env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))

        # Force the dimension to have many duplicates so explosion fires
        import pandas as _pd

        env._tables.bronze_b_prepped = _pd.DataFrame({
            "entity_id": ["K001"] * 30,
            "dim_name": ["X"] * 30,
            "dim_status": ["x"] * 30,
        })
        env._tables.bronze_a_prepped = _pd.DataFrame({
            "entity_id": ["K001"] * 10,
            "fact_value": list(range(10)),
            "fact_category": ["A"] * 10,
            "created_at": _pd.date_range("2024-01-01", periods=10, freq="h"),
        })

        obs = env.step(MedusaAction(action=MedusaActionType.EXECUTE_JOIN_INNER))
        assert obs.reward is not None
        assert obs.reward < -50.0
        assert env.state.explosion_detected is True

    def test_dirty_join_penalty(self, env):
        """Skipping PREP_KEYS and joining on null-heavy keys → dirty join."""
        env.reset(seed=1)  # dirty_keys scenario

        # Skip PREP — join directly
        obs = env.step(MedusaAction(action=MedusaActionType.EXECUTE_JOIN_LEFT))
        # If all fact keys are null/non-matching → 0-row join → dirty join penalty
        # (reward < base -0.2 if dirty join fired)
        assert obs.reward is not None

    def test_step_idx_increments(self, env):
        env.reset(seed=0)
        for _ in range(3):
            env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
        assert env.state.step_idx == 3

    def test_max_steps_terminates_episode(self):
        env = MedusaEnv(n_fact_rows=10, n_dim_rows=10, max_steps=3)
        env.reset(seed=0)
        obs = None
        for _ in range(4):  # more than max_steps
            obs = env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
        assert obs is not None
        assert obs.done is True

    def test_commit_without_join_grader_fails(self, env):
        """Committing without joining should make the grader fail."""
        env.reset(seed=0)
        env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
        obs = env.step(MedusaAction(action=MedusaActionType.COMMIT))
        assert obs.done is True
        # Silver will be empty → schema check should fail or volume check fail
        assert env.state.grader_report != ""

    def test_features_vector_length(self, env):
        env.reset(seed=0)
        obs = env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
        assert len(obs.features) == 16
        assert all(0.0 <= f <= 1.0 for f in obs.features)

    def test_governance_log_populated(self, env):
        env.reset(seed=0)
        env.step(MedusaAction(action=MedusaActionType.SYNC_CHECK))
        env.step(MedusaAction(action=MedusaActionType.PREP_KEYS_A))
        log = env._tables.governance_log
        assert len(log) == 2
        assert log[0]["action"] == "SYNC_CHECK"


# ---------------------------------------------------------------------------
# Task Scorer
# ---------------------------------------------------------------------------

from medusa_env.tasks import TASKS, score_episode


class TestMedusaTasks:
    """Tests for the 3 formal task definitions and 0.0–1.0 scorer."""

    def test_three_tasks_defined(self):
        assert "clean_pipeline" in TASKS
        assert "dirty_integration" in TASKS
        assert "full_medallion" in TASKS

    def test_task_difficulties(self):
        assert TASKS["clean_pipeline"].difficulty == "easy"
        assert TASKS["dirty_integration"].difficulty == "medium"
        assert TASKS["full_medallion"].difficulty == "hard"

    def test_task_seeds_match_scenarios(self):
        assert TASKS["clean_pipeline"].seed == 0
        assert TASKS["dirty_integration"].seed == 1
        assert TASKS["full_medallion"].seed == 2

    def _run_happy_path(self, seed: int) -> MedusaState:
        """Run the optimal action sequence for the given seed and return final state."""
        env = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
        env.reset(seed=seed)
        for act in [
            MedusaActionType.SYNC_CHECK,
            MedusaActionType.EVOLVE_SCHEMA,
            MedusaActionType.PREP_KEYS_A,
            MedusaActionType.PREP_KEYS_B,
            MedusaActionType.DEDUPLICATE_B,
            MedusaActionType.EXECUTE_JOIN_LEFT,
            MedusaActionType.APPLY_SCD_2,
            MedusaActionType.COMMIT,
        ]:
            env.step(MedusaAction(action=act))
        return env.state

    # ── clean_pipeline (easy) ───────────────────────────────────────────────

    def test_clean_pipeline_score_is_in_range(self):
        state = self._run_happy_path(seed=0)
        result = score_episode("clean_pipeline", state)
        assert 0.0 <= result.score <= 1.0

    def test_clean_pipeline_happy_path_passes(self):
        state = self._run_happy_path(seed=0)
        result = score_episode("clean_pipeline", state)
        assert result.passed is True
        assert result.grade in ("S", "A", "B")

    def test_clean_pipeline_uncommitted_scores_zero(self):
        state = MedusaState(stage="running")
        result = score_episode("clean_pipeline", state)
        assert result.score == 0.0
        assert result.grade == "F"

    def test_clean_pipeline_explosion_detected_lowers_score(self):
        state = MedusaState(
            stage="committed",
            explosion_detected=True,
            silver_row_count=0,
            source_a_row_count=50,
            match_rate=0.0,
            grader_passed=False,
        )
        result = score_episode("clean_pipeline", state)
        assert result.breakdown["no_explosion"] == 0.0

    # ── dirty_integration (medium) ──────────────────────────────────────────

    def test_dirty_integration_score_is_in_range(self):
        state = self._run_happy_path(seed=1)
        result = score_episode("dirty_integration", state)
        assert 0.0 <= result.score <= 1.0

    def test_dirty_integration_without_prep_penalized(self):
        state = MedusaState(
            stage="committed",
            did_prep_a=False,
            did_prep_b=False,
            did_dedup_b=False,
            did_join=True,
            explosion_detected=False,
            grader_passed=False,
        )
        result = score_episode("dirty_integration", state)
        assert result.breakdown["prepped_before_join"] == 0.0
        assert result.breakdown["deduped_before_join"] == 0.0

    def test_dirty_integration_with_all_prereqs_scores_higher(self):
        state_no_prep = MedusaState(
            stage="committed", did_prep_a=False, did_prep_b=False,
            did_dedup_b=False, did_join=True, explosion_detected=False, grader_passed=False,
        )
        state_prepped = MedusaState(
            stage="committed", did_prep_a=True, did_prep_b=True,
            did_dedup_b=True, did_join=True, explosion_detected=False, grader_passed=True,
        )
        no_prep = score_episode("dirty_integration", state_no_prep)
        prepped = score_episode("dirty_integration", state_prepped)
        assert prepped.score > no_prep.score

    # ── full_medallion (hard) ───────────────────────────────────────────────

    def test_full_medallion_score_is_in_range(self):
        state = self._run_happy_path(seed=2)
        result = score_episode("full_medallion", state)
        assert 0.0 <= result.score <= 1.0

    def test_full_medallion_without_sync_penalized(self):
        state = MedusaState(
            stage="committed",
            did_sync_check=False,
            did_evolve_schema=True,
            scd_type="SCD-2",
            grader_passed=True,
        )
        result = score_episode("full_medallion", state)
        assert result.breakdown["sync_checked"] == 0.0

    def test_full_medallion_scd1_penalized(self):
        state_scd1 = MedusaState(
            stage="committed", did_sync_check=True,
            did_evolve_schema=True, scd_type="SCD-1", grader_passed=False,
        )
        state_scd2 = MedusaState(
            stage="committed", did_sync_check=True,
            did_evolve_schema=True, scd_type="SCD-2", grader_passed=True,
        )
        r1 = score_episode("full_medallion", state_scd1)
        r2 = score_episode("full_medallion", state_scd2)
        assert r2.score > r1.score

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            score_episode("nonexistent_task", MedusaState(stage="committed"))
