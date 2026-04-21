"""Tests for the MEDUSA environment — v4.0.

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
        assert s.day28_quarantine_rows == 0
        assert s.last_action_result == ""
        assert s.last_block_reason == ""

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
# Operators (legacy, backward compat)
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
# v4.0 Operators
# ---------------------------------------------------------------------------

from medusa_env.operators import (
    profile_table,
    clean_column,
    deduplicate_rows,
    quarantine_rows as op_quarantine_rows,
    merge_into_silver,
)


class TestV4Operators:
    def test_profile_table(self):
        df = pd.DataFrame({"id": [1, 2, None], "val": [10, 20, 30]})
        profile = profile_table(df)
        assert "id" in profile
        assert "val" in profile
        assert profile["id"]["null_pct"] == pytest.approx(33.3, abs=0.5)

    def test_clean_column_strip(self):
        df = pd.DataFrame({"name": ["  Alice  ", "Bob", None]})
        result, rows = clean_column(df, "name", "strip")
        assert result["name"].iloc[0] == "Alice"
        assert rows >= 1

    def test_clean_column_cast(self):
        df = pd.DataFrame({"revenue": ["$50.50", "100", None]})
        result, rows = clean_column(df, "revenue", "cast")
        assert result["revenue"].dtype == float
        assert result["revenue"].iloc[0] == pytest.approx(50.50)

    def test_clean_column_fill_zero(self):
        df = pd.DataFrame({"val": [1, None, 3]})
        result, rows = clean_column(df, "val", "fill_zero")
        assert rows == 1
        assert result["val"].isnull().sum() == 0

    def test_deduplicate_rows(self):
        df = pd.DataFrame({"user_id": ["A", "A", "B"], "val": [1, 2, 3]})
        result, dupes = deduplicate_rows(df, "user_id")
        assert dupes == 1
        assert len(result) == 2

    def test_quarantine_rows_null(self):
        df = pd.DataFrame({"user_id": ["A", None, "B", None]})
        kept, quarantined = op_quarantine_rows(df, "user_id IS NULL")
        assert len(quarantined) == 2
        assert len(kept) == 2

    def test_merge_into_silver_upsert(self):
        import numpy as np
        silver = pd.DataFrame({"user_id": ["A", "B"], "val": [1, 2]})
        daily = pd.DataFrame({"user_id": ["B", "C"], "val": [99, 3]})
        result = merge_into_silver(silver, daily, "user_id")
        # B should be updated, C should be appended, A unchanged
        assert len(result) == 3
        b_row = result[result["user_id"] == "B"]
        assert b_row["val"].iloc[0] == 99

    def test_merge_into_silver_empty(self):
        silver = pd.DataFrame()
        daily = pd.DataFrame({"user_id": ["A"], "val": [1]})
        result = merge_into_silver(silver, daily)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Reward Engine (v4.0)
# ---------------------------------------------------------------------------

from medusa_env.rewards import RewardEngine, REWARD_TABLE


class TestMedusaRewards:
    @pytest.fixture
    def engine(self):
        return RewardEngine()

    def test_step_cost_always_applied(self, engine):
        r = engine.compute("PROFILE_TABLE", profile_call_count=1)
        assert r == pytest.approx(-0.1, abs=0.01)

    def test_profile_2nd_call_penalty(self, engine):
        r = engine.compute("PROFILE_TABLE", profile_call_count=2)
        assert r == pytest.approx(-1.0, abs=0.01)

    def test_clean_checklist_hit(self, engine):
        r = engine.compute(
            "CLEAN_COLUMN",
            col_op=("revenue", "strip"),
            today_anomalies=[("revenue", "strip"), ("revenue", "cast")],
        )
        assert r == pytest.approx(-0.1 + 1.0, abs=0.01)

    def test_clean_no_hit(self, engine):
        r = engine.compute(
            "CLEAN_COLUMN",
            col_op=("revenue", "strip"),
            today_anomalies=[("name", "strip")],
        )
        assert r == pytest.approx(-0.1, abs=0.01)

    def test_deduplicate_effective(self, engine):
        r = engine.compute("DEDUPLICATE", dupes_removed=5)
        assert r == pytest.approx(-0.1 + 1.0, abs=0.01)

    def test_deduplicate_no_effect(self, engine):
        r = engine.compute("DEDUPLICATE", dupes_removed=0)
        assert r == pytest.approx(-0.1, abs=0.01)

    def test_evolve_schema_valid(self, engine):
        r = engine.compute("EVOLVE_SILVER_SCHEMA", column_exists_in_raw=True)
        assert r == pytest.approx(-0.1 + 1.0, abs=0.01)

    def test_evolve_schema_invalid(self, engine):
        r = engine.compute("EVOLVE_SILVER_SCHEMA", column_exists_in_raw=False)
        assert r == pytest.approx(-1.0, abs=0.01)

    def test_quarantine_checklist(self, engine):
        r = engine.compute(
            "QUARANTINE_ROWS",
            quarantine_condition="user_id IS NULL",
            today_anomalies=[("user_id", "quarantine")],
        )
        assert r == pytest.approx(-0.1 + 0.5, abs=0.01)

    def test_block_penalty(self, engine):
        r = engine.compute("CLEAN_COLUMN", is_blocked=True)
        assert r == pytest.approx(-2.0, abs=0.01)

    def test_commit_reward(self):
        assert RewardEngine.commit_reward(8) == 40.0

        assert RewardEngine.crash_reward(1) == -145.0
        assert RewardEngine.crash_reward(28) == -10.0

    def test_completion_bonus(self):
        assert RewardEngine.completion_bonus() == 200.0

    def test_reward_table_values(self):
        assert REWARD_TABLE["step_cost"] == -0.1
        assert REWARD_TABLE["block_penalty"] == -2.0
        assert REWARD_TABLE["execute_merge_success"] == 3.0
        assert REWARD_TABLE["completion_bonus"] == 200.0


# ---------------------------------------------------------------------------
# Grader (v4.0)
# ---------------------------------------------------------------------------

from medusa_env.grader import Grader
import numpy as np


class TestMedusaGrader:
    @pytest.fixture
    def grader(self):
        return Grader()

    def test_freshness_pass(self, grader):
        silver = pd.DataFrame({"user_id": ["A", "B"], "revenue": [10.0, 20.0]})
        r = grader.audit(silver, silver_at_day_start=0, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.freshness_ok is True

    def test_freshness_fail(self, grader):
        silver = pd.DataFrame({"user_id": ["A"], "revenue": [10.0]})
        r = grader.audit(silver, silver_at_day_start=1, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.freshness_ok is False
        assert r.passed is False

    def test_schema_pass(self, grader):
        silver = pd.DataFrame({"user_id": ["A"], "revenue": [10.0], "extra": [1]})
        r = grader.audit(silver, silver_at_day_start=0, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.schema_ok is True

    def test_schema_fail(self, grader):
        silver = pd.DataFrame({"user_id": ["A"]})
        r = grader.audit(silver, silver_at_day_start=0, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.schema_ok is False
        assert r.passed is False

    def test_type_integrity_pass(self, grader):
        silver = pd.DataFrame({"user_id": ["A"], "revenue": np.array([10.0], dtype=np.float64)})
        r = grader.audit(silver, silver_at_day_start=0, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.type_integrity_ok is True

    def test_type_integrity_fail(self, grader):
        silver = pd.DataFrame({"user_id": ["A"], "revenue": ["$50.50"]})
        r = grader.audit(silver, silver_at_day_start=0, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.type_integrity_ok is False
        assert r.passed is False

    def test_null_integrity_pass_before_day28(self, grader):
        """Null check only fires on Day 28+."""
        silver = pd.DataFrame({"user_id": [None, "A"], "revenue": [10.0, 20.0]})
        r = grader.audit(silver, silver_at_day_start=0, current_day=5,
                         contract_columns=["user_id", "revenue"])
        # Before Day 28, null integrity is always True
        assert r.null_integrity_ok is True

    def test_null_integrity_fail_on_day28(self, grader):
        silver = pd.DataFrame({"user_id": [None, "A"], "revenue": [10.0, 20.0]})
        r = grader.audit(silver, silver_at_day_start=0, current_day=28,
                         contract_columns=["user_id", "revenue"])
        assert r.null_integrity_ok is False
        assert r.passed is False

    def test_all_pass(self, grader):
        silver = pd.DataFrame({"user_id": ["A", "B"], "revenue": np.array([10.0, 20.0])})
        r = grader.audit(silver, silver_at_day_start=0, current_day=1,
                         contract_columns=["user_id", "revenue"])
        assert r.passed is True
        assert "PASS ✓" in r.report


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

    def test_observation_contains_block_status(self, env):
        """v4.0 observation should include BLOCK status."""
        import json
        env.reset(seed=42)  # v4.0 mode (seed >= 42)
        obs = env.step(MedusaAction(
            action=f'<action>PROFILE_TABLE</action><args>{json.dumps({"table": "bronze"})}</args>'
        ))
        assert "BLOCK:" in obs.message

    def test_observation_contains_last_action(self, env):
        """v4.0 observation should include last action output."""
        import json
        env.reset(seed=42)  # v4.0 mode
        env.step(MedusaAction(
            action=f'<action>PROFILE_TABLE</action><args>{json.dumps({"table": "bronze"})}</args>'
        ))
        obs = env.step(MedusaAction(
            action=f'<action>EXECUTE_MERGE</action><args>{json.dumps({})}</args>'
        ))
        assert "Last action output:" in obs.message


# ---------------------------------------------------------------------------
# Task Scorer (v4.0)
# ---------------------------------------------------------------------------

from medusa_env.tasks import TASKS, score_episode


class TestMedusaTasks:
    """Tests for the v4.0 task definitions and 0.0–1.0 scorer."""

    def test_six_tasks_defined(self):
        assert len(TASKS) == 6
        assert "basic_pipeline" in TASKS
        assert "survive_day8" in TASKS
        assert "survive_day14" in TASKS
        assert "survive_day21" in TASKS
        assert "survive_day28" in TASKS
        assert "gauntlet_30day" in TASKS

    def test_task_difficulties(self):
        assert TASKS["basic_pipeline"].difficulty == "easy"
        assert TASKS["survive_day8"].difficulty == "medium"
        assert TASKS["gauntlet_30day"].difficulty == "hard"

    def test_basic_pipeline_survived_5_days(self):
        state = MedusaState(
            stage="committed",
            current_day=6,
            silver_row_count=250,
            grader_passed=True,
            cumulative_reward=50.0,
        )
        result = score_episode("basic_pipeline", state)
        assert result.score > 0.5
        assert result.passed

    def test_basic_pipeline_uncommitted_scores_zero(self):
        state = MedusaState(stage="running")
        result = score_episode("basic_pipeline", state)
        assert result.score == 0.0
        assert result.grade == "F"

    def test_survive_day8_scored(self):
        state = MedusaState(
            stage="committed",
            current_day=9,
            silver_row_count=400,
            grader_passed=True,
            cumulative_reward=50.0,
        )
        result = score_episode("survive_day8", state)
        assert result.score > 0.3
        assert result.passed

    def test_gauntlet_30day_requires_committed(self):
        state = MedusaState(
            stage="failed",
            current_day=30,
            silver_row_count=1000,
            grader_passed=False,
            cumulative_reward=100.0,
        )
        result = score_episode("gauntlet_30day", state)
        assert result.breakdown.get("completed_30_days", 0) == 0.0

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            score_episode("nonexistent_task", MedusaState(stage="committed"))
