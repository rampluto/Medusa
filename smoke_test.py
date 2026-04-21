"""MEDUSA v4.0 Smoke Test — validates the 30-day gauntlet end-to-end."""

import sys
sys.path.insert(0, '.')

from server.medusa_env import MedusaEnv
from models import MedusaAction


def act(action_str, params=None):
    """Build a properly formatted MedusaAction."""
    import json
    args_json = json.dumps(params or {})
    return MedusaAction(action=f"<action>{action_str}</action><args>{args_json}</args>")


def run_day(env, day):
    """Run one day: clean anomalies → merge → commit."""
    anomalies = env.state.day_anomalies.get(day, [])
    for col, op in anomalies:
        if op in ('fill_zero', 'strip', 'cast'):
            env.step(act("CLEAN_COLUMN", {"table": "bronze", "col": col, "op": op}))
    env.step(act("EXECUTE_MERGE"))
    return env.step(act("COMMIT_DAY"))


# ── Test 1: Basic 5-day gauntlet ─────────────────────────────────────────
print("=" * 60)
print("TEST 1: Basic 5-day gauntlet")
print("=" * 60)

env = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
env.reset(seed=42)

for day in range(1, 6):
    anomalies = env.state.day_anomalies.get(day, [])
    obs = run_day(env, day)
    print(f"  Day {day}: reward={obs.reward}, silver={env.state.silver_row_count}, "
          f"grader={'PASS' if env.state.grader_passed else 'FAIL'}")

assert env.state.current_day == 6
# Upsert merge: overlapping user_ids across days are updated, not appended
# so total may be less than 5 * 50 = 250
assert env.state.silver_row_count >= 200, f"Expected >= 200, got {env.state.silver_row_count}"
assert env.state.silver_row_count <= 250, f"Expected <= 250, got {env.state.silver_row_count}"
print(f"✓ 5-day gauntlet PASSED (Silver: {env.state.silver_row_count} rows)\n")


# ── Test 2: Day 8 Type Trap ──────────────────────────────────────────────
print("=" * 60)
print("TEST 2: Day 8 Type Trap (revenue as '$50.50')")
print("=" * 60)

env = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
env.reset(seed=42)

# Fast-forward to Day 8
for day in range(1, 8):
    run_day(env, day)

print(f"  Current day: {env.state.current_day}")
print(f"  Day 8 anomalies: {env.state.day_anomalies[8]}")
batch = env._current_batch
print(f"  Revenue sample: {list(batch.raw_data['revenue'].head(3))}")
print(f"  Is trap: {batch.is_trap_day}, type: {batch.trap_type}")

# Without fix — should fail
obs = env.step(act("EXECUTE_MERGE"))
obs = env.step(act("COMMIT_DAY"))
print(f"  Commit WITHOUT fix: reward={obs.reward}, grader={env.state.grader_report[:80]}")
assert obs.reward == -100.0, "Day 8 should crash without cleaning revenue"
print("✓ Day 8 trap correctly crashes without fix\n")

# With fix
env2 = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
env2.reset(seed=42)
for day in range(1, 8):
    run_day(env2, day)

obs = env2.step(act("CLEAN_COLUMN", {"table": "bronze", "col": "revenue", "op": "strip"}))
print(f"  Strip revenue: reward={obs.reward}")
obs = env2.step(act("CLEAN_COLUMN", {"table": "bronze", "col": "revenue", "op": "cast"}))
print(f"  Cast revenue: reward={obs.reward}")
obs = env2.step(act("EXECUTE_MERGE"))
obs = env2.step(act("COMMIT_DAY"))
print(f"  Commit WITH fix: reward={obs.reward}, grader={env2.state.grader_report[:50]}")
assert obs.reward > 0, "Day 8 should pass after cleaning revenue"
print("✓ Day 8 trap correctly passes with fix\n")


# ── Test 3: Profile escalating cost ──────────────────────────────────────
print("=" * 60)
print("TEST 3: Profile escalating cost")
print("=" * 60)

env3 = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
env3.reset(seed=42)

obs1 = env3.step(act("PROFILE_TABLE", {"table": "bronze"}))
print(f"  1st profile: reward={obs1.reward}")
assert obs1.reward == -0.2

obs2 = env3.step(act("PROFILE_TABLE", {"table": "bronze"}))
print(f"  2nd profile (same table): reward={obs2.reward}")
assert obs2.reward == -1.0

print("✓ Profile escalating cost PASSED\n")


# ── Test 4: Block/Retry mechanics ────────────────────────────────────────
print("=" * 60)
print("TEST 4: Block/Retry mechanics")
print("=" * 60)

env4 = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
env4.reset(seed=42)

# Clean same column twice — should block
env4.step(act("CLEAN_COLUMN", {"table": "bronze", "col": "revenue", "op": "strip"}))
obs = env4.step(act("CLEAN_COLUMN", {"table": "bronze", "col": "revenue", "op": "strip"}))
print(f"  Duplicate clean: reward={obs.reward}, msg={obs.message[:60]}")
assert obs.reward == -2.0
assert env4.state.retry_count == 1

print("✓ Block/Retry PASSED\n")


# ── Test 5: Legacy backward compat ───────────────────────────────────────
print("=" * 60)
print("TEST 5: Legacy backward compatibility")
print("=" * 60)

env5 = MedusaEnv(n_fact_rows=50, n_dim_rows=40)
env5.reset(seed=0)

obs = env5.step(MedusaAction(action="SYNC_CHECK"))
print(f"  SYNC_CHECK: reward={obs.reward}")
obs = env5.step(MedusaAction(action="PREP_KEYS_A"))
print(f"  PREP_KEYS_A: reward={obs.reward}")
obs = env5.step(MedusaAction(action="PREP_KEYS_B"))
print(f"  PREP_KEYS_B: reward={obs.reward}")
obs = env5.step(MedusaAction(action="EXECUTE_JOIN_LEFT"))
print(f"  EXECUTE_JOIN_LEFT: reward={obs.reward}")
obs = env5.step(MedusaAction(action="APPLY_SCD_2"))
print(f"  APPLY_SCD_2: reward={obs.reward}")
obs = env5.step(MedusaAction(action="COMMIT"))
print(f"  COMMIT: reward={obs.reward}, done={obs.done}")
print(f"  Grader: {env5.state.grader_report[:60]}")

assert obs.done is True
assert env5.state.stage == "committed"
print("✓ Legacy backward compat PASSED\n")


print("=" * 60)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 60)