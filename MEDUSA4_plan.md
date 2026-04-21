# MEDUSA-Chronos v4.0 — Revised Plan
**Target Theme:** Theme #3 — World Modeling (Professional Tasks)

## 1. The Pitch (30 seconds)
"We built a 30-day enterprise data pipeline that actively breaks. Type mismatches, OOM traps, schema drift, null bombs — injected on specific days, deterministically graded. An LLM data engineer must survive all 30 days using real ETL tool calls against a cumulative Silver layer. No LLM-as-judge, no shortcuts — just a programmatic grader running mathematical assertions on the output. The result: a training environment where you can watch an AI learn to be a data engineer."

**Why Theme #3:**
- Real interaction with tools (not toy grid-world)
- Dynamic system with feedback (data changes daily, Silver accumulates)
- Partial observability (agent doesn't know what trap is coming next)
- Multi-step workflow orchestration (profile → clean → dedup → merge → commit per day)
- Persistent world model (cumulative Silver table is the world state across 30 days)

---

## 2. Architecture — Single Agent, Rich Environment

| Component | Description |
|----------|------------|
| The Steward | LLM policy (Qwen2.5-7B). Takes tool actions. Trained via PPO |
| The Environment | 30-day gauntlet. Feeds daily batches, injects traps, tracks Silver state |
| The Grader | Deterministic Python assertions. Runs on commit_day(). No LLM involvement |

There is no second agent. The environment is the adversary.

---

## 3. The World — Cumulative Silver Layer

The environment simulates a Medallion (Bronze → Silver) data lakehouse.

- Silver is cumulative. It never resets. It is one growing dataset across the full 30-day episode  
- Each day: A new batch of raw data lands in Bronze. The agent must clean, validate, and merge it into Silver  
- Freshness Check: `len(Silver_after_commit) > len(Silver_before_commit)` — today's merge must have actually added rows  

This is the "world model" — Silver is the persistent state that the agent must reason about across 30 days.

---

## 4. Action Space (7 Tools)

| Tool | Effect |
|------|--------|
| profile_table(table) | Returns schema, types, null %, duplicate counts. 1st call/day = -0.2, 2nd call/day on same table = -1.0 |
| clean_column(table, col, op) | Applies cast, strip, or fill_zero. Tracks `(column, operation)` tuples — repeating the exact same (col, op) pair today → BLOCK. Different operations on the same column are allowed (e.g., strip then cast on revenue) |
| deduplicate(table, key) | Removes duplicate rows. Capped at 1 call per table per day. 2nd call on same table → BLOCK |
| evolve_silver_schema(column) | Adds a new column to the Data Contract and Silver schema. Required when schema drift is detected (e.g., Day 21). Returns penalty if column doesn't exist in today's raw data |
| quarantine_rows(table, condition) | Routes rows matching condition to /quarantine. Required for unsalvageable rows (e.g., Day 28 null keys) |
| execute_merge() | Joins daily batch into cumulative Silver. Estimates output based on table state at call time (post-dedup if called). If estimated output exceeds memory limit → BLOCK with reason: "Estimated output exceeds memory limit. Current duplicate ratio: X%" |
| commit_day() | Submits Silver to the Grader. Advances the clock if all checks pass. Failure = Terminal Crash |

### BLOCK & Retry Mechanics
- BLOCK = action is prevented, -2.0 penalty, retry counter increments  
- BLOCK actions count toward the 10-step daily limit  
- Max 3 retries per day → Terminal Crash (-100)  
- Retries reset to 0 on successful commit_day()  

### Step Limit
- Max 10 actions per day (including BLOCKs). If the agent has not called commit_day() within 10 steps, the environment auto-triggers Terminal Crash (-100)  
- This bounds each episode to max 300 steps (10 × 30 days), preventing stalling and GPU burn  

---

## 5. The 30-Day Gauntlet

**IMPORTANT**

Every day has ≥1 anomaly. The data generator must verify that raw data fails at least one grader assertion before writing each day's batch. No free days.

### Major Trap Days (fixed, unique traps)

| Day | Trap | Canonical Resolution | Grader Check |
|-----|------|----------------------|--------------|
| 8 | Type Trap: revenue as "$50.50" | clean_column(strip "$") then clean_column(cast float) | Silver.dtypes['revenue'] == float64 |
| 14 | OOM Trap: massive duplicate keys | profile_table → deduplicate → execute_merge | Merge estimate at call-time. Post-dedup estimate passes |
| 21 | Schema Drift: promo_code appears | evolve_silver_schema("promo_code") | Silver.columns == Current_Contract.columns |
| 28 | Null Nuke: 20% of user_id NULL | quarantine_rows(table, "user_id IS NULL") | NULL user_id rows in /quarantine, not Silver |

### Gap Days (recycled from anomaly pool)

Days 1–7, 9–13, 15–20, 22–27, 29–30 draw from a fixed corruption pool:

| Corruption | Operation | Example Columns |
|-----------|----------|----------------|
| Null injection (5-15% of rows) | clean_column(fill_zero) | discount_amount, quantity, price |
| Trailing whitespace | clean_column(strip) | customer_name, product_name, category |
| Minor type mismatch | clean_column(cast) | quantity as string, price as string |

Each gap day gets 1–2 corruptions assigned deterministically at environment init (seeded by day number). The anomaly checklist for each day is stored as `day_anomalies: Dict[int, List[Tuple[col, op]]]` and is the source of truth for both the grader and the +0.5 reward gate (see §7).

### Quarantine Ceiling (End-of-Episode)

No day-by-day tracking. At Day 30 completion:

`len(Total_Quarantine) / len(Total_Raw_All_30_Days) <= 0.05 (excluding Day 28 approved nulls)`

Failing this negates the final completion bonus.

---

## 6. Observation Space

Each step, the agent receives a structured text prompt:

**Persistent context (always present):**
- Current Data Contract (target schema, column types, primary key)
- Current Day: X/30
- Today's landing zone: files detected, row counts

**Dynamic context (updates per step):**
- Output of the agent's last tool call
- Whether a BLOCK was triggered and why
- Running list of columns already cleaned today

The observation is the "world" as the agent sees it. It must learn to read profile output, infer what's wrong, and choose the right tool — not memorize a fixed sequence.

---

## 7. Reward Structure

| Event | Reward | Anti-Exploit Guard |
|------|--------|-------------------|
| Any valid action (step cost) | -0.2 | — |
| profile_table 2nd call/day on same table | -1.0 | Escalating cost prevents stalling |
| clean_column on a (col, op) on today's anomaly checklist | +0.5 | Only fires for checklist entries |
| clean_column on a (col, op) not on checklist | -0.2 (step cost only) | No positive reward for unnecessary cleaning |
| deduplicate that removes > 0 rows (1st call/day) | +0.5 | 2nd call/day → BLOCK, not reward |
| evolve_silver_schema on a real drifted column | +1.0 | Returns -1.0 if column doesn't exist in today's raw data |
| quarantine_rows on rows matching today's checklist | +0.5 | Quarantining clean rows = -0.2 step cost only |
| BLOCK triggered | -2.0 | Counts toward 10-step limit |
| Successful commit_day() | +Day Number (Day 1 = +1, Day 30 = +30) | Only fires once/day |
| Step limit exceeded (10 steps, no commit) | -100.0 (Terminal Crash) | Prevents infinite stalling |
| Terminal Crash (grader fail) | -100.0 | — |
| Day 30 Completion + Clean Quarantine | +100.0 | — |

**IMPORTANT**

The anomaly checklist gates all positive rewards. The checklist is a `Dict[int, List[Tuple[col, op]]]` generated deterministically at env init.

- Theoretical maximum return: ~+1000  
- Typical early training return: ~-100 (crash on Day 8)  

---

## 8. Training Setup

- Algorithm: PPO via HuggingFace TRL  
- Model: Qwen2.5-7B-Instruct with LoRA  
- Observation format: Structured text prompt → tokenized by the model's tokenizer  
- Episode: One attempt at Days 1–30. Terminates on crash or Day 30 completion  
- OpenEnv registration: Single task. Grader score = normalized episode return  

### Key Metrics to Show Judges
- Crash day distribution shifting rightward (8 → 14 → 21 → 28 → 30 ✅) — the hero chart  
- Average episode return climbing from -100 toward +500  
- Average episode length increasing — shows the agent surviving longer  
- BLOCK frequency declining — shows the agent learning constraints  

Even with limited compute, showing the Day 8 → Day 14 transition is enough. Frame it as: "clear learning signal with headroom for more training."

---

## 9. Anti-Exploit Summary

| Exploit Vector | Guard |
|---------------|-------|
| deduplicate farming | 1 dedup call per table per day. 2nd call = BLOCK |
| clean_column shotgunning | +0.5 only for (col, op) tuples on today's checklist |
| Same-column multi-op abuse | Tracks (col, op) tuples — different ops on same col are allowed, exact repeats are BLOCKed |
| Never committing / stall | Max 10 steps per day (including BLOCKs) → auto Terminal Crash |
| profile_table spam | 2nd call/day on same table costs -1.0 |
| Raw commit on clean days | Data generator verifies raw data fails ≥1 grader assertion per day |
| quarantine_rows abuse | +0.5 only for rows matching today's checklist |

---

## 10. What's NOT in This Plan (Intentionally)

| Removed | Why |
|--------|-----|
| Auditor agent | Was a frozen if/else block dressed as multi-agent. Adds complexity, no real Theme #3 value |
| GRPO | PPO is more practical for long-horizon episodes with a value network |
| Day-by-day quarantine tracking | Simplified to end-of-episode check. Same training signal, less bookkeeping |

---

## 11. Theme #3 Alignment Checklist

| Theme #3 Requirement | How We Hit It |
|--------------------|--------------|
| "Real interaction with tools, APIs, or dynamic systems" | 5 ETL tools operating on actual DataFrames with real pandas operations |
| "Model expected to do real hard work instead of exploiting shortcuts" | Deterministic grader + anti-exploit guards — no reward hacking, no prompt-hacking |
| "Maintain consistent internal state" | Cumulative Silver layer is the persistent world state across 30 days |
| "Update beliefs based on outcomes" | Agent reads profile output and adapts tool choices per day |
| "Orchestrate multi-step workflows" | Every day requires a correct sequence: profile → clean → dedup → merge → commit |
| "Partially observable" | Agent doesn't know which trap is coming. Must infer from data |