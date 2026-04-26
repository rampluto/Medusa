"""Shared prompt and action-vocabulary constants for the Medusa pipeline.

Single source of truth for `VALID_ACTIONS` and `SYSTEM_PROMPT` used by:

  - `scripts/generate_sft_dataset.py`  (SFT data builder)
  - `train_sft.py`                     (SFT trainer, indirectly via dataset)
  - `trainer/train_medusa_grpo.py`     (GRPO trainer)
  - `eval_grpo_olist.py`               (Olist 30-day evaluator)
  - `trainer/grpo_predictor_hub.py`     (UI / Space GRPO agent)
  - `server/agent_policies.py`         (grpo_trained policy validation)
  - `server.medusa_env.generate_llm_prompt` (user turn — Heuristic state signals)

`SYSTEM_PROMPT` is **algorithmically aligned** with `HeuristicPolicy` in
`server/agent_policies.py` (same class as “Heuristic Golden-Path” in the UI).
The LLM must follow that procedure so its emitted action matches the
reference solver, **using only the user turn + this message** (no code execution).

All sites must agree on action names and prompt format, otherwise the
policy collapses (cf. the `deduplicate_rows` mode-collapse incident).
"""

from __future__ import annotations

import json

VALID_ACTIONS: frozenset[str] = frozenset(
    {
        "PROFILE_TABLE",
        "CLEAN_COLUMN",
        "DEDUPLICATE",
        "EVOLVE_SILVER_SCHEMA",
        "QUARANTINE_ROWS",
        "EXECUTE_MERGE",
        "COMMIT_DAY",
    }
)


def _action_catalog_block() -> str:
    return (
        '  PROFILE_TABLE         {"table": "bronze"}  (first action most days; silver only if the task explicitly needs silver)\n'
        '  CLEAN_COLUMN          {"table": "bronze", "col": <str>, "op": <str>}\n'
        "                        op is one of: cast, fill_zero, strip, negative  (or other op the checklist names after mapping below)\n"
        '  DEDUPLICATE           {}  or  {"key": <str>}\n'
        '  EVOLVE_SILVER_SCHEMA  {"column": <str>}\n'
        '  QUARANTINE_ROWS       {"table": "bronze", "condition": "<col> IS NULL"}\n'
        "  EXECUTE_MERGE         {}\n"
        "  COMMIT_DAY            {}\n"
    )


def _valid_action_enumeration() -> str:
    return ", ".join(f'"{a}"' for a in sorted(VALID_ACTIONS))


_EXAMPLE_PROFILE = json.dumps(
    {"action": "PROFILE_TABLE", "params": {"table": "bronze"}},
    ensure_ascii=False,
)
_EXAMPLE_DEDUP_EMPTY = json.dumps(
    {"action": "DEDUPLICATE", "params": {}},
    ensure_ascii=False,
)
_EXAMPLE_CLEAN = json.dumps(
    {
        "action": "CLEAN_COLUMN",
        "params": {"table": "bronze", "col": "example_col", "op": "strip"},
    },
    ensure_ascii=False,
)

SYSTEM_PROMPT: str = (
    "You are the **Heuristic Golden-Path** policy, expressed as an LLM. You must return "
    "**exactly the same** action the rule-based `HeuristicPolicy` in the Medusa codebase "
    "would return on this turn, so the 30-day episode can pass every day. No creativity — "
    "execute the same deterministic program they use.\n"
    "\n"
    "## Closed vocabulary\n"
    "The `action` string must be one of: "
    f"{_valid_action_enumeration()}.\n"
    "Tool param shapes (match the Heuristic / golden-path code):\n"
    f"{_action_catalog_block()}"
    "\n"
    "## Raw-op → CLEAN_COLUMN `op` (and tool choice) — same as `_HEURISTIC_OP_MAP`\n"
    "When the checklist line shows `raw_op` (lowercased) for a column, map before emitting CLEAN_COLUMN:\n"
    "  - type_mixed   → op=cast\n"
    "  - fill_null   → op=fill_zero\n"
    "  - whitespace  → op=strip\n"
    "  - negative     → op=fill_zero\n"
    "If `raw_op` is anything else, pass it through as `op` **only** if it is a valid clean op; "
    "otherwise use the string from the environment’s checklist (strip/cast/fill_zero/etc.).\n"
    "\n"
    "## Decision procedure (must follow in order; read the **user** state each time)\n"
    "Let `cleaned` = the list in “Columns cleaned this day (col, op)”. Treat as a set of (col, op) "
    "pairs. Let `uniqueness_b` and the booleans come from the user message. Let **checklist** = "
    "“=== ANOMALY CHECKLIST (process strictly top to bottom) ===” numbered lines, in order.\n"
    "\n"
    "**Step A — profile once per day (matches Heuristic first branch)**\n"
    "If the user line `Profiled at least one table today` is **No** (or missing / empty profile state), "
    f"your **only** output is: ```json\n{_EXAMPLE_PROFILE}\n```\n"
    "Do not skip this to fix anomalies; the environment expects bronze profile first in the same way "
    "as the reference “Heuristic Golden-Path” policy.\n"
    "\n"
    "**Step B — walk the ANOMALY CHECKLIST in order (each line: column=… raw_op=…)**\n"
    "For each line in top-to-bottom order, if this step is not satisfied yet, **emit one action and stop**; "
    "the reference code returns immediately after each tool call.\n"
    "Lowercase `raw_op` for tests:\n"
    "  - If raw_op is deduplicate / dedup / duplicate: if Is deduplication complete for today is No, return "
    "DEDUPLICATE with key equal to that column; if it is already Yes, continue to the next line.\n"
    "  - If raw_op is quarantine: if (column, the literal tag quarantine) is not in cleaned, return "
    "QUARANTINE_ROWS with table bronze and condition col IS NULL; else continue.\n"
    "  - If raw_op is evolve: if Schema evolved is No and (column, evolve) is not in cleaned, return "
    "EVOLVE_SILVER_SCHEMA with that column; else continue.\n"
    "  - Else: map raw_op to mapped per the table; if (column, mapped) is in cleaned, continue; else return "
    "CLEAN_COLUMN with table bronze, col, and op mapped (always include table=bronze).\n"
    "\n"
    "**Step C — dedup fallback (after checklist exhausted)**\n"
    "If you reach here: if “Is deduplication complete for today” is **No** **and** `uniqueness_b` < 1.0, "
    f"return: ```json\n{_EXAMPLE_DEDUP_EMPTY}\n```\n"
    "\n"
    "**Step D — merge**\n"
    "If “Merge into silver done today” is **No**, return EXECUTE_MERGE with `{{}}`.\n"
    "\n"
    "**Step E — commit**\n"
    "Otherwise return COMMIT_DAY with `{{}}`.\n"
    "\n"
    "If the user’s “Schema Drift” line and “ACTIVE ANOMALIES” look like duplicates of the same work, the "
    "**checklist order in Step B still wins**; the drift line is a reminder, not a second parallel rule set.\n"
    "\n"
    "## Output format (strict)\n"
    "One ```json block only; inside, a **single** JSON object with exactly `action` and `params` (double-quoted). "
    "No Python, no `repr`, no `metadata=`, no text outside the fence.\n"
    "Example for a strip / whitespace fix:\n"
    "```json\n"
    f"{_EXAMPLE_CLEAN}\n"
    "```\n"
    "Do not emit anything after the closing ``` of the json fence."
)
