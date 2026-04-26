"""Build the SFT JSONL dataset on the real Olist 30-day gauntlet.

This is a self-contained sibling of `scripts/generate_sft_dataset.py`. The
key difference is the swap requested in MEDUSA4 planning:

    -    generator = RandomizedSchemaGenerator(episode_seed=seed, n_rows=50)
    +    generator = OlistDayGenerator(episode_seed=seed, n_rows=100,
    +                                  data_dir="data/olist")

so the SFT distribution is exactly the eval distribution
(`eval_grpo_olist.py`) and the rule-based expert produces ~100% commit
rate, ~100% JSON-valid rate, and total reward indistinguishable from the
`--run-baseline` line.

Notes (kept as design rationale):
  * Olist episodes are deterministic (anomalies live in
    `anomalies_map.json`), so `--episodes 1` is enough for the action
    target distribution. To inflate row count for SFT, prefer
    `--paraphrase-passes` over `--episodes`: same (prompt, action) pair,
    different `<think>` text drawn from a small phrasing bank.
  * The legacy 10% "wrong action" branch is OFF by default (`--no-mistakes`
    is the default) — the polished demo case wants 100% expert. Pass
    `--with-mistakes` to re-enable it for recovery training.
  * `n_rows=100` matches `eval_grpo_olist.py` exactly. Some role detections
    (negative-value detection in particular) are sensitive to row count;
    do not lower it without re-checking the role-detection heuristics.

CLI examples
------------
# Polished demo distribution: 1 episode, 4 paraphrase passes, no mistakes
python trainer/generate_sft_dataset_olist.py \
    --episodes 1 --paraphrase-passes 4 \
    --out trainer/sft_dataset_olist.jsonl

# Recovery-aware distribution: 5 episodes, mistakes on, 2 paraphrase passes
python trainer/generate_sft_dataset_olist.py \
    --episodes 5 --paraphrase-passes 2 --with-mistakes \
    --out trainer/sft_dataset_olist_recovery.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path so we can import the env / scenarios / models.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medusa_prompts import SYSTEM_PROMPT  # noqa: E402
from models import MedusaAction  # noqa: E402
from scenarios import OlistDayGenerator  # noqa: E402
from server.medusa_env import MedusaEnv  # noqa: E402


# ---------------------------------------------------------------------------
# Expert solver (situation-aware version, supports paraphrasing)
# ---------------------------------------------------------------------------
#
# Same seven-action vocabulary the eval / `--run-baseline` expert uses.
# `_classify_situation` returns enough metadata that we can re-render the
# `<think>` reasoning N times per (prompt, action) pair without re-running
# the env (which is wasteful for Olist — every episode is byte-identical).
# ---------------------------------------------------------------------------


_BAD_KEYWORDS = ("BLOCK", "INVALID", "not found", "ERROR", "Penalty")


_PARAPHRASES: dict[str, list[str]] = {
    "recovery": [
        "Wait, my previous action failed: {err} Let me correct my approach. ",
        "The last step came back with an error: {err} I'll fix the plan and continue. ",
        "Previous action didn't land cleanly: {err} Re-routing. ",
        "Got a non-OK signal from the env: {err} Adjusting and retrying. ",
    ],
    "evolve_drift": [
        "Schema drift detected: new columns {cols}. I must evolve the silver schema to data contract spec before merging.",
        "I see new columns {cols} in this batch — silver doesn't know about them yet. EVOLVE_SILVER_SCHEMA on '{first}' first; merge will block otherwise.",
        "Bronze has columns {cols} that aren't in silver. Step one is to EVOLVE_SILVER_SCHEMA('{first}') so the merge can succeed.",
        "Schema drift on this day: {cols}. The silver schema needs '{first}' added before any merge step.",
    ],
    "anomaly_quarantine": [
        "I see an unhandled anomaly on column '{col}'. It requires a 'quarantine' operation.",
        "Column '{col}' has rows that need to be quarantined. Routing the bad rows out of bronze before the merge.",
        "Anomaly checklist says quarantine '{col}' — filing those rows off into the quarantine bucket.",
        "'{col}' has invalid values; quarantine is the right tool here, not a fill or cast.",
    ],
    "anomaly_evolve": [
        "I see an unhandled anomaly on column '{col}'. It requires an 'evolve' operation.",
        "Column '{col}' is missing from the silver schema. EVOLVE_SILVER_SCHEMA is the next move.",
        "The anomaly map flags '{col}' as evolve — adding it to silver before merge.",
    ],
    "anomaly_deduplicate": [
        "I see an unhandled anomaly on column '{col}'. It requires a 'deduplicate' operation.",
        "Duplicate keys flagged on '{col}'. Running DEDUPLICATE keyed on it.",
        "'{col}' has duplicate-row anomalies — DEDUPLICATE with key='{col}' fixes this.",
    ],
    "anomaly_type_mixed": [
        "I see an unhandled anomaly on column '{col}'. It requires a 'type_mixed' operation.",
        "Column '{col}' is type_mixed (string + numeric); CLEAN_COLUMN op=cast normalises it.",
        "Mixed-type values in '{col}' — casting to a single type before merge.",
    ],
    "anomaly_fill_null": [
        "I see an unhandled anomaly on column '{col}'. It requires a 'fill_null' operation.",
        "Null values detected in '{col}'. Backfilling with zero so downstream aggregations don't break.",
        "Anomaly checklist: fill_null on '{col}'. CLEAN_COLUMN op=fill_zero.",
    ],
    "anomaly_whitespace": [
        "I see an unhandled anomaly on column '{col}'. It requires a 'whitespace' operation.",
        "Column '{col}' has stray whitespace — stripping it via CLEAN_COLUMN op=strip.",
        "Whitespace anomaly on '{col}'; a strip pass cleans this up.",
    ],
    "anomaly_negative": [
        "I see an unhandled anomaly on column '{col}'. It requires a 'negative' operation.",
        "Column '{col}' has spurious negative values flagged in the anomaly checklist; CLEAN_COLUMN handles it.",
        "Negative-value anomaly on '{col}'. Cleaning with CLEAN_COLUMN.",
    ],
    "anomaly_generic": [
        "I see an unhandled anomaly on column '{col}'. It requires a '{op}' operation.",
        "Anomaly map flags '{col}' for op='{op}'. CLEAN_COLUMN with that op is the right move.",
    ],
    "dedup": [
        "All initial anomalies are handled, but I have not deduplicated the batch yet. Must deduplicate.",
        "Bronze cleanup is done; running DEDUPLICATE so the merge gets a unique key set.",
        "Anomalies are clean. Time to dedupe before the merge stage runs.",
    ],
    "merge": [
        "The batch is cleaned and deduplicated. Ready to merge into Silver.",
        "Cleaning + dedup are complete. EXECUTE_MERGE writes the batch into the silver layer.",
        "Bronze is golden now — merging into silver.",
    ],
    "commit": [
        "Merge executed successfully. Committing the day.",
        "Silver merge succeeded. COMMIT_DAY closes out the day.",
        "Day's pipeline is finished. Committing.",
    ],
}


_OP_TO_KEY: dict[str, str] = {
    "quarantine": "anomaly_quarantine",
    "evolve": "anomaly_evolve",
    "deduplicate": "anomaly_deduplicate",
    "type_mixed": "anomaly_type_mixed",
    "fill_null": "anomaly_fill_null",
    "whitespace": "anomaly_whitespace",
    "negative": "anomaly_negative",
}


def _classify_situation(env: MedusaEnv) -> tuple[str, dict[str, Any]]:
    """Inspect env state, return (situation_key, format_kwargs).

    Mirrors the branching of the expert solver in `eval_grpo_olist.py`
    without producing the final string — we keep the situation key so the
    caller can render multiple paraphrases of the same logical reasoning.
    """
    state = env.state

    if state.new_schema_cols and not state.did_evolve_schema:
        return (
            "evolve_drift",
            {"cols": list(state.new_schema_cols), "first": state.new_schema_cols[0]},
        )

    if state.unhandled_anomalies_today:
        col = next(iter(state.unhandled_anomalies_today))
        op = state.unhandled_anomalies_today[col][0]
        key = _OP_TO_KEY.get(op, "anomaly_generic")
        return key, {"col": col, "op": op}

    if not state.did_dedup_today:
        return "dedup", {}
    if not state.did_merge_today:
        return "merge", {}
    return "commit", {}


def _expert_action_for_situation(
    env: MedusaEnv, key: str, fmt: dict[str, Any]
) -> MedusaAction:
    """Map (situation_key, fmt) -> the deterministic expert MedusaAction."""
    state = env.state

    if key == "evolve_drift":
        return MedusaAction(
            action="EVOLVE_SILVER_SCHEMA",
            params={"column": state.new_schema_cols[0]},
        )
    if key == "anomaly_quarantine":
        col = fmt["col"]
        return MedusaAction(
            action="QUARANTINE_ROWS",
            params={"table": "bronze", "condition": f"{col} IS NULL"},
        )
    if key == "anomaly_evolve":
        return MedusaAction(action="EVOLVE_SILVER_SCHEMA", params={"column": fmt["col"]})
    if key == "anomaly_deduplicate":
        return MedusaAction(action="DEDUPLICATE", params={"key": fmt["col"]})
    if key == "anomaly_type_mixed":
        return MedusaAction(action="CLEAN_COLUMN", params={"col": fmt["col"], "op": "cast"})
    if key == "anomaly_fill_null":
        return MedusaAction(action="CLEAN_COLUMN", params={"col": fmt["col"], "op": "fill_zero"})
    if key == "anomaly_whitespace":
        return MedusaAction(action="CLEAN_COLUMN", params={"col": fmt["col"], "op": "strip"})
    if key == "anomaly_negative":
        return MedusaAction(action="CLEAN_COLUMN", params={"col": fmt["col"], "op": "negative"})
    if key == "anomaly_generic":
        return MedusaAction(action="CLEAN_COLUMN", params={"col": fmt["col"], "op": fmt["op"]})
    if key == "dedup":
        return MedusaAction(action="DEDUPLICATE", params={})
    if key == "merge":
        return MedusaAction(action="EXECUTE_MERGE", params={})
    return MedusaAction(action="COMMIT_DAY", params={})


def _recovery_prefix(env: MedusaEnv, paraphrase_idx: int) -> str:
    last = env.state.last_action_result or ""
    if not any(k in last for k in _BAD_KEYWORDS):
        return ""
    bank = _PARAPHRASES["recovery"]
    template = bank[paraphrase_idx % len(bank)]
    return template.format(err=last.strip())


def _render_reason(key: str, fmt: dict[str, Any], paraphrase_idx: int) -> str:
    bank = _PARAPHRASES.get(key)
    if not bank:
        return ""
    template = bank[paraphrase_idx % len(bank)]
    return template.format(**fmt)


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------


def _format_assistant(reasoning: str, action: MedusaAction) -> str:
    """Format the assistant turn: <think>...</think> + fenced JSON action.

    The fence is critical: env messages can contain '{...}' (grader
    reports etc.), so leaking those into the reasoning text would break a
    greedy '\\{.*\\}' regex parse. Wrapping the action JSON in
    ```json...``` makes the eval-side parser unambiguous regardless of
    reasoning content.
    """
    action_json = json.dumps({"action": action.action, "params": action.params})
    return (
        f"<think>\n{reasoning}\n</think>\n"
        f"```json\n{action_json}\n```"
    )


def _emit_row(
    fp,
    prompt: str,
    reasoning: str,
    action: MedusaAction,
) -> None:
    chatml = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _format_assistant(reasoning, action)},
        ]
    }
    fp.write(json.dumps(chatml) + "\n")


def generate_dataset(
    out_path: Path,
    episodes: int = 1,
    n_rows: int = 100,
    data_dir: str = "data/olist",
    max_steps: int = 100,
    paraphrase_passes: int = 4,
    no_mistakes: bool = True,
    mistake_prob: float = 0.10,
    rng_seed: int = 0,
    base_seed: int = 1000,
) -> int:
    """Build the SFT dataset and return the number of rows written.

    paraphrase_passes
        How many copies of each transition to emit, each with a different
        `<think>` phrasing drawn from the paraphrase bank. The action
        target and prompt are unchanged — cheap data multiplier.
    no_mistakes
        If True (default), never inject the random "wrong action" branch.
        Recommended for the polished demo: SFT distribution = eval
        distribution = 100% expert.
    mistake_prob
        Probability of injecting a wrong action in any non-recovering
        step when `no_mistakes=False`. Mistakes are only ever injected
        on the canonical (paraphrase index 0) pass; later paraphrase
        passes always render the expert trajectory cleanly.
    """
    if paraphrase_passes < 1:
        raise ValueError("--paraphrase-passes must be >= 1")

    rng = random.Random(rng_seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    successful_steps = 0

    bad_actions = [
        MedusaAction(action="WRONG_SYNTAX_ACTION", params={"col": "fake"}),
        MedusaAction(action="CLEAN_COLUMN", params={"col": "hallucinated_col", "op": "cast"}),
        # If called before dedup, EXECUTE_MERGE blocks → recoverable.
        MedusaAction(action="EXECUTE_MERGE", params={}),
    ]

    print(
        f"[gen] olist episodes={episodes} paraphrase_passes={paraphrase_passes} "
        f"no_mistakes={no_mistakes} -> {out_path}"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(episodes):
            seed = base_seed + ep

            # The hack: real Olist generator + n_rows=100 + same expert.
            generator = OlistDayGenerator(
                episode_seed=seed,
                n_rows=n_rows,
                data_dir=data_dir,
            )
            env = MedusaEnv(day_generator=generator, max_steps=max_steps)
            obs = env.reset()

            transitions: list[tuple[str, str, dict[str, Any], MedusaAction]] = []

            while not obs.done:
                prompt = env.generate_llm_prompt()
                key, fmt = _classify_situation(env)
                action = _expert_action_for_situation(env, key, fmt)

                is_recovering = any(
                    k in env.state.last_action_result for k in _BAD_KEYWORDS
                )

                if (
                    not no_mistakes
                    and not is_recovering
                    and rng.random() < mistake_prob
                ):
                    bad_action = rng.choice(bad_actions)
                    _emit_row(f, prompt, "I will attempt this syntax blindly.", bad_action)
                    successful_steps += 1
                    obs = env.step(bad_action)
                    continue

                reasoning_0 = _recovery_prefix(env, 0) + _render_reason(key, fmt, 0)
                _emit_row(f, prompt, reasoning_0, action)
                successful_steps += 1
                transitions.append((prompt, key, fmt, action))

                obs = env.step(action)

            for p_idx in range(1, paraphrase_passes):
                for prompt, key, fmt, action in transitions:
                    reasoning = _render_reason(key, fmt, p_idx)
                    _emit_row(f, prompt, reasoning, action)
                    successful_steps += 1

            cumulative = env.state.cumulative_reward
            print(
                f"[gen] episode {ep + 1}/{episodes} (seed={seed}) "
                f"transitions={len(transitions)} reward={cumulative:.2f}"
            )

    print(f"[gen] wrote {successful_steps} rows to {out_path}")
    return successful_steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Olist-grounded SFT JSONL dataset."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="trainer/sft_dataset_olist.jsonl",
        help="Output JSONL path (default: %(default)s).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "Number of full 30-day episodes to roll out. Olist episodes are "
            "deterministic, so 1 is enough unless --with-mistakes is on "
            "(in which case extra episodes give different RNG mistakes)."
        ),
    )
    parser.add_argument("--data-dir", type=str, default="data/olist")
    parser.add_argument(
        "--rows-per-day",
        type=int,
        default=100,
        help="n_rows passed to OlistDayGenerator (default: 100, matches eval).",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--paraphrase-passes",
        type=int,
        default=4,
        help=(
            "Emit this many `<think>` phrasings per (prompt, action). The "
            "action target is unchanged — cheap multiplier."
        ),
    )
    parser.add_argument(
        "--mistake-prob",
        type=float,
        default=0.10,
        help="Wrong-action probability when --with-mistakes is set.",
    )

    mistakes = parser.add_mutually_exclusive_group()
    mistakes.add_argument(
        "--no-mistakes",
        dest="no_mistakes",
        action="store_true",
        default=True,
        help="Drop the random wrong-action branch (default).",
    )
    mistakes.add_argument(
        "--with-mistakes",
        dest="no_mistakes",
        action="store_false",
        help="Re-enable the random wrong-action branch (recovery training).",
    )

    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        out_path=Path(args.out),
        episodes=args.episodes,
        n_rows=args.rows_per_day,
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        paraphrase_passes=args.paraphrase_passes,
        no_mistakes=args.no_mistakes,
        mistake_prob=args.mistake_prob,
        rng_seed=args.rng_seed,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
