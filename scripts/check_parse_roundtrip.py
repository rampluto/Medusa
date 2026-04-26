#!/usr/bin/env python3
"""Sanity-check that SFT dataset rows round-trip through eval's parse_action.

Usage:
    python scripts/check_parse_roundtrip.py            # generates 1 episode in-memory
    python scripts/check_parse_roundtrip.py --jsonl data/sft_random.jsonl

The check exercises three things in one shot:
  1. The shared SYSTEM_PROMPT and VALID_ACTIONS in `medusa_prompts.py` are in
     sync between the SFT generator and the evaluator.
  2. Every assistant turn produced by `scripts/generate_sft_dataset.py` parses
     cleanly via `eval_grpo_olist.parse_action`, with `error is None` and the
     emitted action name in `VALID_ACTIONS`.
  3. The 10%-deliberate-error trajectories (which legitimately emit unknown
     action names like `WRONG_SYNTAX_ACTION`) are correctly classified as
     `invalid_action:<name>` rather than something flakier (invalid_json,
     missing_json, etc.). Those are *expected* failures — they teach error
     recovery — but we want them to fail in a *parseable* way so the SFT
     model still learns the JSON envelope.

Exit code 0 = all rows parsed; 1 = at least one row had an unexpected error
(e.g. invalid_json on a happy-path row). Print summary either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medusa_prompts import VALID_ACTIONS  # noqa: E402
from eval_grpo_olist import parse_action  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Round-trip SFT rows through eval parser.")
    p.add_argument(
        "--jsonl",
        default="",
        help=(
            "Path to an existing SFT JSONL produced by generate_sft_dataset.py. "
            "If empty, generates one fresh episode in-memory."
        ),
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=200,
        help="Stop after checking this many rows (default 200).",
    )
    return p.parse_args()


def iter_rows_from_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


def iter_rows_from_fresh_episode():
    """Generate one episode in-memory using the same code path as the SFT script."""
    import random

    from scripts.generate_sft_dataset import (
        RandomizedSchemaGenerator,
        get_expert_action,
    )
    from server.medusa_env import MedusaEnv
    from medusa_prompts import SYSTEM_PROMPT
    from models import MedusaAction

    random.seed(0)
    seed = 1000
    generator = RandomizedSchemaGenerator(episode_seed=seed, n_rows=50)
    env = MedusaEnv(day_generator=generator, max_steps=100)
    obs = env.reset()

    while not obs.done:
        prompt = env.generate_llm_prompt()
        reasoning, action = get_expert_action(env)

        # Mirror the deliberate-error path so we exercise it too.
        bad_keywords = ["BLOCK", "INVALID", "not found", "ERROR", "Penalty"]
        is_recovering = any(k in env.state.last_action_result for k in bad_keywords)
        if random.random() < 0.10 and not is_recovering:
            reasoning = "I will attempt this syntax blindly."
            bad_actions = [
                MedusaAction(action="WRONG_SYNTAX_ACTION", params={"col": "fake"}),
                MedusaAction(
                    action="CLEAN_COLUMN", params={"col": "hallucinated_col", "op": "cast"}
                ),
                MedusaAction(action="EXECUTE_MERGE", params={}),
            ]
            action = random.choice(bad_actions)

        action_json = json.dumps({"action": action.action, "params": action.params})
        action_text = (
            f"<think>\n{reasoning}\n</think>\n"
            f"```json\n{action_json}\n```"
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": action_text},
            ]
        }
        obs = env.step(action)


def main() -> int:
    args = parse_args()

    if args.jsonl:
        path = Path(args.jsonl)
        if not path.exists():
            print(f"[error] jsonl not found: {path}", file=sys.stderr)
            return 2
        rows = iter_rows_from_jsonl(path)
        source = str(path)
    else:
        rows = iter_rows_from_fresh_episode()
        source = "fresh in-memory episode"

    print(f"[check] source: {source}")
    print(f"[check] valid actions: {sorted(VALID_ACTIONS)}")

    n_total = 0
    n_happy = 0
    n_intentional_bad = 0
    unexpected_errors: Counter[str] = Counter()
    intentional_breakdown: Counter[str] = Counter()
    sample_failures: list[tuple[str, str]] = []

    for row in rows:
        if n_total >= args.max_rows:
            break
        n_total += 1
        msgs = row.get("messages", [])
        assistant = next(
            (m["content"] for m in msgs if m.get("role") == "assistant"), ""
        )

        action, error = parse_action(assistant)

        if action is not None and error is None:
            if action.action in VALID_ACTIONS:
                n_happy += 1
            else:
                # Should not happen — parse_action only returns action when name is valid.
                unexpected_errors[f"valid_object_invalid_name:{action.action}"] += 1
                if len(sample_failures) < 5:
                    sample_failures.append((assistant[:200], f"name={action.action}"))
        elif error and error.startswith("invalid_action:"):
            n_intentional_bad += 1
            intentional_breakdown[error] += 1
        else:
            unexpected_errors[error or "none"] += 1
            if len(sample_failures) < 5:
                sample_failures.append((assistant[:200], error or "none"))

    print(f"\n[check] rows checked          : {n_total}")
    print(f"[check] happy-path parsed     : {n_happy}")
    print(f"[check] intentional bad-action: {n_intentional_bad}")
    print(f"[check] unexpected errors     : {sum(unexpected_errors.values())}")
    if intentional_breakdown:
        print(f"[check]   intentional breakdown: {dict(intentional_breakdown)}")
    if unexpected_errors:
        print(f"[check]   unexpected breakdown : {dict(unexpected_errors)}")
        print("\n[check] sample failures (truncated):")
        for raw, err in sample_failures:
            print(f"  --- error={err}")
            print(f"  {raw!r}")

    if unexpected_errors:
        print("\n[check] FAIL: at least one row failed to parse cleanly.")
        return 1

    print("\n[check] OK: all rows round-trip cleanly through eval.parse_action.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
