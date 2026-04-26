#!/usr/bin/env python3
"""Recover GRPO training stats from Hugging Face Jobs stdout logs and plot them.

Background
----------
`trainer/train_medusa_grpo.py` only pushes the final LoRA adapter to the Hub.
The intermediate `reward_log.jsonl` and `checkpoint-*/trainer_state.json` files
live on the runner's ephemeral disk and are destroyed when the HF Job ends.

However, the per-step training metrics are emitted to stdout in two patterns:

    [train] step=N {<json dict of metrics>}
    [reward] call=N mean=X moving=Y failures={...}

HF Jobs retains stdout/stderr after a job ends, accessible via
`hf jobs logs <JOB_ID>`. This script:

  1. Fetches the full log for a given JOB_ID (or reads a previously saved file).
  2. Parses both line patterns into pandas DataFrames.
  3. Writes CSV artifacts (df_train.csv, df_reward.csv) for downstream use.
  4. Renders standard training-health plots (reward, env reward, JSON reward,
     entropy, grad norm, completion length) plus a reward-trajectory plot.
  5. Prints a one-screen training summary.

Usage
-----
    # End-to-end (fetch + parse + plot):
    python recover_grpo_logs.py --job-id <JOB_ID> --output-dir grpo_recovery

    # Offline (when you already have logs saved to a file):
    python recover_grpo_logs.py --log-file logs.txt --output-dir grpo_recovery

    # Only fetch, no plotting:
    python recover_grpo_logs.py --job-id <JOB_ID> --no-plot

Outputs (under --output-dir):
    raw_logs.txt              full stdout/stderr of the job
    df_train.csv              one row per training step, all metrics
    df_reward.csv             one row per reward checkpoint (every 10 steps)
    panels.png                6-panel training-health figure
    reward_trajectory.png     batch-mean and moving-mean reward over steps

Requires: pandas, matplotlib (and the `hf` CLI on PATH if --job-id is used).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required: pip install pandas") from exc


STEP_RE = re.compile(r"\[train\] step=(\d+)\s+(\{.*?\})\s*$", re.MULTILINE)
REWARD_RE = re.compile(
    r"\[reward\] call=(\d+)\s+mean=([-\d.eE]+)\s+moving=([-\d.eE]+)\s+failures=(\{[^}]*\})"
)

PANEL_COLUMNS: list[tuple[str, str]] = [
    ("reward", "Total reward"),
    ("rewards/medusa_env_reward/mean", "Env reward (medusa)"),
    ("rewards/json_format_reward/mean", "JSON format reward"),
    ("entropy", "Policy entropy"),
    ("grad_norm", "Gradient norm"),
    ("completions/mean_length", "Mean completion length"),
]


_RATE_LIMIT_PATTERNS = ("rate limit", "/whoami-v2", "Too Many Requests")


def _is_rate_limited(stderr: str) -> bool:
    text = stderr.lower()
    return any(p.lower() in text for p in _RATE_LIMIT_PATTERNS)


def fetch_logs(
    job_id: str,
    dest: Path,
    namespace: str | None = None,
    max_retries: int = 6,
    initial_backoff: float = 30.0,
) -> Path:
    """Pull stdout+stderr for an HF Job and save to `dest`. Returns `dest`.

    Retries with exponential backoff when the HF API rate-limits `/whoami-v2`
    (the `hf` CLI hits that endpoint on every invocation). Does NOT call
    `hf auth whoami` itself, since that would consume the same rate budget.
    """
    if shutil.which("hf") is None:
        raise SystemExit(
            "The `hf` CLI was not found on PATH. Install with `pip install -U huggingface_hub` "
            "and run `hf auth login`, or pass --log-file with a previously saved log."
        )
    cmd = ["hf", "jobs", "logs"]
    if namespace:
        cmd += ["--namespace", namespace]
    cmd += [job_id]

    print(f"[fetch] {' '.join(cmd)} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    backoff = initial_backoff
    last_stderr = ""
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        last_stderr = (result.stderr or "").strip()

        if result.returncode == 0 and result.stdout:
            dest.write_text(result.stdout, encoding="utf-8")
            size_mb = dest.stat().st_size / 1e6
            print(f"[fetch] ok: {size_mb:.2f} MB (attempt {attempt})")
            return dest

        # Failed. Decide whether to retry.
        if _is_rate_limited(last_stderr) and attempt < max_retries:
            print(
                f"[fetch] rate-limited (attempt {attempt}/{max_retries}); "
                f"sleeping {backoff:.0f}s before retry"
            )
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 300.0)
            continue

        # Non-rate-limit failure or out of retries.
        break

    # Final failure: surface the real error and give actionable hints.
    print(f"[fetch] giving up after {attempt} attempt(s); exit={result.returncode}")
    if last_stderr:
        print("[fetch] stderr (last 20 lines):")
        for line in last_stderr.splitlines()[-20:]:
            print(f"  {line}")

    hints = ["", "Common causes:"]
    if _is_rate_limited(last_stderr):
        hints += [
            "  * HF /whoami-v2 rate limit. Wait ~5-15 min and retry, or use a token",
            "    that isn't shared by other concurrent processes. The `hf` CLI calls",
            "    /whoami-v2 on every invocation; you can cap the calls by avoiding",
            "    `hf auth whoami` and back-to-back `hf jobs ...` commands.",
        ]
    else:
        hints += [
            "  * Job belongs to an org. Pass: --hf-namespace <org-name>",
            "  * JOB_ID typo. Verify with: hf jobs ps -a",
            "  * Old CLI. Upgrade: pip install -U huggingface_hub",
        ]
    raise SystemExit("\n".join(hints))


def parse_train_lines(text: str) -> pd.DataFrame:
    """Parse `[train] step=N {...}` lines into a DataFrame keyed by step."""
    rows: list[dict] = []
    for match in STEP_RE.finditer(text):
        try:
            metrics = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        metrics["step"] = int(match.group(1))
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("step").sort_values("step").reset_index(drop=True)
    return df


def parse_reward_lines(text: str) -> pd.DataFrame:
    """Parse `[reward] call=N mean=... moving=... failures={...}` lines."""
    rows: list[dict] = []
    for match in REWARD_RE.finditer(text):
        try:
            failures = ast.literal_eval(match.group(4))
        except (SyntaxError, ValueError):
            failures = {}
        rows.append(
            {
                "call": int(match.group(1)),
                "batch_mean": float(match.group(2)),
                "moving_mean": float(match.group(3)),
                "failures": failures,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["call", "batch_mean", "moving_mean", "failures"])
    return (
        pd.DataFrame(rows)
        .drop_duplicates("call")
        .sort_values("call")
        .reset_index(drop=True)
    )


def aggregate_failures(failure_dicts: Iterable[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for d in failure_dicts:
        for key, val in d.items():
            counts[key] += int(val)
    return dict(counts)


def save_csvs(df_train: pd.DataFrame, df_reward: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "df_train.csv"
    df_train.to_csv(train_path, index=False)
    print(f"[save] {train_path} ({len(df_train)} rows)")

    reward_path = out_dir / "df_reward.csv"
    if not df_reward.empty:
        df_reward = df_reward.copy()
        df_reward["failures"] = df_reward["failures"].apply(json.dumps)
    df_reward.to_csv(reward_path, index=False)
    print(f"[save] {reward_path} ({len(df_reward)} rows)")


def plot_panels(df_train: pd.DataFrame, out_path: Path, window: int = 20) -> None:
    import matplotlib.pyplot as plt  # local import keeps CLI startup fast

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, PANEL_COLUMNS):
        if col not in df_train.columns:
            ax.set_visible(False)
            continue
        series = df_train[col]
        ax.plot(df_train["step"], series, lw=0.6, alpha=0.35, label="raw")
        ax.plot(
            df_train["step"],
            series.rolling(window).mean(),
            lw=2,
            label=f"{window}-step MA",
        )
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("GRPO training panels", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def plot_reward_trajectory(
    df_train: pd.DataFrame,
    df_reward: pd.DataFrame,
    out_path: Path,
    reward_log_every: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    if not df_reward.empty:
        x = df_reward["call"] * reward_log_every
        ax.plot(x, df_reward["batch_mean"], lw=0.8, alpha=0.45, label="batch mean")
        ax.plot(x, df_reward["moving_mean"], lw=2.2, label="moving mean (last 100 calls)")
    elif "reward" in df_train.columns:
        ax.plot(
            df_train["step"],
            df_train["reward"].rolling(20).mean(),
            lw=2,
            label="reward (20-step MA, fallback)",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("reward")
    ax.set_title("Reward trajectory across training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def print_summary(df_train: pd.DataFrame, df_reward: pd.DataFrame) -> None:
    if df_train.empty:
        print("[summary] no training rows parsed.")
        return

    last_step = int(df_train["step"].max())
    zero_grad_frac = float((df_train.get("grad_norm", pd.Series(dtype=float)) < 1e-3).mean())
    zero_std_frac = float(df_train.get("frac_reward_zero_std", pd.Series(dtype=float)).mean())

    final_reward_mean = df_train["reward"].tail(50).mean() if "reward" in df_train.columns else float("nan")
    if "rewards/medusa_env_reward/mean" in df_train.columns:
        env = df_train["rewards/medusa_env_reward/mean"]
        final_env = env.tail(50).mean()
        peak_env_ma = env.rolling(20).mean().max()
    else:
        final_env = float("nan")
        peak_env_ma = float("nan")

    failures_total = aggregate_failures(df_reward["failures"]) if not df_reward.empty else {}

    print("\n=== GRPO training summary =================================")
    print(f"steps observed                : {last_step}")
    print(f"frac steps with ~zero grad    : {zero_grad_frac:.1%}")
    print(f"avg frac_reward_zero_std      : {zero_std_frac:.1%}")
    print(f"final 50-step reward mean     : {final_reward_mean:.3f}")
    print(f"final 50-step env reward mean : {final_env:.3f}")
    print(f"peak env reward (20-step MA)  : {peak_env_ma:.3f}")
    print(f"total failures across run     : {failures_total or '{}'}")
    print("===========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover GRPO training stats from HF Jobs logs and plot.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--job-id",
        help="HF Jobs ID. Logs are pulled via `hf jobs logs <JOB_ID>`.",
    )
    src.add_argument(
        "--log-file",
        type=Path,
        help="Path to a previously saved log file (skip the fetch step).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("grpo_recovery"),
        help="Directory to write CSVs and plots into.",
    )
    parser.add_argument(
        "--hf-namespace",
        default=None,
        help="Pass --namespace to `hf jobs logs` (use when the job is in an org).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Retry `hf jobs logs` this many times when /whoami-v2 is rate-limited.",
    )
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=30.0,
        help="Initial sleep (seconds) between retries; grows ×1.5 up to 300s.",
    )
    parser.add_argument(
        "--reward-log-every",
        type=int,
        default=10,
        help="Match the trainer's --reward-log-every; used to map call -> step on the reward plot.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Window size for rolling-mean overlays on the panel plot.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (still produces CSVs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.job_id:
        log_path = fetch_logs(
            args.job_id,
            out_dir / "raw_logs.txt",
            namespace=args.hf_namespace,
            max_retries=args.max_retries,
            initial_backoff=args.initial_backoff,
        )
    else:
        log_path = args.log_file
        if not log_path.exists():
            raise SystemExit(f"--log-file does not exist: {log_path}")
        # Copy in for a self-contained recovery dir.
        target = out_dir / "raw_logs.txt"
        if log_path.resolve() != target.resolve():
            target.write_bytes(log_path.read_bytes())
        log_path = target
        print(f"[load] using log file {log_path} ({log_path.stat().st_size/1e6:.2f} MB)")

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    df_train = parse_train_lines(text)
    df_reward = parse_reward_lines(text)
    print(f"[parse] train rows : {len(df_train)}")
    print(f"[parse] reward rows: {len(df_reward)}")

    if df_train.empty and df_reward.empty:
        print(
            "[parse] No GRPO log lines matched. Verify the log contains `[train] step=...` "
            "or `[reward] call=...` patterns.",
            file=sys.stderr,
        )
        return 1

    save_csvs(df_train, df_reward, out_dir)

    if not args.no_plot:
        if df_train.empty:
            print("[plot] skipping panels.png (no training rows)")
        else:
            plot_panels(df_train, out_dir / "panels.png", window=args.rolling_window)
        plot_reward_trajectory(
            df_train,
            df_reward,
            out_dir / "reward_trajectory.png",
            reward_log_every=args.reward_log_every,
        )

    print_summary(df_train, df_reward)
    print(f"[done] artifacts in {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
