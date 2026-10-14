"""Run the pipeline several times and check that its output is reproducible.

Every column except `reason` must be byte-identical across runs, because the
verdict and the numbers come from deterministic code. The `reason` column is
LLM-written, so differences there are reported but do not fail the check.

Usage: python scripts/repro_check.py [--runs N] [-- MAIN_ARGS...]
  e.g. python scripts/repro_check.py -- --no-llm
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FLAGS_PATH = ROOT / "outputs" / "flags.csv"
RUN_LOG_PATH = ROOT / "outputs" / "run_log.json"
REPORT_PATH = ROOT / "outputs" / "repro_report.txt"
KEY_COLS = ["route", "week_of"]
FREE_TEXT_COLS = ["reason"]


def run_pipeline(run_no: int, snapshot_dir: Path, main_args: list[str]) -> tuple[Path, dict]:
    result = subprocess.run(
        [sys.executable, "-m", "freight_watch.main", *main_args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"Run {run_no} failed with exit code {result.returncode}")
    if not FLAGS_PATH.exists():
        raise SystemExit(f"Run {run_no} did not write {FLAGS_PATH}")
    snapshot = snapshot_dir / f"flags_run{run_no}.csv"
    shutil.copy(FLAGS_PATH, snapshot)
    run_log = json.loads(RUN_LOG_PATH.read_text()) if RUN_LOG_PATH.exists() else {}
    return snapshot, run_log


def describe_run(run_no: int, log: dict) -> str:
    llm = log.get("llm", {})
    retrieval = log.get("retrieval", {}).get("mode", "unknown")
    if not llm.get("enabled"):
        llm_text = "LLM off (template reasons)"
    else:
        llm_text = (
            f"LLM on, {llm.get('calls', 0)} calls, {llm.get('total_tokens', 0)} tokens, "
            f"{llm.get('llm_reasons', 0)} LLM / {llm.get('template_reasons', 0)} template reasons"
        )
    return f"  run {run_no}: retrieval={retrieval}; {llm_text}"


def read_flags(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def diff_strict(base: pd.DataFrame, other: pd.DataFrame, base_path: Path, other_path: Path) -> list[str]:
    strict_cols = [c for c in base.columns if c not in FREE_TEXT_COLS]
    if list(base.columns) != list(other.columns):
        return [f"column mismatch: {list(base.columns)} vs {list(other.columns)}"]
    a = base[strict_cols].to_csv(index=False).splitlines()
    b = other[strict_cols].to_csv(index=False).splitlines()
    return list(difflib.unified_diff(a, b, base_path.name, other_path.name, lineterm=""))


def diff_free_text(base: pd.DataFrame, other: pd.DataFrame, run_no: int) -> list[str]:
    merged = base.merge(other, on=KEY_COLS, how="inner", suffixes=("_base", "_other"))
    lines = []
    for col in FREE_TEXT_COLS:
        changed = merged[merged[f"{col}_base"] != merged[f"{col}_other"]]
        for _, row in changed.iterrows():
            lines.append(f"  [{row['route']} {row['week_of']}] {col}")
            lines.append(f"    run 1: {row[f'{col}_base']}")
            lines.append(f"    run {run_no}: {row[f'{col}_other']}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("main_args", nargs="*", help="extra arguments for freight_watch.main, after --")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")

    with tempfile.TemporaryDirectory() as tmp:
        results = [run_pipeline(i, Path(tmp), args.main_args) for i in range(1, args.runs + 1)]
        snapshots = [snapshot for snapshot, _ in results]
        frames = [read_flags(p) for p in snapshots]

        base = frames[0]
        strict_cols = [c for c in base.columns if c not in FREE_TEXT_COLS]
        command = " ".join(["python -m freight_watch.main", *args.main_args])
        report = [
            "Reproducibility check",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"Runs: {args.runs} x `{command}`",
            f"Rows per run: {', '.join(str(len(f)) for f in frames)}",
            f"Strict columns (must be identical): {', '.join(strict_cols)}",
            f"Free-text columns (differences reported only): {', '.join(FREE_TEXT_COLS)}",
            "",
            "Run settings:",
            *(describe_run(i, log) for i, (_, log) in enumerate(results, start=1)),
            "",
        ]

        strict_ok = True
        free_text_diffs = 0
        for run_no, (frame, path) in enumerate(zip(frames[1:], snapshots[1:]), start=2):
            strict = diff_strict(base, frame, snapshots[0], path)
            free = diff_free_text(base, frame, run_no)
            free_text_diffs += len(free) // 3
            report.append(f"Run 1 vs run {run_no}:")
            if strict:
                strict_ok = False
                report.append("  strict columns: DIFFERENT")
                report.extend(f"    {line}" for line in strict)
            else:
                report.append("  strict columns: identical")
            if free:
                report.append(f"  reason text: {len(free) // 3} row(s) differ")
                report.extend(free)
            else:
                report.append("  reason text: identical")
            report.append("")

    verdict = "PASS" if strict_ok else "FAIL"
    report.append(
        f"Result: {verdict} (strict columns {'identical' if strict_ok else 'differ'} "
        f"across all runs; {free_text_diffs} reason-text difference(s))"
    )

    text = "\n".join(report) + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(text, end="")
    print(f"Report written to {REPORT_PATH.relative_to(ROOT)}")
    return 0 if strict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
