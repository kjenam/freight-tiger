"""Score outputs/flags.csv against the hand-labeled route-weeks in eval/labels.csv.

Reports:
  * flag precision / recall: did the rule surface the route-weeks that deserve a look?
  * review precision / recall: of those, are the "Yes" (needs human review) verdicts right?
  * verdict accuracy and note-match accuracy on the labeled rows;
  * an audit over every output row: each justified row's note must apply to the
    route, overlap the week and raise cost; each "Yes" row must have no such note;
    and the printed numbers must match a fresh recomputation from the shipments.

Usage: python eval/evaluate.py [--flags PATH] [--labels PATH]
Exit code is 1 if the audit finds a problem, 0 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from freight_watch import config  # noqa: E402
from freight_watch.load import load_shipments  # noqa: E402
from freight_watch.metrics import compute_metrics  # noqa: E402
from freight_watch.notes import load_notes, notes_by_id  # noqa: E402

FLAGS_PATH = config.OUTPUT_DIR / "flags.csv"
LABELS_PATH = ROOT / "eval" / "labels.csv"
REPORT_PATH = config.OUTPUT_DIR / "eval_report.txt"

YES = "Yes"
JUSTIFIED = "No (justified)"
NOT_FLAGGED = "Not flagged"
VERDICTS = {YES, JUSTIFIED, NOT_FLAGGED}
PCT_RE = re.compile(r"^([+-]\d+\.\d)%")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def ratio(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.2f}" if den else "n/a (0 rows)"


def score_labels(flags: pd.DataFrame, labels: pd.DataFrame) -> tuple[list[str], list[str]]:
    bad = sorted(set(labels["expected_flagged"]) - VERDICTS)
    if bad:
        raise SystemExit(f"labels file has unknown expected_flagged values: {bad}")

    out = flags.set_index(["route", "week_of"])
    merged = labels.copy()
    keys = list(zip(merged["route"], merged["week_of"]))
    merged["got_flagged"] = [out.loc[k, "flagged"] if k in out.index else NOT_FLAGGED for k in keys]
    merged["got_note_id"] = [out.loc[k, "matched_note_id"] if k in out.index else "" for k in keys]

    exp_pos = merged["expected_flagged"] != NOT_FLAGGED
    got_pos = merged["got_flagged"] != NOT_FLAGGED
    exp_yes = merged["expected_flagged"] == YES
    got_yes = merged["got_flagged"] == YES
    verdict_ok = merged["expected_flagged"] == merged["got_flagged"]

    note_rows = (merged["expected_flagged"] == JUSTIFIED) | (merged["got_flagged"] == JUSTIFIED)
    note_ok = note_rows & (merged["expected_note_id"] == merged["got_note_id"])

    lines = [
        f"Labeled route-weeks: {len(merged)} "
        f"({int(exp_yes.sum())} Yes, {int((merged['expected_flagged'] == JUSTIFIED).sum())} justified, "
        f"{int((~exp_pos).sum())} not flagged)",
        f"Flag precision (surfaced rows that deserved it): {ratio(int((exp_pos & got_pos).sum()), int(got_pos.sum()))}",
        f"Flag recall (deserving rows that were surfaced):  {ratio(int((exp_pos & got_pos).sum()), int(exp_pos.sum()))}",
        f"Review precision ('Yes' rows that are truly unexplained): {ratio(int((exp_yes & got_yes).sum()), int(got_yes.sum()))}",
        f"Review recall (unexplained rows marked 'Yes'):            {ratio(int((exp_yes & got_yes).sum()), int(exp_yes.sum()))}",
        f"Verdict accuracy (3-way exact match): {ratio(int(verdict_ok.sum()), len(merged))}",
        f"Note-match accuracy (rows justified in labels or output): {ratio(int(note_ok.sum()), int(note_rows.sum()))}",
    ]

    mismatches = []
    for _, r in merged[~verdict_ok | (note_rows & ~note_ok)].iterrows():
        expected = r["expected_flagged"] + (f" {r['expected_note_id']}" if r["expected_note_id"] else "")
        got = r["got_flagged"] + (f" {r['got_note_id']}" if r["got_note_id"] else "")
        mismatches.append(f"  {r['route']} {r['week_of']}: expected {expected}, got {got} -- {r['rationale']}")
    return lines, mismatches


def audit_notes(flags: pd.DataFrame) -> list[str]:
    notes = load_notes()
    by_id = notes_by_id(notes)
    problems = []
    for _, r in flags.iterrows():
        where = f"{r['route']} {r['week_of']}"
        note_id = r["matched_note_id"]
        valid = [n.note_id for n in notes if n.applies_to_route(r["route"]) and n.overlaps_week(r["week_of"]) and n.raises_cost]
        if r["flagged"] == JUSTIFIED:
            note = by_id.get(note_id)
            if note is None:
                problems.append(f"{where}: justified by unknown note {note_id!r}")
                continue
            if not note.applies_to_route(r["route"]):
                problems.append(f"{where}: {note_id} applies to {note.applies_to}, not this route")
            if not note.overlaps_week(r["week_of"]):
                problems.append(f"{where}: {note_id} window {note.start}..{note.end} does not overlap the week")
            if not note.raises_cost:
                problems.append(f"{where}: {note_id} does not describe a cost increase")
        elif r["flagged"] == YES:
            if note_id:
                problems.append(f"{where}: flagged Yes but matched_note_id is {note_id!r}")
            if valid:
                problems.append(f"{where}: flagged Yes but {', '.join(valid)} would justify it")
        else:
            problems.append(f"{where}: unexpected flagged value {r['flagged']!r}")
    return problems


def audit_numbers(flags: pd.DataFrame) -> list[str]:
    weekly = compute_metrics(load_shipments())
    weekly["week_of"] = weekly["week_of"].dt.strftime("%Y-%m-%d")
    truth = weekly.set_index(["route", "week_of"])
    problems = []
    for _, r in flags.iterrows():
        key = (r["route"], r["week_of"])
        where = f"{r['route']} {r['week_of']}"
        if key not in truth.index:
            problems.append(f"{where}: no such route-week in the shipments")
            continue
        t = truth.loc[key]
        if r["cost_per_tonne_km"] != f"{t['cost_per_tonne_km']:.2f}":
            problems.append(f"{where}: cost {r['cost_per_tonne_km']} != recomputed {t['cost_per_tonne_km']:.2f}")
        for col, metric in (("vs_own_history", "own_pct"), ("vs_similar_routes", "peer_pct")):
            m = PCT_RE.match(r[col])
            expected = f"{t[metric]:+.1f}"
            if m is None:
                problems.append(f"{where}: {col} {r[col]!r} does not start with a signed percentage")
            elif m.group(1) != expected:
                problems.append(f"{where}: {col} {m.group(1)}% != recomputed {expected}%")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flags", type=Path, default=FLAGS_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if not args.flags.exists():
        raise SystemExit(f"{args.flags} not found; run `python -m freight_watch.main` first")
    flags = read_csv(args.flags)
    labels = read_csv(args.labels)

    score_lines, mismatches = score_labels(flags, labels)
    note_problems = audit_notes(flags)
    number_problems = audit_numbers(flags)
    justified = int((flags["flagged"] == JUSTIFIED).sum())

    report = [
        "Evaluation report",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Flags file: {args.flags}",
        f"Labels file: {args.labels}",
        "",
        "Labeled set",
        *(f"  {line}" for line in score_lines),
    ]
    report += ["  Mismatches:", *mismatches] if mismatches else ["  Mismatches: none"]
    report += [
        "",
        f"Audit over all {len(flags)} output rows ({justified} justified, {len(flags) - justified} Yes)",
        f"  Note checks (route, window, raises_cost; no missed justification): "
        f"{'OK' if not note_problems else f'{len(note_problems)} problem(s)'}",
        *(f"    {p}" for p in note_problems),
        f"  Number checks (cost and percentages match recomputation): "
        f"{'OK' if not number_problems else f'{len(number_problems)} problem(s)'}",
        *(f"    {p}" for p in number_problems),
    ]
    audit_ok = not note_problems and not number_problems
    report += ["", f"Audit result: {'PASS' if audit_ok else 'FAIL'}"]

    text = "\n".join(report) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    print(text, end="")
    print(f"Report written to {args.report}")
    return 0 if audit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
