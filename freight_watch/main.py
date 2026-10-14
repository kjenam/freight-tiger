"""Run the pipeline end to end: shipments -> weekly metrics -> flags -> note verdicts -> reasons.

Writes ``outputs/flags.csv`` (same columns and formatting as the sample) and
``outputs/run_log.json`` (inputs, thresholds, counts, retrieval mode, token and cost log).

Usage: python -m freight_watch.main [--no-llm] [--no-retrieval]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from freight_watch import config
from freight_watch.explain import DEFAULT_MODEL, Explainer, ReasonInput, TokenTracker, make_client
from freight_watch.flagging import candidates
from freight_watch.justify import justify
from freight_watch.load import load_shipments
from freight_watch.metrics import compute_metrics
from freight_watch.notes import Note, load_notes

FLAGS_CSV = config.OUTPUT_DIR / "flags.csv"
RUN_LOG_JSON = config.OUTPUT_DIR / "run_log.json"

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]
SAMPLE_STRICT_COLUMNS = OUTPUT_COLUMNS[:-1]

DEFAULT_WINDOW_DAYS = config.DEFAULT_NOTE_WINDOW_WEEKS * 7
RETRIEVAL_K = 3


def note_record(note: Note) -> dict[str, Any]:
    """The mapping shape that justify, retrieval and explain expect."""
    return {
        "note_id": note.note_id,
        "date": note.posted.isoformat(),
        "applies_to": note.applies_to,
        "note": note.text,
        "start": note.start,
        "end": note.end,
        "raises_cost": note.raises_cost,
    }


def build_retriever(records: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any]]:
    try:
        from freight_watch.retrieval import DEFAULT_EMBED_MODEL, NoteIndex

        index = NoteIndex(records, default_window_days=DEFAULT_WINDOW_DAYS)
    except Exception as exc:  # missing chromadb / model download failure: verdicts don't need it
        return None, {"mode": "date_only", "error": f"{type(exc).__name__}: {exc}"}
    return index, {"mode": "chroma", "embed_model": DEFAULT_EMBED_MODEL, "k": RETRIEVAL_K, "notes_indexed": index.size}


def fmt_own(pct: float) -> str:
    return f"{pct:+.1f}% vs this route's past average"


def fmt_peer(pct: float) -> str:
    if pct is None or math.isnan(pct):
        return "n/a (no similar-length route this week)"
    return f"{pct:+.1f}% vs similar-length routes this week"


def build_rows(
    flagged: pd.DataFrame,
    records: list[dict[str, Any]],
    retriever: Any | None,
    explainer: Explainer,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    by_id = {r["note_id"]: r for r in records}
    rows, details = [], []
    for c in flagged.itertuples(index=False):
        week = c.week_of.date()
        verdict = justify(
            c.route,
            week,
            records,
            retriever=retriever,
            k=RETRIEVAL_K,
            default_window_days=DEFAULT_WINDOW_DAYS,
        )
        cost = round(float(c.cost_per_tonne_km), 2)
        own = round(float(c.own_pct), 1)
        peer = round(float(c.peer_pct), 1) if not math.isnan(c.peer_pct) else float("nan")

        result = explainer.reason(
            ReasonInput(
                route=c.route,
                week_of=week.isoformat(),
                cost_per_tonne_km=cost,
                vs_own_pct=own,
                vs_peer_pct=peer,
                flagged=verdict.flagged,
                matched_note=by_id.get(verdict.matched_note_id),
                closest_note=by_id.get(verdict.closest_note_id),
            )
        )
        rows.append(
            {
                "route": c.route,
                "week_of": week.isoformat(),
                "cost_per_tonne_km": f"{cost:.2f}",
                "vs_own_history": fmt_own(own),
                "vs_similar_routes": fmt_peer(peer),
                "flagged": verdict.flagged,
                "matched_note_id": verdict.matched_note_id or "",
                "reason": result.text,
            }
        )
        details.append(
            {
                "route": c.route,
                "week_of": week.isoformat(),
                "flagged": verdict.flagged,
                "matched_note_id": verdict.matched_note_id,
                "closest_note_id": verdict.closest_note_id,
                "retrieved_note_ids": list(verdict.candidate_note_ids),
                "own_history_weeks": int(c.own_history_weeks),
                "peer_count": int(c.peer_count),
                "reason_source": result.source,
                "reason_problems": result.problems,
            }
        )
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return out.sort_values(["week_of", "route"], ignore_index=True), details


def sample_check(out: pd.DataFrame, sample_path: Path = config.SAMPLE_CSV) -> dict[str, Any]:
    """Every sample row must appear in the output with identical non-reason columns."""
    if not sample_path.exists():
        return {"checked": False, "reason": f"{sample_path.name} not found"}
    # The sample's reason text contains unquoted commas, so only the leading strict columns are parsed.
    with sample_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        sample = [dict(zip(SAMPLE_STRICT_COLUMNS, line)) for line in reader if line]
    ours = out.set_index(["route", "week_of"])
    rows = []
    for s in sample:
        key = (s["route"], s["week_of"])
        if key not in ours.index:
            rows.append({"route": s["route"], "week_of": s["week_of"], "match": False, "mismatches": ["missing"]})
            continue
        got = ours.loc[key]
        mismatches = [
            {"column": col, "expected": s[col], "got": got[col]}
            for col in SAMPLE_STRICT_COLUMNS[2:]
            if got[col] != s[col]
        ]
        rows.append({"route": s["route"], "week_of": s["week_of"], "match": not mismatches, "mismatches": mismatches})
    return {"checked": True, "all_match": all(r["match"] for r in rows), "rows": rows}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(use_llm: bool = True, use_retrieval: bool = True) -> dict[str, Any]:
    shipments = load_shipments()
    weekly = compute_metrics(shipments)
    flagged = candidates(weekly)

    notes = load_notes()
    records = [note_record(n) for n in notes]
    if use_retrieval:
        retriever, retrieval_info = build_retriever(records)
    else:
        retriever, retrieval_info = None, {"mode": "date_only", "disabled_by_flag": True}

    client = make_client() if use_llm else None
    explainer = Explainer(client=client, model=DEFAULT_MODEL, tracker=TokenTracker(model=DEFAULT_MODEL))

    out, details = build_rows(flagged, records, retriever, explainer)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FLAGS_CSV, index=False)

    log = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            p.name: {"path": str(p.relative_to(config.ROOT)), "sha256": sha256(p)}
            for p in (config.SHIPMENTS_CSV, config.NOTES_CSV, config.NOTES_STRUCTURED_JSON)
        },
        "config": {
            "history_weeks": config.HISTORY_WEEKS,
            "own_rising_min_pct": config.OWN_RISING_MIN_PCT,
            "own_abnormal_pct": config.OWN_ABNORMAL_PCT,
            "peer_abnormal_pct": config.PEER_ABNORMAL_PCT,
            "default_note_window_weeks": config.DEFAULT_NOTE_WINDOW_WEEKS,
        },
        "data": {
            "shipments": len(shipments),
            "routes": int(shipments["route"].nunique()),
            "weeks": int(shipments["week_of"].nunique()),
            "first_week": shipments["week_of"].min().date().isoformat(),
            "last_week": shipments["week_of"].max().date().isoformat(),
            "route_weeks": len(weekly),
            "route_weeks_without_history": int(weekly["own_pct"].isna().sum()),
        },
        "results": {
            "rows_written": len(out),
            "flagged_yes": int((out["flagged"] == "Yes").sum()),
            "justified": int((out["flagged"] == "No (justified)").sum()),
            "matched_notes": out.loc[out["matched_note_id"] != "", "matched_note_id"].value_counts().sort_index().to_dict(),
        },
        "retrieval": retrieval_info,
        "llm": {"enabled": client is not None, **explainer.tracker.to_dict()},
        "sample_check": sample_check(out),
        "outputs": {"flags_csv": str(FLAGS_CSV.relative_to(config.ROOT))},
        "rows": details,
    }
    RUN_LOG_JSON.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    return log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-llm", action="store_true", help="use the template wording only")
    parser.add_argument("--no-retrieval", action="store_true", help="skip the Chroma vector store")
    args = parser.parse_args()

    log = run(use_llm=not args.no_llm, use_retrieval=not args.no_retrieval)
    res, llm, check = log["results"], log["llm"], log["sample_check"]
    print(
        f"Wrote {res['rows_written']} rows to {FLAGS_CSV.relative_to(config.ROOT)} "
        f"({res['flagged_yes']} flagged, {res['justified']} justified)"
    )
    print(f"Retrieval: {log['retrieval']['mode']}")
    print(
        f"LLM: {'on' if llm['enabled'] else 'off'}; {llm['calls']} calls, {llm['total_tokens']} tokens, "
        f"{llm['llm_reasons']} LLM reasons, {llm['template_reasons']} template reasons"
    )
    if check.get("checked"):
        print(f"Sample rows match: {'yes' if check['all_match'] else 'NO'}")
    print(f"Run log: {RUN_LOG_JSON.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
