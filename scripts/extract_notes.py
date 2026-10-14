"""One-time LLM pass that turns each free-text context note into structured fields.

The output is a *draft*. A human reviews it, fixes anything wrong, and commits the
result as data/notes_structured.json. The pipeline only ever reads the reviewed file,
so the LLM never decides at run time whether a note justifies a cost rise.

Usage:
    python scripts/extract_notes.py              # writes data/notes_structured.draft.json
    python scripts/extract_notes.py --diff       # also prints differences vs the reviewed file
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freight_watch.config import DATA_DIR, NOTES_CSV, ROOT  # noqa: E402
from freight_watch.config import NOTES_STRUCTURED_JSON as REVIEWED_JSON  # noqa: E402

DRAFT_JSON = DATA_DIR / "notes_structured.draft.json"
MODEL = "openai/gpt-oss-20b"
FIELDS = ("note_id", "applies_to", "start", "end", "raises_cost")

SYSTEM_PROMPT = """You extract structured facts from short freight-logistics notes.
Return ONLY a JSON object with exactly these keys:
  "start": ISO date (YYYY-MM-DD) when the described condition began. Use the note's
           posted date unless the text gives an explicit earlier/later start.
  "end":   ISO date (YYYY-MM-DD) of the last day the condition applied, or null if the
           text gives no end. "Roads were passable again by Mar 9" means end = Mar 8.
           "a week" / "festival week" starting on the posted date means end = posted date + 6 days.
  "raises_cost": true only if the note says freight/transport costs on the stated
           route(s) went UP because of the condition. false if costs were unaffected,
           absorbed, the route is outside the dataset, conditions were normal/improved,
           or the note is purely informational.
Infer missing years from the posted date. Do not add any other keys."""


def build_user_prompt(row: dict) -> str:
    return (
        f"Posted date: {row['date']}\n"
        f"Applies to: {row['applies_to']}\n"
        f"Note: {row['note']}"
    )


def read_notes() -> list[dict]:
    with NOTES_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate(row: dict, raw: dict) -> dict:
    """Coerce the LLM's answer into the committed schema; route and id always come from the CSV."""
    def iso_or_none(value):
        if value in (None, "", "null"):
            return None
        return date.fromisoformat(str(value)).isoformat()

    start = iso_or_none(raw.get("start")) or row["date"]
    end = iso_or_none(raw.get("end"))
    if end is not None and end < start:
        raise ValueError(f"{row['note_id']}: end {end} is before start {start}")
    raises_cost = raw.get("raises_cost")
    if not isinstance(raises_cost, bool):
        raise ValueError(f"{row['note_id']}: raises_cost must be a boolean, got {raises_cost!r}")
    return {
        "note_id": row["note_id"],
        "applies_to": row["applies_to"],
        "start": start,
        "end": end,
        "raises_cost": raises_cost,
    }


def extract(rows: list[dict]) -> tuple[list[dict], dict]:
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(ROOT / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set; add it to .env or the environment.")
    client = Groq()

    results, usage = [], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for row in rows:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ],
        )
        usage["calls"] += 1
        usage["prompt_tokens"] += resp.usage.prompt_tokens
        usage["completion_tokens"] += resp.usage.completion_tokens
        results.append(validate(row, json.loads(resp.choices[0].message.content)))
    return results, usage


def diff_against_reviewed(draft: list[dict]) -> int:
    if not REVIEWED_JSON.exists():
        print(f"No reviewed file at {REVIEWED_JSON.relative_to(ROOT)} yet.")
        return 0
    reviewed = {n["note_id"]: n for n in json.loads(REVIEWED_JSON.read_text())["notes"]}
    changes = 0
    for note in draft:
        old = reviewed.get(note["note_id"], {})
        for field in FIELDS:
            if old.get(field) != note[field]:
                changes += 1
                print(f"{note['note_id']}.{field}: reviewed={old.get(field)!r} draft={note[field]!r}")
    print(f"{changes} field(s) differ from the reviewed file.")
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diff", action="store_true", help="compare the draft with the reviewed file")
    args = parser.parse_args()

    draft, usage = extract(read_notes())
    DRAFT_JSON.write_text(json.dumps({"model": MODEL, "notes": draft}, indent=2) + "\n")
    print(f"Wrote {len(draft)} notes to {DRAFT_JSON.relative_to(ROOT)}")
    print(f"Token usage: {usage}")
    if args.diff:
        diff_against_reviewed(draft)


if __name__ == "__main__":
    main()
