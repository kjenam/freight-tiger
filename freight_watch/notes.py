import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from freight_watch import config

ALL_ROUTES = "All Routes"
REQUIRED_COLUMNS = ["note_id", "date", "applies_to", "note"]


@dataclass(frozen=True)
class Note:
    note_id: str
    posted: date
    applies_to: str
    text: str
    start: date
    end: date
    end_is_default: bool
    raises_cost: bool

    def applies_to_route(self, route: str) -> bool:
        return self.applies_to in (route, ALL_ROUTES)

    def overlaps_week(self, week_of) -> bool:
        """True if the note's [start, end] window touches the Monday-to-Sunday week starting at week_of."""
        monday = pd.Timestamp(week_of).date()
        return self.start <= monday + timedelta(days=6) and self.end >= monday


def _to_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an ISO date: {value!r}") from exc


def load_notes(
    csv_path: Path = config.NOTES_CSV,
    structured_path: Path = config.NOTES_STRUCTURED_JSON,
    default_window_weeks: int = config.DEFAULT_NOTE_WINDOW_WEEKS,
) -> list[Note]:
    """Join the raw notes with their reviewed structured fields; every note must appear in both."""
    raw = pd.read_csv(csv_path, dtype=str)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"notes file is missing columns: {missing}")
    if raw["note_id"].duplicated().any():
        raise ValueError("notes file has duplicate note_id values")

    structured = {n["note_id"]: n for n in json.loads(Path(structured_path).read_text())["notes"]}
    csv_ids, json_ids = set(raw["note_id"]), set(structured)
    if csv_ids != json_ids:
        raise ValueError(
            f"structured notes out of sync with CSV: missing={sorted(csv_ids - json_ids)} "
            f"extra={sorted(json_ids - csv_ids)}"
        )

    notes = []
    for row in raw.itertuples(index=False):
        s = structured[row.note_id]
        if s["applies_to"] != row.applies_to:
            raise ValueError(f"{row.note_id}: applies_to {s['applies_to']!r} != CSV {row.applies_to!r}")
        if not isinstance(s["raises_cost"], bool):
            raise ValueError(f"{row.note_id}: raises_cost must be true or false")

        start = _to_date(s["start"], f"{row.note_id}.start")
        end_is_default = s["end"] is None
        if end_is_default:
            end = start + timedelta(weeks=default_window_weeks, days=-1)
        else:
            end = _to_date(s["end"], f"{row.note_id}.end")
        if end < start:
            raise ValueError(f"{row.note_id}: end {end} is before start {start}")

        notes.append(
            Note(
                note_id=row.note_id,
                posted=_to_date(row.date, f"{row.note_id}.date"),
                applies_to=row.applies_to,
                text=row.note,
                start=start,
                end=end,
                end_is_default=end_is_default,
                raises_cost=s["raises_cost"],
            )
        )
    return sorted(notes, key=lambda n: n.note_id)


def notes_by_id(notes: list[Note]) -> dict[str, Note]:
    return {n.note_id: n for n in notes}
