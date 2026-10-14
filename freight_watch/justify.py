"""Deterministic note validator: decides whether a context note justifies a cost rise.

The verdict never depends on embeddings or an LLM. A note justifies a flagged
route-week only when all three hold:
  1. ``applies_to`` is the route itself or ``"All Routes"``;
  2. the note's effective window overlaps the week (Monday to Sunday);
  3. ``raises_cost`` is true.

Retrieval is used only to pick the "closest note" mentioned in the wording of
unexplained rows, and only among notes that already pass the route filter and
lie near the week in time.

Each note is a mapping with at least ``note_id``, ``applies_to``, ``start``,
``end`` (may be ``None`` for open-ended notes) and ``raises_cost``. ``date`` and
``note`` (the text) are carried through when present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, Mapping, Protocol, Sequence

from freight_watch import config

ALL_ROUTES = "All Routes"
JUSTIFIED = "No (justified)"
UNEXPLAINED = "Yes"

DEFAULT_WINDOW_DAYS = config.DEFAULT_NOTE_WINDOW_WEEKS * 7
DEFAULT_LOOKAROUND_DAYS = 21


class Retriever(Protocol):
    def search(
        self, route: str, week_of: date, k: int, lookaround_days: int
    ) -> list["RetrievalHit"]: ...


@dataclass(frozen=True)
class RetrievalHit:
    note_id: str
    similarity: float


@dataclass(frozen=True)
class Justification:
    flagged: str
    matched_note_id: str | None
    closest_note_id: str | None
    candidate_note_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def justified(self) -> bool:
        return self.flagged == JUSTIFIED


def to_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def week_bounds(week_of: date | datetime | str) -> tuple[date, date]:
    start = to_date(week_of)
    return start, start + timedelta(days=6)


def note_window(
    note: Mapping, default_window_days: int = DEFAULT_WINDOW_DAYS
) -> tuple[date, date]:
    """Inclusive (start, end) of the period a note is in effect.

    Open-ended notes (no ``end``) last ``default_window_days`` from their start.
    """
    start = to_date(note["start"])
    end_raw = note.get("end")
    if end_raw in (None, "") or end_raw != end_raw:  # NaN from pandas
        end = start + timedelta(days=default_window_days - 1)
    else:
        end = to_date(end_raw)
    if end < start:
        raise ValueError(f"note {note['note_id']}: end {end} is before start {start}")
    return start, end


def applies_to_route(note: Mapping, route: str) -> bool:
    return note["applies_to"] in (route, ALL_ROUTES)


def gap_days(a: tuple[date, date], b: tuple[date, date]) -> int:
    """Days between two inclusive intervals; 0 when they overlap."""
    if a[1] < b[0]:
        return (b[0] - a[1]).days
    if b[1] < a[0]:
        return (a[0] - b[1]).days
    return 0


def _specificity(note: Mapping, route: str) -> int:
    return 0 if note["applies_to"] == route else 1


def matching_notes(
    route: str,
    week_of: date | datetime | str,
    notes: Iterable[Mapping],
    default_window_days: int = DEFAULT_WINDOW_DAYS,
) -> list[Mapping]:
    """Notes passing all three validator checks, best first.

    Ordering: exact route before "All Routes", then the note whose start date is
    closest to the week's Monday, then ``note_id`` so ties are deterministic.
    """
    week = week_bounds(week_of)
    passing = [
        n
        for n in notes
        if applies_to_route(n, route)
        and bool(n["raises_cost"])
        and gap_days(note_window(n, default_window_days), week) == 0
    ]
    return sorted(
        passing,
        key=lambda n: (
            _specificity(n, route),
            abs((to_date(n["start"]) - week[0]).days),
            n["note_id"],
        ),
    )


def closest_note(
    route: str,
    week_of: date | datetime | str,
    notes: Iterable[Mapping],
    hits: Sequence[RetrievalHit] | None = None,
    default_window_days: int = DEFAULT_WINDOW_DAYS,
    lookaround_days: int = DEFAULT_LOOKAROUND_DAYS,
) -> Mapping | None:
    """Nearest-in-time note for this route within ``lookaround_days`` of the week.

    When retrieval ``hits`` are given, only those notes are considered and their
    similarity breaks ties after date gap and specificity. Without hits, every
    note is considered (useful for tests and when the vector store is absent).
    """
    week = week_bounds(week_of)
    by_id = {n["note_id"]: n for n in notes}
    similarity = {h.note_id: h.similarity for h in hits} if hits is not None else {}
    pool = [by_id[i] for i in similarity if i in by_id] if hits is not None else list(by_id.values())

    scored = []
    for n in pool:
        if not applies_to_route(n, route):
            continue
        gap = gap_days(note_window(n, default_window_days), week)
        if gap > lookaround_days:
            continue
        scored.append(
            ((gap, _specificity(n, route), -similarity.get(n["note_id"], 0.0), n["note_id"]), n)
        )
    if not scored:
        return None
    return min(scored, key=lambda pair: pair[0])[1]


def justify(
    route: str,
    week_of: date | datetime | str,
    notes: Sequence[Mapping],
    retriever: Retriever | None = None,
    k: int = 3,
    default_window_days: int = DEFAULT_WINDOW_DAYS,
    lookaround_days: int = DEFAULT_LOOKAROUND_DAYS,
) -> Justification:
    """Verdict for one flagged route-week."""
    matches = matching_notes(route, week_of, notes, default_window_days)
    hits = (
        retriever.search(route, to_date(week_of), k=k, lookaround_days=lookaround_days)
        if retriever is not None
        else None
    )
    candidates = tuple(h.note_id for h in hits) if hits is not None else ()

    if matches:
        return Justification(
            flagged=JUSTIFIED,
            matched_note_id=matches[0]["note_id"],
            closest_note_id=matches[0]["note_id"],
            candidate_note_ids=candidates,
        )

    near = closest_note(
        route, week_of, notes, hits, default_window_days, lookaround_days
    )
    return Justification(
        flagged=UNEXPLAINED,
        matched_note_id=None,
        closest_note_id=near["note_id"] if near is not None else None,
        candidate_note_ids=candidates,
    )
