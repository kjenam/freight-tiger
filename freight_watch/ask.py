"""Ask questions about the flagged weeks, answered only from outputs/flags.csv and the context notes.

Usage:
    python -m freight_watch.ask "why did Chennai-Bangalore get pricier in March 2025?"
    python -m freight_watch.ask --show-context "which weeks were unexplained?"

Retrieval is hybrid: route names and months/dates in the question are parsed
deterministically and used as hard filters, then Chroma (all-MiniLM-L6-v2) ranks the
remaining flag rows and notes by meaning. The LLM sees only the retrieved records and
must cite them; an answer that cites a note id or date not in its context is replaced
by a plain listing of the evidence.
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from freight_watch import config
from freight_watch.explain import DEFAULT_MODEL, NOTE_ID_RE, TokenTracker, make_client, model_options

FLAGS_CSV = config.OUTPUT_DIR / "flags.csv"
TOP_K = 6
MAX_CONTEXT = 25
MAX_ANSWER_TOKENS = 600
# A semantic hit is kept only if its cosine distance is below MAX_DISTANCE and within
# RELATIVE_GAP of the best hit. Measured with all-MiniLM-L6-v2 on this data: on-topic
# hits sit around 0.27-0.46, and an unrelated question's best hit is around 0.87.
MAX_DISTANCE = 0.70
RELATIVE_GAP = 0.15
ALL_ROUTES = "All Routes"
JUSTIFIED = "No (justified)"
UNEXPLAINED = "Yes"
NO_ANSWER = "I couldn't find anything in the flagged weeks or context notes that answers this."

UNEXPLAINED_RE = re.compile(
    r"\b(unexplained|unjustified|no (explanation|justification|reason|matching note)|"
    r"not (been )?(justified|explained)|need(s|ed)? (a )?review|flagged for review|"
    r"without (a |any )?(reason|explanation|note))\b",
    re.I,
)
JUSTIFIED_RE = re.compile(r"\b(justified|explained|had (a |an )?(reason|explanation)|matched (a )?note)\b", re.I)

ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
YEAR_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)")
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9
MONTH_YEAR_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?,?\s+(\d{4})\b", re.I)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
MONTH_DAY_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})\b(?!,?\s*\d{4})", re.I)


@dataclass
class Record:
    doc_id: str
    kind: str  # "flag" or "note"
    route: str
    day: str  # week_of for flags, posted date for notes
    text: str
    note_id: str = ""
    verdict: str = ""  # flags only: the `flagged` column


@dataclass
class Query:
    routes: list[str] = field(default_factory=list)
    periods: list[tuple[date, date]] = field(default_factory=list)
    verdict: str = ""


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_records() -> list[Record]:
    if not FLAGS_CSV.exists():
        sys.exit(f"{FLAGS_CSV.relative_to(config.ROOT)} not found; run `python -m freight_watch.main` first.")
    flags = pd.read_csv(FLAGS_CSV, dtype=str, keep_default_na=False)
    notes = pd.read_csv(config.NOTES_CSV, dtype=str, keep_default_na=False)

    records = []
    for r in flags.itertuples(index=False):
        note = f" Matched note: {r.matched_note_id}." if r.matched_note_id else ""
        records.append(Record(
            doc_id=f"flag:{r.route}:{r.week_of}",
            kind="flag",
            route=r.route,
            day=r.week_of,
            note_id=r.matched_note_id,
            verdict=r.flagged,
            text=(
                f"Flagged week: {r.route}, week of {r.week_of}. Cost {r.cost_per_tonne_km} INR per tonne-km, "
                f"{r.vs_own_history}, {r.vs_similar_routes}. Verdict: {r.flagged}.{note} Reason: {r.reason}"
            ),
        ))
    for n in notes.itertuples(index=False):
        records.append(Record(
            doc_id=f"note:{n.note_id}",
            kind="note",
            route=n.applies_to,
            day=n.date,
            note_id=n.note_id,
            text=f"Context note {n.note_id} dated {n.date}, applies to {n.applies_to}: {n.note}",
        ))
    return records


def parse_query(question: str, routes: list[str]) -> Query:
    """Pull route names, date ranges and a verdict out of the question; no model involved."""
    q = Query()
    if UNEXPLAINED_RE.search(question):
        q.verdict = UNEXPLAINED
    elif JUSTIFIED_RE.search(question):
        q.verdict = JUSTIFIED
    norm_q = f" {_norm(question)} "
    q.routes = [r for r in routes if f" {_norm(r)} " in norm_q]
    if not q.routes:
        city_routes: dict[str, set[str]] = {}
        for r in routes:
            for city in r.split("-"):
                city_routes.setdefault(_norm(city), set()).add(r)
        for city, rs in city_routes.items():
            if f" {city} " in norm_q:
                q.routes.extend(sorted(rs - set(q.routes)))

    taken: list[tuple[int, int]] = []

    def free(m: re.Match) -> bool:
        if any(m.start() < e and s < m.end() for s, e in taken):
            return False
        taken.append(m.span())
        return True

    for m in ISO_DATE_RE.finditer(question):
        if free(m):
            day = date(int(m[1]), int(m[2]), int(m[3]))
            monday = day - timedelta(days=day.weekday())
            q.periods.append((monday, monday + timedelta(days=6)))
    for m in YEAR_MONTH_RE.finditer(question):
        if free(m) and 1 <= int(m[2]) <= 12:
            q.periods.append(_month(int(m[1]), int(m[2])))
    for m in MONTH_YEAR_RE.finditer(question):
        if free(m):
            q.periods.append(_month(int(m[2]), MONTHS[m[1].lower()]))
    for m in YEAR_RE.finditer(question):
        if free(m):
            y = int(m[1])
            q.periods.append((date(y, 1, 1), date(y, 12, 31)))
    return q


def _month(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def structured_matches(records: list[Record], q: Query) -> list[Record]:
    """Flag rows that satisfy every filter the question named, oldest first."""
    if not (q.routes or q.periods or q.verdict):
        return []
    # "Was the Mumbai-Delhi rise explained?" must still see an unexplained row, so the
    # verdict only filters when nothing narrower was named.
    by_verdict = q.verdict and not (q.routes or q.periods)
    hits = []
    for rec in records:
        if rec.kind != "flag":
            continue
        if by_verdict and rec.verdict != q.verdict:
            continue
        if q.routes and rec.route not in q.routes:
            continue
        if q.periods:
            # A week counts if any of its seven days falls in a requested period.
            start = date.fromisoformat(rec.day)
            end = start + timedelta(days=6)
            if not any(start <= p_end and p_start <= end for p_start, p_end in q.periods):
                continue
        hits.append(rec)
    return sorted(hits, key=lambda r: (r.day, r.route))


def semantic_matches(records: list[Record], question: str, q: Query, k: int) -> list[tuple[Record, float]]:
    from freight_watch.retrieval import embedding_function, ephemeral_client

    collection = ephemeral_client().create_collection(
        name=f"ask-{uuid.uuid4().hex[:12]}",
        embedding_function=embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[r.doc_id for r in records],
        documents=[r.text for r in records],
        metadatas=[{"kind": r.kind, "route": r.route} for r in records],
    )
    where = {"route": {"$in": q.routes + [ALL_ROUTES]}} if q.routes else None
    res = collection.query(query_texts=[question], n_results=min(k, len(records)), where=where)
    by_id = {r.doc_id: r for r in records}
    return [(by_id[i], d) for i, d in zip(res["ids"][0], res["distances"][0])]


def retrieve(records: list[Record], question: str, k: int = TOP_K) -> list[Record]:
    routes = sorted({r.route for r in records if r.route != ALL_ROUTES})
    q = parse_query(question, routes)
    exact = structured_matches(records, q)
    hits = semantic_matches(records, question, q, k)
    best = min((d for _, d in hits), default=0.0)
    semantic = [r for r, d in hits if d <= MAX_DISTANCE and d <= best + RELATIVE_GAP]

    picked, seen = [], set()
    for rec in exact + semantic:
        if rec.doc_id not in seen:
            seen.add(rec.doc_id)
            picked.append(rec)
    # Bring in every note a flag row mentions (matched or closest) so the LLM can quote it.
    notes_by_id = {r.note_id: r for r in records if r.kind == "note"}
    for rec in [r for r in picked if r.kind == "flag"]:
        for note_id in NOTE_ID_RE.findall(rec.text):
            note = notes_by_id.get(note_id)
            if note and note.doc_id not in seen:
                seen.add(note.doc_id)
                picked.append(note)
    return picked[:MAX_CONTEXT]


SYSTEM_PROMPT = f"""You answer questions about freight cost flags for a logistics team.
Use ONLY the records in the context; each starts with its citation in brackets. Do not use outside knowledge or guess.
Cite every fact: flagged weeks as [route week_of] (e.g. [Delhi-Jaipur 2024-11-11]) and
notes by id (e.g. [N001]). Only cite ids and dates that appear in the context.
Write dates exactly as YYYY-MM-DD with plain hyphens.
A verdict of "No (justified)" means a context note explains the rise; "Yes" means the
rise is unexplained and needs human review. Keep the answer under 120 words.
If the context does not answer the question, reply exactly: {NO_ANSWER}"""


def citation(rec: Record) -> str:
    return f"[{rec.route} {rec.day}]" if rec.kind == "flag" else f"[{rec.note_id}]"


def build_prompt(question: str, context: list[Record]) -> str:
    lines = [f"{citation(rec)} {rec.text}" for rec in context]
    return "Context:\n" + "\n".join(lines) + f"\n\nQuestion: {question}"


def normalize_dashes(text: str) -> str:
    """Models often emit non-breaking or en dashes, which would slip past the date check."""
    return re.sub(r"[\u2010-\u2015\u2212]", "-", text)


def grounding_problems(answer: str, context: list[Record]) -> list[str]:
    """Note ids and ISO dates the answer cites that are not in its context."""
    allowed_ids, allowed_dates = set(), set()
    for rec in context:
        allowed_ids.update(NOTE_ID_RE.findall(rec.text))
        allowed_dates.add(rec.day)
        allowed_dates.update(m[0] for m in ISO_DATE_RE.finditer(rec.text))
        # "Feb 24 to Mar 8" in a note means those days in the note's own year.
        for m in MONTH_DAY_RE.finditer(rec.text):
            try:
                allowed_dates.add(date(int(rec.day[:4]), MONTHS[m[1].lower()], int(m[2])).isoformat())
            except ValueError:
                pass
    problems = [f"unknown note {n}" for n in sorted(set(NOTE_ID_RE.findall(answer)) - allowed_ids)]
    problems += [f"unknown date {d}" for d in sorted({m[0] for m in ISO_DATE_RE.finditer(answer)} - allowed_dates)]
    return problems


def fallback_answer(context: list[Record]) -> str:
    if not context:
        return NO_ANSWER
    lines = ["Here is what the flagged weeks and notes say:"]
    for rec in context:
        if rec.kind == "flag":
            lines.append(f"- [{rec.route} {rec.day}] {rec.text.split('Verdict: ', 1)[1]}")
        else:
            lines.append(f"- [{rec.note_id}] {rec.text.split(': ', 1)[1]}")
    return "\n".join(lines)


def answer(question: str, k: int = TOP_K, client=None, tracker: TokenTracker | None = None):
    """Return (answer, context, mode), where mode says who wrote the answer."""
    tracker = tracker or TokenTracker(model=DEFAULT_MODEL)
    context = retrieve(load_records(), question, k)
    if not context:
        return NO_ANSWER, context, "no relevant records"
    client = client or make_client()
    if client is None:
        return fallback_answer(context), context, "template (no GROQ_API_KEY)"
    try:
        resp = client.chat.completions.create(
            model=tracker.model,
            temperature=0,
            seed=0,
            max_tokens=MAX_ANSWER_TOKENS,
            **model_options(tracker.model),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, context)},
            ],
        )
    except Exception as exc:  # network, auth, rate limit: still show the evidence
        return fallback_answer(context), context, f"template (API error: {type(exc).__name__})"
    tracker.add_usage(getattr(resp, "usage", None))
    text = normalize_dashes(resp.choices[0].message.content or "").strip()
    problems = grounding_problems(text, context) if text else ["empty answer"]
    if problems:
        return fallback_answer(context), context, f"template (LLM answer rejected: {', '.join(problems)})"
    return text, context, f"llm ({tracker.model})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=TOP_K, help="semantic hits to retrieve (default %(default)s)")
    parser.add_argument("--show-context", action="store_true", help="print the records given to the LLM")
    args = parser.parse_args()

    tracker = TokenTracker(model=DEFAULT_MODEL)
    text, context, mode = answer(args.question, args.k, tracker=tracker)
    print(text)
    print()
    if context:
        sources = [f"{r.route} {r.day}" if r.kind == "flag" else r.note_id for r in context]
        print(f"Sources: {'; '.join(sources)}")
    if args.show_context:
        print("\nContext given to the model:")
        print(build_prompt(args.question, context))
    print(
        f"Answered by: {mode} | tokens in/out: {tracker.prompt_tokens}/{tracker.completion_tokens} "
        f"| est. cost ${tracker.cost_usd() or 0:.6f}"
    )


if __name__ == "__main__":
    main()
