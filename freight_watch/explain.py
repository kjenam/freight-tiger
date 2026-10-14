"""Turn a deterministic verdict into a one-line human-readable reason.

The verdict (`flagged`, `matched_note_id`) and every number are decided
upstream. This module only words them: an LLM call at temperature 0 when a
Groq key is available, checked against the facts it was given, with a fixed
template as the fallback.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

DEFAULT_MODEL = "openai/gpt-oss-20b"
JUSTIFIED = "No (justified)"
UNEXPLAINED = "Yes"
MAX_REASON_CHARS = 400

# USD per 1M tokens (input, output), from https://groq.com/pricing
PRICING_USD_PER_M = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
}

NOTE_ID_RE = re.compile(r"\bN\d{3}\b")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

CLAIMS_JUSTIFICATION = [
    r"\bmatches note\b",
    r"\bclear explanation\b",
    r"\bis justified\b",
    r"\bjustified by\b",
    r"\bjustifies\b",
    r"\bexplained by\b",
    r"\bexplains the\b",
    r"\baccounts for\b",
]
DENIES_JUSTIFICATION = [
    r"\bno matching note\b",
    r"\bunexplained\b",
    r"\bnot justified\b",
    r"\bno (genuine )?justification\b",
    r"\bworth a human review\b",
]
REVIEW_SIGNAL = [r"\breview\b", r"\bunexplained\b", r"\bflagged\b"]

SYSTEM_PROMPT = """You write the `reason` column of a freight cost alert report.
A verdict has already been decided. Your only job is to word it in one or two plain sentences.

Rules:
- Use only the facts given. Do not invent causes, numbers, dates or note ids.
- If the verdict is JUSTIFIED: start with "Matches note <id> dated <date>:", briefly paraphrase that note, and end with "The cost rise has a clear explanation."
- If the verdict is UNEXPLAINED and a closest note is given: say what that note is about and that it does not describe a reason for a cost rise on this route, then end with "No genuine justification found; flagged for review."
- If the verdict is UNEXPLAINED and no note is given: say "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."
- Never mention any note id other than the one given.
- Plain text only: no markdown, no quotes around the answer, no line breaks.

Examples:
Matches note N002 dated 2025-01-20: a regional festival week drove a temporary surcharge on this corridor. The cost rise has a clear explanation.
The closest note (N006, 2025-09-22) mentions stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route. No genuine justification found; flagged for review."""


@dataclass
class ReasonInput:
    route: str
    week_of: str
    cost_per_tonne_km: float
    vs_own_pct: float
    vs_peer_pct: float
    flagged: str
    matched_note: Mapping[str, Any] | None = None
    closest_note: Mapping[str, Any] | None = None

    @property
    def note(self) -> Mapping[str, Any] | None:
        return self.matched_note if self.flagged == JUSTIFIED else self.closest_note


@dataclass
class ReasonResult:
    text: str
    source: str  # "llm" or "template"
    problems: list[str] = field(default_factory=list)


@dataclass
class TokenTracker:
    model: str = DEFAULT_MODEL
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_reasons: int = 0
    template_reasons: int = 0
    fallback_causes: Counter = field(default_factory=Counter)

    def add_usage(self, usage: Any) -> None:
        self.calls += 1
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def record(self, result: ReasonResult, fallback_cause: str | None = None) -> None:
        if result.source == "llm":
            self.llm_reasons += 1
        else:
            self.template_reasons += 1
            if fallback_cause:
                self.fallback_causes[fallback_cause] += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self) -> float | None:
        rates = PRICING_USD_PER_M.get(self.model)
        if rates is None:
            return None
        in_rate, out_rate = rates
        return (self.prompt_tokens * in_rate + self.completion_tokens * out_rate) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        rates = PRICING_USD_PER_M.get(self.model)
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usd_per_million_tokens": {"input": rates[0], "output": rates[1]} if rates else None,
            "cost_usd": self.cost_usd(),
            "llm_reasons": self.llm_reasons,
            "template_reasons": self.template_reasons,
            "fallback_causes": dict(self.fallback_causes),
        }


def _note_sentence(text: str) -> str:
    text = " ".join(str(text).split()).rstrip(".")
    if len(text) > 1 and text[0].isupper() and not text[1].isupper():
        text = text[0].lower() + text[1:]
    return text


def template_reason(inp: ReasonInput) -> str:
    note = inp.note
    if inp.flagged == JUSTIFIED:
        if note is None:
            raise ValueError(f"{inp.route} {inp.week_of}: justified row has no matched note")
        return (
            f"Matches note {note['note_id']} dated {note['date']}: {_note_sentence(note['note'])}. "
            "The cost rise has a clear explanation."
        )
    if note is None:
        return (
            "No matching note found for this route or date range. "
            "Cost rise looks unexplained and worth a human review."
        )
    return (
        f"The closest note ({note['note_id']}, {note['date']}) says {_note_sentence(note['note'])} "
        "-- it does not describe a reason for a cost rise on this route. "
        "No genuine justification found; flagged for review."
    )


def build_prompt(inp: ReasonInput) -> str:
    verdict = "JUSTIFIED" if inp.flagged == JUSTIFIED else "UNEXPLAINED"
    lines = [
        f"Route: {inp.route}",
        f"Week of: {inp.week_of}",
        f"Cost per tonne-km: {inp.cost_per_tonne_km:.2f} INR",
        f"Vs this route's past average: {inp.vs_own_pct:+.1f}%",
        f"Vs similar-length routes this week: {inp.vs_peer_pct:+.1f}%",
        f"Verdict: {verdict}",
    ]
    note = inp.note
    if note is None:
        lines.append("Note: none")
    else:
        label = "Matched note" if verdict == "JUSTIFIED" else "Closest note (does NOT justify the rise)"
        lines.append(f"{label}: {note['note_id']} dated {note['date']} for {note['applies_to']}: {note['note']}")
    lines.append("Write the reason.")
    return "\n".join(lines)


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def grounding_problems(reason: str, inp: ReasonInput, prompt: str) -> list[str]:
    """Return why an LLM reason cannot be trusted; an empty list means it passes."""
    problems = []
    if not reason:
        return ["empty"]
    if len(reason) > MAX_REASON_CHARS:
        problems.append("too_long")

    note = inp.note
    allowed_ids = {note["note_id"]} if note is not None else set()
    cited = set(NOTE_ID_RE.findall(reason))
    if cited - allowed_ids:
        problems.append("cites_other_note")

    if inp.flagged == JUSTIFIED:
        if note is None or note["note_id"] not in cited:
            problems.append("missing_matched_note")
        if _matches_any(reason, DENIES_JUSTIFICATION):
            problems.append("denies_justification")
    else:
        if _matches_any(reason, CLAIMS_JUSTIFICATION):
            problems.append("claims_justification")
        if not _matches_any(reason, REVIEW_SIGNAL):
            problems.append("missing_review_signal")

    allowed_numbers = set(NUMBER_RE.findall(prompt))
    if set(NUMBER_RE.findall(reason)) - allowed_numbers:
        problems.append("unknown_number")
    return problems


def _clean(text: str) -> str:
    text = " ".join(text.split())
    return text.strip().strip('"').strip("'").strip()


def model_options(model: str) -> dict[str, Any]:
    """Extra request options a model needs. gpt-oss reasons before answering, and that
    reasoning counts toward max_tokens, so keep it short."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {}


def make_client() -> Any | None:
    """Return a Groq client, or None when no key is configured."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("GROQ_API_KEY"):
        return None
    from groq import Groq

    return Groq()


class Explainer:
    def __init__(
        self,
        client: Any | None = None,
        model: str = DEFAULT_MODEL,
        tracker: TokenTracker | None = None,
        max_tokens: int = 400,
    ):
        self.client = client
        self.model = model
        self.tracker = tracker or TokenTracker(model=model)
        self.max_tokens = max_tokens

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            seed=0,
            max_tokens=self.max_tokens,
            **model_options(self.model),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        self.tracker.add_usage(getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    def reason(self, inp: ReasonInput) -> ReasonResult:
        if self.client is None:
            result = ReasonResult(template_reason(inp), "template", ["no_client"])
            self.tracker.record(result, "no_client")
            return result

        prompt = build_prompt(inp)
        try:
            text = _clean(self._call_llm(prompt))
        except Exception as exc:  # network, auth, rate limit: never fail the run over wording
            result = ReasonResult(template_reason(inp), "template", [f"api_error:{type(exc).__name__}"])
            self.tracker.record(result, "api_error")
            return result

        problems = grounding_problems(text, inp, prompt)
        if problems:
            result = ReasonResult(template_reason(inp), "template", problems)
            self.tracker.record(result, "grounding:" + ",".join(problems))
            return result

        result = ReasonResult(text, "llm")
        self.tracker.record(result)
        return result
