"""In-memory Chroma vector store over the context notes (all-MiniLM-L6-v2 embeddings).

Retrieval only proposes candidate notes; it never decides a verdict. See
``justify.py`` for the deterministic validator.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from functools import lru_cache
from typing import Mapping, Sequence

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from huggingface_hub import try_to_load_from_cache

from freight_watch import config
from freight_watch.justify import (
    ALL_ROUTES,
    DEFAULT_LOOKAROUND_DAYS,
    DEFAULT_WINDOW_DAYS,
    RetrievalHit,
    note_window,
    to_date,
)

DEFAULT_EMBED_MODEL = config.EMBEDDING_MODEL


def _is_cached(model_name: str) -> bool:
    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return isinstance(try_to_load_from_cache(repo_id, "modules.json"), str)


@lru_cache(maxsize=None)
def embedding_function(model_name: str = DEFAULT_EMBED_MODEL):
    # A cached model loads offline; otherwise every load retries Hub HEAD requests,
    # which stalls for minutes behind proxies that break TLS.
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name, device="cpu", local_files_only=_is_cached(model_name)
    )


def ephemeral_client() -> chromadb.ClientAPI:
    return chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))


def query_text(route: str, week_of: date) -> str:
    return f"cost increase on {route} week of {week_of.isoformat()}"


def _document(note: Mapping) -> str:
    return f"{note['applies_to']} ({to_date(note['date']).isoformat()}): {note['note']}"


class NoteIndex:
    """Chroma collection of notes with route and date-window metadata."""

    def __init__(
        self,
        notes: Sequence[Mapping],
        model_name: str = DEFAULT_EMBED_MODEL,
        default_window_days: int = DEFAULT_WINDOW_DAYS,
        client: chromadb.ClientAPI | None = None,
    ):
        self._client = client or ephemeral_client()
        self._collection = self._client.create_collection(
            name=f"notes-{uuid.uuid4().hex[:12]}",
            embedding_function=embedding_function(model_name),
            metadata={"hnsw:space": "cosine"},
        )
        ordered = sorted(notes, key=lambda n: n["note_id"])
        metadatas = []
        for n in ordered:
            start, end = note_window(n, default_window_days)
            metadatas.append(
                {
                    "note_id": n["note_id"],
                    "applies_to": n["applies_to"],
                    "date": to_date(n["date"]).isoformat(),
                    "start_ord": start.toordinal(),
                    "end_ord": end.toordinal(),
                    "raises_cost": bool(n["raises_cost"]),
                }
            )
        self._collection.add(
            ids=[n["note_id"] for n in ordered],
            documents=[_document(n) for n in ordered],
            metadatas=metadatas,
        )
        self.size = len(ordered)

    def search(
        self,
        route: str,
        week_of: date,
        k: int = 3,
        lookaround_days: int = DEFAULT_LOOKAROUND_DAYS,
    ) -> list[RetrievalHit]:
        """Top-k notes for this route (or All Routes) whose window is near the week."""
        week_start = to_date(week_of)
        week_end = week_start + timedelta(days=6)
        where = {
            "$and": [
                {"applies_to": {"$in": [route, ALL_ROUTES]}},
                {"end_ord": {"$gte": week_start.toordinal() - lookaround_days}},
                {"start_ord": {"$lte": week_end.toordinal() + lookaround_days}},
            ]
        }
        return self._query(query_text(route, week_start), k, where)

    def search_text(self, text: str, k: int = 3) -> list[RetrievalHit]:
        """Unfiltered semantic search, for free-form questions."""
        return self._query(text, k, None)

    def _query(self, text: str, k: int, where: dict | None) -> list[RetrievalHit]:
        if self.size == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_texts=[text],
            n_results=min(k, self.size),
            where=where,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        hits = [
            RetrievalHit(note_id=i, similarity=round(1.0 - float(d), 6))
            for i, d in zip(ids, distances)
        ]
        return sorted(hits, key=lambda h: (-h.similarity, h.note_id))
