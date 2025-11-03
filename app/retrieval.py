from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import fuzz

from .config import Settings
from .embedding import Embedder
from .index import EmailIndex, IndexedChunk


@dataclass
class RetrievedChunk:
    chunk: IndexedChunk
    score: float


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = Embedder(settings)
        self._index = EmailIndex(settings)

    def search(self, query: str, *, where: Optional[dict] = None) -> List[RetrievedChunk]:
        query_embedding = self._embedder.embed([query])
        if not query_embedding:
            return []
        raw_hits = self._index.query(
            query_embedding=query_embedding[0],
            n_results=self._settings.top_k,
            where=where,
        )
        reranked = []
        for chunk in raw_hits:
            keyword_score = fuzz.partial_ratio(query.lower(), chunk.snippet.lower()) / 100.0
            similarity_score = 1 - chunk.distance if chunk.distance else 0.0
            combined = (similarity_score * 0.7) + (keyword_score * 0.3)
            reranked.append(RetrievedChunk(chunk=chunk, score=combined))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked
