from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import chromadb
from chromadb import errors as chroma_errors

from .config import Settings


@dataclass
class IndexedChunk:
    chunk_id: str
    message_id: str
    thread_id: str
    subject: str
    from_address: str
    to: List[str]
    date: str
    chunk_index: int
    snippet: str
    raw_path: str
    distance: float


class EmailIndex:
    def __init__(self, settings: Settings) -> None:
        self._client = chromadb.PersistentClient(path=str(settings.index_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = settings.chroma_collection

    def add_chunks(
        self,
        *,
        documents: Iterable[str],
        embeddings: Iterable[List[float]],
        metadatas: Iterable[dict],
        ids: Iterable[str],
    ) -> None:
        docs = list(documents)
        embeds = list(embeddings)
        metas = list(metadatas)
        ids_list = list(ids)
        if not docs:
            return
        if not(len(docs) == len(embeds) == len(metas) == len(ids_list)):
            raise ValueError("Documents, embeddings, metadatas, and ids must align in length.")
        self._collection.add(
            ids=ids_list,
            documents=docs,
            embeddings=embeds,
            metadatas=metas,
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except chroma_errors.InvalidCollectionException:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def query(
        self,
        *,
        query_embedding: List[float],
        n_results: int,
        where: Optional[dict] = None,
    ) -> List[IndexedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        hits: List[IndexedChunk] = []
        for idx, chunk_id in enumerate(result["ids"][0]):
            metadata = result["metadatas"][0][idx]
            doc = result["documents"][0][idx]
            distance = result["distances"][0][idx] if result.get("distances") else 0.0
            hits.append(
                IndexedChunk(
                    chunk_id=chunk_id,
                    message_id=metadata.get("message_id", "unknown"),
                    thread_id=metadata.get("thread_id", metadata.get("message_id", "unknown")),
                    subject=metadata.get("subject", "(no subject)"),
                    from_address=metadata.get("from_address", "unknown"),
                    to=metadata.get("to", []),
                    date=metadata.get("date", ""),
                    chunk_index=metadata.get("chunk_index", 0),
                    snippet=doc,
                    raw_path=metadata.get("raw_path", ""),
                    distance=distance,
                )
            )
        return hits
