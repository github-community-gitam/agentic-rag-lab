from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.embeddings import EmbeddingProvider, MockEmbeddingProvider
from rag.vectorstore import FaissVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    text: str
    source: str
    score: float | None
    metadata: dict[str, Any]


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: FaissVectorStore,
        chunks: list[Any],
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunks = list(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        normalized_query = query.strip()
        if not self.chunks or self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            return []

        query_embedding = self.embedding_provider.embed([normalized_query])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return [
            RetrievalResult(
                chunk_id=result["chunk_id"],
                text=result["text"],
                source=result["source"],
                score=result.get("score"),
                metadata=result.get("metadata", {}),
            )
            for result in results
        ]


def make_default_retriever(chunks: list[Any] | None = None) -> Retriever:
    """Convenience constructor used by examples and local tooling."""
    store = FaissVectorStore()
    documents = chunks or []
    if documents:
        embeddings = MockEmbeddingProvider().embed([chunk.text for chunk in documents])
        store.add(documents, embeddings)
    return Retriever(embedding_provider=MockEmbeddingProvider(), vector_store=store, chunks=documents)
