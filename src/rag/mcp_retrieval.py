from __future__ import annotations

from typing import Any

from rag.embeddings import MockEmbeddingProvider
from rag.retriever import Retriever
from rag.vectorstore import FaissVectorStore


def build_mcp_tool(
    vector_store: FaissVectorStore,
    chunks: list[Any],
    embedding_provider: Any | None = None,
) -> dict[str, Any]:
    """Create a lightweight MCP-style retrieval tool interface."""
    provider = embedding_provider or MockEmbeddingProvider()

    def handler(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")

        query = payload.get("query")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")

        top_k = int(payload.get("top_k", 3))
        retriever = Retriever(embedding_provider=provider, vector_store=vector_store, chunks=chunks)
        results = retriever.retrieve(query, top_k=top_k)

        return {
            "results": [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "source": result.source,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in results
            ]
        }

    return {
        "name": "retrieval_tool",
        "description": "Retrieve the most relevant chunks for a user query.",
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query for local retrieval"},
                "top_k": {"type": "integer", "minimum": 1, "default": 3},
            },
            "additionalProperties": True,
        },
        "handler": handler,
    }


def execute_retrieval_tool(payload: dict[str, Any], retriever: Retriever | None = None) -> dict[str, Any]:
    """Convenience wrapper for direct tool invocation."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dictionary")

    if retriever is None:
        retriever = Retriever(
            embedding_provider=MockEmbeddingProvider(),
            vector_store=FaissVectorStore(),
            chunks=[],
        )

    query = payload.get("query")
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string")

    top_k = int(payload.get("top_k", 3))
    results = retriever.retrieve(query, top_k=top_k)
    return {
        "results": [
            {
                "chunk_id": result.chunk_id,
                "text": result.text,
                "source": result.source,
                "score": result.score,
                "metadata": result.metadata,
            }
            for result in results
        ]
    }
