from __future__ import annotations

import pytest

from rag.chunkers import Chunk
from rag.embeddings import MockEmbeddingProvider
from rag.retriever import Retriever
from rag.vectorstore import FaissVectorStore


@pytest.fixture
def retriever() -> Retriever:
    chunks = [
        Chunk(chunk_id="c1", text="MCP tools help clients call servers.", document_id="doc-1", metadata={"source": "mcp.txt", "filename": "mcp.txt"}),
        Chunk(chunk_id="c2", text="Vector search uses embeddings to rank chunks.", document_id="doc-2", metadata={"source": "rag.txt", "filename": "rag.txt"}),
    ]
    store = FaissVectorStore()
    store.add(chunks, MockEmbeddingProvider().embed([chunk.text for chunk in chunks]))
    return Retriever(embedding_provider=MockEmbeddingProvider(), vector_store=store, chunks=chunks)


def test_retriever_basic_query(retriever: Retriever) -> None:
    results = retriever.retrieve("MCP tools")

    assert len(results) >= 1
    assert results[0].chunk_id in {"c1", "c2"}
    assert results[0].source in {"mcp.txt", "rag.txt"}


def test_retriever_top_k(retriever: Retriever) -> None:
    results = retriever.retrieve("MCP tools", top_k=1)

    assert len(results) == 1


def test_retriever_rejects_empty_query(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retriever_deterministic(retriever: Retriever) -> None:
    first = retriever.retrieve("MCP tools", top_k=2)
    second = retriever.retrieve("MCP tools", top_k=2)

    assert [result.chunk_id for result in first] == [result.chunk_id for result in second]


def test_retriever_handles_no_indexed_documents() -> None:
    retriever = Retriever(embedding_provider=MockEmbeddingProvider(), vector_store=FaissVectorStore(), chunks=[])

    assert retriever.retrieve("what is retrieval") == []


def test_retriever_metadata_in_results(retriever: Retriever) -> None:
    result = retriever.retrieve("MCP tools", top_k=1)[0]

    assert result.metadata["filename"] in {"mcp.txt", "rag.txt"}
    assert result.score is not None
