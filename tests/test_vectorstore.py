from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunkers import Chunk
from rag.vectorstore import FaissVectorStore


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id="chunk-1", text="one", document_id="doc-1", metadata={"source": "a.txt", "filename": "a.txt"}),
        Chunk(chunk_id="chunk-2", text="two", document_id="doc-1", metadata={"source": "b.txt", "filename": "b.txt"}),
        Chunk(chunk_id="chunk-3", text="three", document_id="doc-2", metadata={"source": "c.txt", "filename": "c.txt"}),
    ]


def test_faiss_vector_store_add_and_query(chunks: list[Chunk], sample_embeddings: list[list[float]]) -> None:
    store = FaissVectorStore()
    store.add(chunks, sample_embeddings)

    results = store.search(sample_embeddings[0], top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] in {"chunk-1", "chunk-2", "chunk-3"}
    assert "metadata" in results[0]


def test_faiss_vector_store_top_k_clamps_to_available(chunks: list[Chunk], sample_embeddings: list[list[float]]) -> None:
    store = FaissVectorStore()
    store.add(chunks, sample_embeddings)

    results = store.search(sample_embeddings[0], top_k=10)
    assert len(results) == 3


def test_faiss_vector_store_rejects_invalid_top_k(chunks: list[Chunk], sample_embeddings: list[list[float]]) -> None:
    store = FaissVectorStore()
    store.add(chunks, sample_embeddings)

    with pytest.raises(ValueError):
        store.search(sample_embeddings[0], top_k=0)

    with pytest.raises(ValueError):
        store.search(sample_embeddings[0], top_k=-1)


def test_faiss_vector_store_rejects_dimension_mismatch(chunks: list[Chunk]) -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError):
        store.add(chunks, [[0.1, 0.2]])


def test_faiss_vector_store_empty_index_returns_empty_search() -> None:
    store = FaissVectorStore()

    assert store.search([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], top_k=3) == []


def test_faiss_vector_store_save_and_load_round_trip(chunks: list[Chunk], sample_embeddings: list[list[float]], tmp_path: Path) -> None:
    store = FaissVectorStore()
    store.add(chunks, sample_embeddings)
    path = tmp_path / "retrieval.index"

    store.save(path)
    restored = FaissVectorStore()
    restored.load(path)

    assert restored.search(sample_embeddings[0], top_k=1)[0]["chunk_id"] == store.search(sample_embeddings[0], top_k=1)[0]["chunk_id"]
    assert restored.index.ntotal == 3
def test_faiss_vector_store_rejects_non_finite_embeddings(
    chunks: list[Chunk],
) -> None:
    store = FaissVectorStore()

    embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, float("nan")],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ]

    with pytest.raises(ValueError):
        store.add(chunks, embeddings)