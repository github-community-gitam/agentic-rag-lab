from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunkers import Chunk, chunk_document
from rag.loaders import Document, load_documents


@pytest.fixture
def temp_document_dir(tmp_path: Path) -> Path:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "alpha.txt").write_text("alpha beta gamma\n", encoding="utf-8")
    (documents_dir / "beta.txt").write_text("delta epsilon zeta\n", encoding="utf-8")
    return documents_dir


@pytest.fixture
def sample_documents(temp_document_dir: Path) -> list[Document]:
    return load_documents(temp_document_dir)


@pytest.fixture
def sample_chunks(sample_documents: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in sample_documents:
        chunks.extend(chunk_document(document, chunk_size=10, overlap=2))
    return chunks


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ]


@pytest.fixture
def temp_faiss_index(tmp_path: Path):
    return tmp_path / "store.index"


@pytest.fixture
def schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas" / "retrieval.schema.json"
