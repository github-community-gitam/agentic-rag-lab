from __future__ import annotations

import pytest

from rag.chunkers import chunk_document
from rag.loaders import Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        document_id="doc-1",
        text="abcdefghijklmno",
        metadata={"source": "fixture", "filename": "sample.txt"},
    )


def test_chunk_document_empty_text() -> None:
    document = Document(document_id="doc-empty", text="", metadata={"source": "fixture", "filename": "empty.txt"})

    assert chunk_document(document, chunk_size=10, overlap=2) == []

def test_chunk_document_whitespace_only_text() -> None:
    document = Document(
        document_id="doc-whitespace",
        text="   ",
        metadata={"source": "fixture", "filename": "whitespace.txt"},
    )

    assert chunk_document(document, chunk_size=10, overlap=2) == []


def test_chunk_document_newline_only_text() -> None:
    document = Document(
        document_id="doc-newlines",
        text="\n\n",
        metadata={"source": "fixture", "filename": "newlines.txt"},
    )

    assert chunk_document(document, chunk_size=10, overlap=2) == []


def test_chunk_document_shorter_than_chunk_size(sample_document: Document) -> None:
    chunks = chunk_document(sample_document, chunk_size=20, overlap=2)

    assert len(chunks) == 1
    assert chunks[0].text == sample_document.text


def test_chunk_document_exact_boundary(sample_document: Document) -> None:
    chunks = chunk_document(sample_document, chunk_size=15, overlap=0)

    assert len(chunks) == 1
    assert [chunk.text for chunk in chunks] == [sample_document.text]


def test_chunk_document_with_overlap(sample_document: Document) -> None:
    chunks = chunk_document(sample_document, chunk_size=10, overlap=2)

    assert len(chunks) == 2
    assert [chunk.text for chunk in chunks] == [
        "abcdefghij",
        "ijklmno",
    ]


def test_chunk_document_invalid_chunk_size(sample_document: Document) -> None:
    with pytest.raises((TypeError, ValueError)):
        chunk_document(sample_document, chunk_size=0, overlap=0)

    with pytest.raises((TypeError, ValueError)):
        chunk_document(sample_document, chunk_size=-1, overlap=0)


def test_chunk_document_invalid_overlap(sample_document: Document) -> None:
    with pytest.raises((TypeError, ValueError)):
        chunk_document(sample_document, chunk_size=10, overlap=-1)

    with pytest.raises((TypeError, ValueError)):
        chunk_document(sample_document, chunk_size=10, overlap=10)


def test_chunk_document_metadata_preserved(sample_document: Document) -> None:
    chunks = chunk_document(sample_document, chunk_size=10, overlap=2)

    assert chunks[0].metadata["source"] == "fixture"
    assert chunks[0].metadata["filename"] == "sample.txt"


def test_chunk_document_deterministic_repeated_execution(sample_document: Document) -> None:
    first = chunk_document(sample_document, chunk_size=10, overlap=2)
    second = chunk_document(sample_document, chunk_size=10, overlap=2)

    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
