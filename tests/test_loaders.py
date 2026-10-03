from __future__ import annotations

from pathlib import Path

import pytest

from rag.loaders import load_documents


def test_load_documents_valid_directory(temp_document_dir: Path) -> None:
    documents = load_documents(temp_document_dir)

    assert len(documents) == 2
    assert [doc.filename for doc in documents] == ["alpha.txt", "beta.txt"]
    assert documents[0].text == "alpha beta gamma\n"


def test_load_documents_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert load_documents(empty_dir) == []


def test_load_documents_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_dir)


def test_load_documents_utf8_and_unsupported_extensions(tmp_path: Path) -> None:
    doc_dir = tmp_path / "utf8"
    doc_dir.mkdir()
    (doc_dir / "hello.txt").write_text("héllo\n", encoding="utf-8")
    (doc_dir / "skip.bin").write_bytes(b"\x00\x01\x02")
    (doc_dir / "notes.md").write_text("markdown\n", encoding="utf-8")

    documents = load_documents(doc_dir)

    assert [doc.filename for doc in documents] == ["hello.txt", "notes.md"]
    assert documents[0].text == "héllo\n"
    assert documents[0].source == "hello.txt"
    assert "/tmp" not in documents[0].source


def test_load_documents_markdown_files_are_supported(tmp_path: Path) -> None:
    doc_dir = tmp_path / "workshop"
    doc_dir.mkdir()
    (doc_dir / "overview.md").write_text("# Overview\n\nRAG helps retrieval.\n", encoding="utf-8")
    (doc_dir / "notes.txt").write_text("plain text\n", encoding="utf-8")

    documents = load_documents(doc_dir)

    assert [doc.filename for doc in documents] == ["notes.txt", "overview.md"]
    assert all(doc.source.endswith((".txt", ".md")) for doc in documents)


def test_workshop_corpus_is_a_markdown_collection() -> None:
    documents = load_documents(Path(__file__).resolve().parents[1] / "fixtures" / "documents")

    assert len(documents) == 6
    assert all(doc.filename.endswith(".md") for doc in documents)
    assert all(doc.metadata["source"].endswith(".md") for doc in documents)

def test_load_documents_unique_document_ids_for_same_stem(tmp_path: Path) -> None:
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()

    (doc_dir / "notes.txt").write_text("text file\n", encoding="utf-8")
    (doc_dir / "notes.md").write_text("markdown file\n", encoding="utf-8")

    documents = load_documents(doc_dir)

    assert len(documents) == 2
    assert documents[0].document_id != documents[1].document_id