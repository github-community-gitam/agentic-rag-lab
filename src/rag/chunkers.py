from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.loaders import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_document(document: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split a document into deterministic chunks with optional overlap."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise TypeError("chunk_size must be a positive integer")
    if not isinstance(overlap, int):
        raise TypeError("overlap must be an integer")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not document.text:
        return []

    if len(document.text) <= chunk_size:
        return [
            Chunk(
                chunk_id=f"{document.document_id}::0",
                text=document.text,
                document_id=document.document_id,
                metadata=dict(document.metadata),
            )
        ]

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0

    while start < len(document.text):
        end = min(start + chunk_size, len(document.text))
        chunk_text = document.text[start:end]
        if not chunk_text:
            break

        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::{len(chunks)}",
                text=chunk_text,
                document_id=document.document_id,
                metadata=dict(document.metadata),
            )
        )

        if end >= len(document.text):
            break
        start += step

    return chunks
