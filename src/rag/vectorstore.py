from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.chunkers import Chunk


class FaissVectorStore:
    """Simple vector store built on FAISS for local retrieval experiments."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []
        self._dimension: int | None = None

    @property
    def index(self) -> faiss.Index | None:
        return self._index

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks and not embeddings:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if len(chunks) == 0:
            raise ValueError("chunks cannot be empty when embeddings are provided")

        first_embedding = embeddings[0]
        if not isinstance(first_embedding, list):
            raise TypeError("Each embedding must be a list of floats")

        dimension = len(first_embedding)
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")

        if self._index is None:
            self._index = faiss.IndexFlatL2(dimension)
            self._dimension = dimension
        elif self._dimension != dimension:
            raise ValueError(f"embedding dimensionality mismatch: expected {self._dimension}, got {dimension}")

        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != dimension:
            raise ValueError("embeddings must be a 2D array with a consistent dimension")

        self._index.add(array)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if self._dimension is None:
            raise ValueError("vector store has no indexed dimension")
        if query.shape[1] != self._dimension:
            raise ValueError(f"query dimension mismatch: expected {self._dimension}, got {query.shape[1]}")

        limit = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query, limit)
        results: list[dict[str, Any]] = []

        for distance_value, chunk_index in zip(distances[0], indices[0], strict=True):
            if chunk_index < 0 or chunk_index >= len(self._chunks):
                continue
            chunk = self._chunks[int(chunk_index)]
            distance = float(distance_value)
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.metadata.get("source", ""),
                    "score": -distance,
                    "distance": distance,
                    "metadata": dict(chunk.metadata),
                }
            )

        return results

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise ValueError("cannot save an empty vector store")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(destination))

        metadata_payload = {
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "metadata": dict(chunk.metadata),
                }
                for chunk in self._chunks
            ]
        }
        meta_path = destination.with_suffix(destination.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata_payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Index file not found: {source}")

        self._index = faiss.read_index(str(source))
        self._dimension = self._index.d
        self._chunks = []

        meta_path = source.with_suffix(source.suffix + ".meta.json")
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            for item in payload.get("chunks", []):
                self._chunks.append(
                    Chunk(
                        chunk_id=item["chunk_id"],
                        text=item["text"],
                        document_id=item["document_id"],
                        metadata=item.get("metadata", {}),
                    )
                )
