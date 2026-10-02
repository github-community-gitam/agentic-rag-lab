from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one deterministic embedding vector per input text."""


class MockEmbeddingProvider:
    """Deterministic mock embeddings for workshop retrieval exercises.

    This is intentionally not a semantically strong production embedding model.
    The implementation uses a stable SHA-256 hash to keep results reproducible and
    independent of Python's randomized hash seed.
    """

    def __init__(self, dimension: int = 8) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if texts == []:
            return []

        vectors: list[list[float]] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Each text value must be a string")
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector: list[float] = []
            for index in range(self.dimension):
                chunk = digest[(index * 4) : ((index + 1) * 4)]
                value = int.from_bytes(chunk, byteorder="big", signed=False) / (2**32 - 1)
                vector.append((value * 2.0) - 1.0)
            vectors.append(vector)
        return vectors
