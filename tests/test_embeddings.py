from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import MockEmbeddingProvider


def test_mock_embedding_provider_empty_input() -> None:
    provider = MockEmbeddingProvider()

    assert provider.embed([]) == []


def test_mock_embedding_provider_single_text() -> None:
    provider = MockEmbeddingProvider()
    embedding = provider.embed(["hello world"])[0]

    assert len(embedding) == 8
    assert all(isinstance(value, float) for value in embedding)


def test_mock_embedding_provider_multiple_texts() -> None:
    provider = MockEmbeddingProvider()
    embeddings = provider.embed(["alpha", "beta", "gamma"])

    assert len(embeddings) == 3
    assert all(len(vector) == 8 for vector in embeddings)


def test_mock_embedding_provider_is_deterministic() -> None:
    provider = MockEmbeddingProvider()

    assert provider.embed(["same text"]) == provider.embed(["same text"])
    assert provider.embed(["alpha", "beta"]) == provider.embed(["alpha", "beta"])


def test_mock_embedding_provider_matches_numpy_float32_shape() -> None:
    provider = MockEmbeddingProvider()
    embedding = provider.embed(["demo"])[0]
    array = np.asarray(embedding, dtype=np.float32)

    assert array.shape == (8,)
def test_mock_embedding_provider_rejects_dimension_above_hash_capacity() -> None:
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dimension=9)