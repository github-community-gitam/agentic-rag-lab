from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from rag.chunkers import Chunk
from rag.embeddings import MockEmbeddingProvider
from rag.mcp_retrieval import build_mcp_tool, execute_retrieval_tool
from rag.vectorstore import FaissVectorStore


@pytest.fixture
def retrieval_fixture() -> tuple[list[Chunk], FaissVectorStore]:
    chunks = [
        Chunk(chunk_id="c1", text="MCP tools are defined with schemas.", document_id="doc-1", metadata={"source": "mcp.txt", "filename": "mcp.txt"}),
        Chunk(chunk_id="c2", text="Retrieval works with vector similarity.", document_id="doc-2", metadata={"source": "rag.txt", "filename": "rag.txt"}),
    ]
    store = FaissVectorStore()
    store.add(chunks, MockEmbeddingProvider().embed([chunk.text for chunk in chunks]))
    return chunks, store


def test_mcp_tool_valid_input(retrieval_fixture: tuple[list[Chunk], FaissVectorStore]) -> None:
    chunks, store = retrieval_fixture
    tool = build_mcp_tool(store, chunks)

    payload = {"query": "MCP tools", "top_k": 3}
    response = tool["handler"](payload)

    assert response["results"]
    assert isinstance(response["results"][0]["chunk_id"], str)


def test_mcp_tool_default_top_k(retrieval_fixture: tuple[list[Chunk], FaissVectorStore]) -> None:
    _, store = retrieval_fixture
    tool = build_mcp_tool(store, [
        Chunk(chunk_id="c1", text="one", document_id="d1", metadata={"source": "one.txt", "filename": "one.txt"}),
        Chunk(chunk_id="c2", text="two", document_id="d2", metadata={"source": "two.txt", "filename": "two.txt"}),
    ])
    response = tool["handler"]({"query": "two"})

    assert len(response["results"]) <= 3


def test_mcp_tool_custom_top_k(retrieval_fixture: tuple[list[Chunk], FaissVectorStore]) -> None:
    _, store = retrieval_fixture
    tool = build_mcp_tool(store, [
        Chunk(chunk_id="c1", text="one", document_id="d1", metadata={"source": "one.txt", "filename": "one.txt"}),
        Chunk(chunk_id="c2", text="two", document_id="d2", metadata={"source": "two.txt", "filename": "two.txt"}),
    ])
    response = tool["handler"]({"query": "two", "top_k": 1})

    assert len(response["results"]) == 1


def test_mcp_tool_empty_query(retrieval_fixture: tuple[list[Chunk], FaissVectorStore]) -> None:
    _, store = retrieval_fixture
    tool = build_mcp_tool(store, [
        Chunk(chunk_id="c1", text="one", document_id="d1", metadata={"source": "one.txt", "filename": "one.txt"}),
    ])

    with pytest.raises(ValueError):
        tool["handler"]({"query": "   "})


def test_mcp_tool_invalid_top_k(retrieval_fixture: tuple[list[Chunk], FaissVectorStore]) -> None:
    _, store = retrieval_fixture
    tool = build_mcp_tool(store, [
        Chunk(chunk_id="c1", text="one", document_id="d1", metadata={"source": "one.txt", "filename": "one.txt"}),
    ])

    with pytest.raises(ValueError):
        tool["handler"]({"query": "one", "top_k": 0})


def test_mcp_tool_structured_output(retrieval_fixture: tuple[list[Chunk], FaissVectorStore]) -> None:
    _, store = retrieval_fixture
    tool = build_mcp_tool(store, [
        Chunk(chunk_id="c1", text="one", document_id="d1", metadata={"source": "one.txt", "filename": "one.txt"}),
        Chunk(chunk_id="c2", text="two", document_id="d2", metadata={"source": "two.txt", "filename": "two.txt"}),
    ])

    response = tool["handler"]({"query": "one", "top_k": 2})

    assert set(response.keys()) == {"results"}
    assert set(response["results"][0].keys()) >= {"chunk_id", "text", "source", "score"}


def test_mcp_tool_schema_compatibility(retrieval_fixture: tuple[list[Chunk], FaissVectorStore], schema_path: Path) -> None:
    _, store = retrieval_fixture
    tool = build_mcp_tool(store, [
        Chunk(chunk_id="c1", text="one", document_id="d1", metadata={"source": "one.txt", "filename": "one.txt"}),
    ])

    response = tool["handler"]({"query": "one"})
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=response, schema=schema)


def test_execute_retrieval_tool() -> None:
    response = execute_retrieval_tool({"query": "test"}, retriever=None)
    assert isinstance(response, dict)
    assert "results" in response

def test_mcp_tool_rejects_invalid_top_k_type(
    retrieval_fixture: tuple[list[Chunk], FaissVectorStore],
) -> None:
    _, store = retrieval_fixture

    tool = build_mcp_tool(
        store,
        [
            Chunk(
                chunk_id="c1",
                text="one",
                document_id="d1",
                metadata={"source": "one.txt", "filename": "one.txt"},
            ),
        ],
    )

    with pytest.raises(TypeError):
        tool["handler"]({"query": "one", "top_k": "2"})