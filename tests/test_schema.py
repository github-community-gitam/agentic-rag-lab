from __future__ import annotations

import json

import jsonschema
import pytest


@pytest.fixture
def schema_path() -> str:
    return "schemas/retrieval.schema.json"


def test_retrieval_schema_accepts_valid_response() -> None:
    with open("schemas/retrieval.schema.json", "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    payload = {
        "results": [
            {
                "chunk_id": "c1",
                "text": "hello",
                "source": "demo.txt",
                "score": 0.75,
            }
        ]
    }

    jsonschema.validate(instance=payload, schema=schema)


def test_retrieval_schema_rejects_missing_results() -> None:
    with open("schemas/retrieval.schema.json", "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance={}, schema=schema)


def test_retrieval_schema_rejects_missing_chunk_id() -> None:
    with open("schemas/retrieval.schema.json", "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    payload = {"results": [{"text": "hello", "source": "demo.txt", "score": 0.75}]}

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_retrieval_schema_rejects_missing_source() -> None:
    with open("schemas/retrieval.schema.json", "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    payload = {"results": [{"chunk_id": "c1", "text": "hello", "score": 0.75}]}

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)


def test_retrieval_schema_rejects_invalid_score_type() -> None:
    with open("schemas/retrieval.schema.json", "r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    payload = {"results": [{"chunk_id": "c1", "text": "hello", "source": "demo.txt", "score": "high"}]}

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(instance=payload, schema=schema)
