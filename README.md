# Agentic RAG Lab

A small, deterministic, offline-first playground for retrieval fundamentals. The project teaches the core pieces of a retrieval pipeline without relying on external APIs, paid services, or hosted runtimes.

## Purpose

This repository is intentionally focused on the foundations of retrieval-augmented generation:

- deterministic document loading
- text chunking
- consistent mock embeddings
- FAISS-backed vector indexing
- top-k retrieval
- an MCP-compatible retrieval tool layer
- reproducible, local tests

The goal is to make workshop participants comfortable with the mechanics behind retrieval before they move on to larger agent frameworks or hosted vector services.

## Architecture

Documents
  ↓
Loader
  ↓
Chunker
  ↓
Mock Embeddings
  ↓
FAISS
  ↓
Retriever
  ↓
MCP Retrieval Tool

## Requirements

- Python 3.12
- Local package installation with pip
- No API keys or external services required

## Setup

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
```

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e .
```

## Test

```bash
pytest -q
```

## Run Example

```bash
python examples/rag_demo.py
```

## No API Key Required

This repository is offline-first and intentionally uses deterministic mock embeddings rather than paid external embedding providers. The examples and tests are designed to run entirely on local fixtures. No secret, credential, or hosted service is required for standard execution.

## Workshop Corpus

The fixture corpus is intentionally built as a compact but realistic RAG workshop. Instead of three tiny text files, the project now includes six original Markdown documents that cover the main concepts of retrieval:

- a project overview and RAG fundamentals
- document loading and indexing
- chunking and segmentation
- embeddings and vector search
- MCP tool contracts and structured tool discovery
- evaluation and end-to-end retrieval workflows

These documents are designed to support questions such as:

- "What is retrieval augmented generation?"
- "How does a retriever choose the best chunks?"
- "What does an MCP tool do?"

The loader supports both `.txt` and `.md` files and preserves deterministic ordering, while the workshop corpus emphasizes the Markdown workflow that is common in documentation-heavy RAG use cases.

## Architecture Explanation

### Documents
The loader reads UTF-8 text files from a directory and converts them into a lightweight `Document` model. It sorts files deterministically, filters unsupported file types, and preserves metadata such as source and filename.

### Loader
The loader is purposely small and explicit. It is responsible for turning raw files into consistent `Document` objects without performing any network access.

### Chunker
The chunker splits a document into overlapping or contiguous text units. It validates chunk size and overlap to avoid invalid configurations and always produces non-empty chunks.

### Mock Embeddings
The embeddings layer derives fixed-length vectors deterministically from string content using `hashlib.sha256`. This keeps retrieval reproducible across runs and avoids Python hash randomization.

### FAISS
FAISS provides an in-memory vector index for local search. This project uses an `IndexFlatL2` index. Distances are lower-is-better, and the retriever converts them to a "score" using the negative distance so higher scores indicate stronger matches.

### Retriever
The retriever combines a query embedding with a vector store and returns ordered retrieval results. Query strings must be non-empty; results are sorted by relevance and can be bounded by `top_k`.

### MCP Retrieval Tool
The MCP integration is intentionally thin and keeps retrieval logic independent from the transport layer. The MCP handler validates input, invokes the retriever, and returns JSON-compatible structured results.

## Why these design decisions?

### Why mock embeddings?
Mock embeddings keep the workshop reproducible, deterministic, and fully offline. They teach the mechanics of retrieval without requiring any external model access.

### Why FAISS?
FAISS is a widely used local vector indexing library and works well for a small educational lab. It keeps the focus on indexing and retrieval behavior without adding cloud dependencies.

### Why deterministic tests?
The tests are designed to be local, stable, and machine-independent. That makes the repository suitable for workshop environments and CI without depending on randomness or external systems.

### Why retrieval is independent from MCP?
Separation keeps the retrieval logic reusable and testable outside the MCP layer. The MCP tool simply adapts the same retrieve interface to structured JSON payloads.

### Why the repository does not use LangChain or LlamaIndex?
Those frameworks are useful in larger systems, but they hide the mechanics that this workshop is meant to teach. This lab keeps the implementation explicit and understandable.

### Why external APIs are optional rather than required?
The repository is designed to work entirely offline. This reduces setup friction for contributors and ensures that learner exercises are repeatable regardless of network access or account setup.

## Contributor Guide

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, branch workflow, and contribution expectations.

## Project Structure

```text
agentic-rag-lab/
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── loaders.py
│       ├── chunkers.py
│       ├── embeddings.py
│       ├── vectorstore.py
│       ├── retriever.py
│       └── mcp_retrieval.py
├── tests/
├── fixtures/
├── schemas/
├── examples/
├── .github/
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── README.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt
└── .github/workflows/ci.yml
```
