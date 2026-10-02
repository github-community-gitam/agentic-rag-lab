# Contributing to Agentic RAG Lab

Thank you for helping improve this workshop-focused repository.

## Local setup

- Use Python 3.12.
- Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Testing

Run the local test suite before opening a PR:

```bash
pytest -q
ruff check .
```

## Branching and commits

- Use a feature branch such as `feat/add-chunker-optimizer`
- Keep commit messages clear and scoped
- Keep the repository offline-first and deterministic

## Pull request expectations

- Add or update tests for changed behavior
- Validate the offline/mock workflow
- Do not add paid APIs or cloud dependencies to core examples
- Avoid broad refactors not related to the change

## Working on chunking

When modifying chunking, check:

- chunk size validation
- overlap validation
- deterministic ordering
- empty-document handling
- metadata preservation

## Working on retrieval

When modifying retrieval, check:

- query validation
- top-k boundaries
- ordering by relevance
- empty-index handling
- deterministic output

## Working on MCP integration

Keep retrieval independent from the MCP transport layer.

- validate JSON input
- return structured output
- do not couple the implementation to external services
- prefer small integration tests over broad mock-heavy tests

## Offline-first policy

This project should never depend on paid APIs or external services for core examples. If a new dependency is needed, prefer local, minimal packages and document the reason clearly.
