from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.chunkers import chunk_document
from rag.embeddings import MockEmbeddingProvider
from rag.loaders import load_documents
from rag.retriever import Retriever
from rag.vectorstore import FaissVectorStore


def main() -> None:
    fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures" / "documents"
    documents = load_documents(fixtures_dir)
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=120, overlap=20))

    embedding_provider = MockEmbeddingProvider()
    embeddings = embedding_provider.embed([chunk.text for chunk in chunks])
    vector_store = FaissVectorStore()
    vector_store.add(chunks, embeddings)

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunks=chunks,
    )

    queries = [
        "What is retrieval augmented generation?",
        "What does an MCP tool do?",
        "How does a vector store rank relevant chunks?",
    ]

    for query in queries:
        results = retriever.retrieve(query, top_k=3)
        print(f"Query: {query}")
        print("Results:")
        for index, result in enumerate(results, start=1):
            print(f"  {index}. {result.source} (score={result.score:.4f})")
            print(f"     {result.text[:120]}...")
        print()


if __name__ == "__main__":
    main()
