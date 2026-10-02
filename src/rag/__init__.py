"""Core retrieval primitives for the Agentic RAG Lab."""

from rag.chunkers import Chunk, chunk_document
from rag.embeddings import EmbeddingProvider, MockEmbeddingProvider
from rag.loaders import Document, load_documents
from rag.retriever import RetrievalResult, Retriever
from rag.vectorstore import FaissVectorStore

__all__ = [
    "Chunk",
    "Document",
    "EmbeddingProvider",
    "FaissVectorStore",
    "MockEmbeddingProvider",
    "RetrievalResult",
    "Retriever",
    "chunk_document",
    "load_documents",
]
