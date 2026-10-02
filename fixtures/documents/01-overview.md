# Retrieval-Augmented Generation Overview

Retrieval-augmented generation, or RAG, combines a searchable knowledge base with a language model. Instead of asking the model to memorize everything in its weights, the system first retrieves the most relevant evidence from a corpus and then uses that evidence as context.

A typical RAG pipeline starts with a document loader, which reads source files from disk or a remote store. The loader converts each source into a normalized document object with metadata such as filename, language, and source path. The next stage splits each document into chunks so that retrieval can focus on smaller, semantically meaningful passages.

Once chunks are created, an embedding model maps them into a vector space. A FAISS index stores those vectors for efficient similarity search. At query time, the system embeds the user question, searches the index, and returns the most relevant chunks. The selected passages are then supplied to a downstream model for final reasoning or synthesis.

This project is intentionally small, transparent, and offline-first. It teaches the mechanics behind retrieval without hiding them behind a large framework.
