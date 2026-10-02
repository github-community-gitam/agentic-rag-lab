# Document Loading and Indexing

Document loading is the first step in any RAG system. The loader reads text files, normalizes whitespace, and preserves metadata that will later be used for debugging and traceability. A realistic loader should be deterministic, support UTF-8 content, and ignore files that are not appropriate for retrieval.

In this workshop, the loader accepts plain text and Markdown files and produces a `Document` object for each file. A normalized document record includes the document ID, the raw text, and metadata such as the filename and source path. The loader sorts files consistently so that the corpus order remains stable across repeated runs.

After loading, the indexer prepares those documents for search. Each chunk is turned into an embedding, and those embeddings are inserted into a FAISS index. A vector store preserves the relationship between a chunk and its metadata, which allows the retriever to return the matching text and its source document.

This separation of concerns is important. Loading extracts content, indexing prepares the data for search, and retrieval answers the user question by comparing the query embedding to the stored vectors.
