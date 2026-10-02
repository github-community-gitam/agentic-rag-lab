# Embeddings and Vector Search

Embeddings transform text into dense vectors so that similarity can be computed with mathematical operations instead of string comparisons. In a retrieval pipeline, each chunk receives an embedding that captures semantic relationships between words and phrases.

A mock embedding provider is useful in education because it is deterministic and reproducible. It does not require external models or API calls. Instead, the provider derives a stable vector from the chunk text using a hashing-based approach. This makes the workshop easy to run locally and ensures tests remain repeatable.

Once embeddings are created, a vector store such as FAISS can index the vectors. The store keeps both the vector and a reference to the chunk metadata, so the retriever can return the original text and source file. Query-time retrieval is then a nearest-neighbor search: convert the query into an embedding, compare it against the stored vectors, and select the best matches.

The ranking function is measured by distance. Lower distance means a stronger match, and a retrieval layer can convert that to a score that is easier to explain in a result table.
