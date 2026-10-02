# Chunking Strategies

Chunking is the process of splitting a document into smaller segments before indexing. A chunk should be large enough to preserve context, but small enough to stay semantically focused. The right size depends on the use case, but the engineering principle is consistent: short chunks reduce noise and improve retrieval precision.

A basic chunker takes a document and a chunk size, then slices the text into contiguous blocks. Many systems also add overlap so that adjacent chunks share some context. Overlap is helpful when the relevant fact is split across boundaries, but too much overlap can create redundant retrieval results.

Deterministic chunking matters because it makes the retrieval system stable and easy to test. In this project, chunk size and overlap are explicit parameters, and invalid combinations are rejected. That prevents accidental behavior where a chunker silently produces empty or irregular segments.

Good chunking turns a document into a set of retrieval units. Each unit can be ranked independently against a query, which allows the system to identify the most relevant slice of information instead of retrieving an entire article.
