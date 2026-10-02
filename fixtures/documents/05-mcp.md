# MCP Tool Contracts

The Model Context Protocol defines a clean contract for connecting clients and servers. In practice, an MCP server exposes tools, resources, and prompts that other systems can inspect and invoke programmatically. This makes tool access predictable, structured, and machine-readable.

A tool contract includes a name, a description, and input schema. The client can ask the server which tools are available before invoking one. That allows an agent to discover capabilities without hard-coding them in advance. In a retrieval workflow, a tool may accept a user query and return the top-k most relevant chunks.

The retrieval tool in this project is intentionally thin. It validates the input query, invokes the retriever, and converts the returned results into a JSON-friendly payload. The MCP boundary is separated from the retrieval logic so that the same retrieval behavior can be reused outside a protocol implementation.

This explicit boundary is helpful for debugging. The protocol defines the interface, while the underlying code still owns the actual search semantics.
