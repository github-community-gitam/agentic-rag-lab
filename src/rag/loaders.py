from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_TEXT_EXTENSIONS = {"txt", "md", "csv", "json", "yaml", "yml", "log", "rst"}


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", ""))


def load_documents(directory: str | Path) -> list[Document]:
    """Load UTF-8 text documents from a directory in deterministic order."""
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Document directory not found: {path}")

    documents: list[Document] = []
    for file_path in sorted(path.iterdir(), key=lambda item: item.name):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower().lstrip(".")
        if suffix not in SUPPORTED_TEXT_EXTENSIONS:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative_source = file_path.relative_to(path).as_posix()
        documents.append(
            Document(
                document_id=file_path.stem,
                text=content,
                metadata={
                    "source": relative_source,
                    "filename": file_path.name,
                    "document_type": suffix,
                },
            )
        )

    return documents
