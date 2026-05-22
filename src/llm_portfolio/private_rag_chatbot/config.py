"""Configuration helpers for the Private Knowledge Chatbot (RAG) case study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RagConfig:
    model: str = "gpt-4o-mini"
    persist_directory: str = "vector_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4


def resolve_kb_dir(base_dir: str | Path = ".", knowledge_dir: str = "knowledge-base") -> Path:
    return (Path(base_dir) / knowledge_dir).resolve()
