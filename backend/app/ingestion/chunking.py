"""Structural-first, semantic-fallback chunker.

Strategy:
  1. Walk the parsed document by section headings. Each section becomes one or
     more chunks.
  2. If a section exceeds the token budget, split it at sentence boundaries with
     `semantic_text_splitter`, counting tokens with the embedding model's own
     tokenizer (no model weights are loaded).
  3. Tables are NEVER merged with surrounding prose; each table is its own
     chunk type with the markdown body and a JSON preview retained in payload.

The chunk model is deliberately storage-agnostic — the embedder reads `text`
and the Qdrant upserter reads `payload`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from semantic_text_splitter import TextSplitter

from app.retrieval.embedder import embedding_tokenizer

ChunkKind = Literal["text", "table"]

# Leaves room for the context header within a 512-token encoder window.
DEFAULT_MAX_TOKENS = 448
DEFAULT_OVERLAP_TOKENS = 48


@dataclass(slots=True)
class Chunk:
    text: str
    kind: ChunkKind
    page: int
    section_path: tuple[str, ...] = ()
    table_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    payload: dict[str, object] = field(default_factory=dict)


@lru_cache(maxsize=8)
def _get_splitter(max_tokens: int, overlap: int) -> TextSplitter:
    return TextSplitter.from_huggingface_tokenizer(
        embedding_tokenizer(), max_tokens, overlap=overlap
    )


def chunk_text(
    text: str,
    *,
    page: int,
    section_path: tuple[str, ...] = (),
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split prose at sentence boundaries into chunks of at most `max_tokens` model tokens."""
    if not text.strip():
        return []
    splitter = _get_splitter(max_tokens, overlap)
    pieces = splitter.chunks(text)
    return [
        Chunk(
            text=piece,
            kind="text",
            page=page,
            section_path=section_path,
        )
        for piece in pieces
    ]


def contextualize(
    text: str, *, doc_name: str, section_path: tuple[str, ...] | list[str] = ()
) -> str:
    """Prefix a chunk with its document and section so it embeds with its context."""
    header = [f"Document: {doc_name}"] if doc_name else []
    if section_path:
        header.append("Section: " + " > ".join(section_path))
    return "\n".join([*header, "", text]) if header else text


def make_table_chunk(
    *,
    markdown: str,
    page: int,
    table_id: str,
    section_path: tuple[str, ...] = (),
    bbox: tuple[float, float, float, float] | None = None,
    json_preview: list[dict[str, object]] | None = None,
) -> Chunk:
    return Chunk(
        text=markdown,
        kind="table",
        page=page,
        section_path=section_path,
        table_id=table_id,
        bbox=bbox,
        payload={"table_preview": json_preview or []},
    )
