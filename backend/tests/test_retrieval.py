from __future__ import annotations

import hashlib
import math
import re
import uuid

import pytest

from app.retrieval import orchestrator, qdrant_client
from app.retrieval.reranker import RerankResult
from app.settings import get_infra

DIM = 64


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bucket(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM


def fake_dense(text: str) -> list[float]:
    vec = [0.0] * DIM
    for tok in _tokens(text):
        vec[_bucket(tok)] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def fake_sparse(text: str) -> tuple[list[int], list[float]]:
    counts: dict[int, float] = {}
    for tok in _tokens(text):
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 100_000
        counts[idx] = counts.get(idx, 0.0) + 1.0
    return list(counts), list(counts.values())


def passthrough_rerank(query: str, candidates: list, *, top_k: int = 6) -> list[RerankResult]:
    return [RerankResult(c.payload, c.score_before, c.score_before) for c in candidates][:top_k]


@pytest.fixture
def memory_index(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(get_infra(), "qdrant_mode", "memory")
    monkeypatch.setattr(qdrant_client, "_client", None)
    monkeypatch.setattr(qdrant_client, "dense_dim", lambda: DIM)
    monkeypatch.setattr(orchestrator, "embed_dense_query", fake_dense)
    monkeypatch.setattr(orchestrator, "embed_sparse_query", fake_sparse)
    monkeypatch.setattr(orchestrator, "rerank", passthrough_rerank)

    async def index(session_id: str, chunks: list[dict]) -> None:
        await qdrant_client.upsert_chunks(
            session_id,
            ids=[str(uuid.uuid4()) for _ in chunks],
            dense=[fake_dense(c["text"]) for c in chunks],
            sparse=[fake_sparse(c["text"]) for c in chunks],
            payloads=chunks,
        )

    return index


async def test_retrieve_respects_doc_filter(memory_index) -> None:
    await memory_index(
        "s1",
        [
            {"doc_id": "a", "doc_name": "a.pdf", "page": 1, "text": "Revenue rose to 10 million."},
            {"doc_id": "b", "doc_name": "b.pdf", "page": 2, "text": "Revenue fell to 4 million."},
        ],
    )
    everything = await orchestrator.retrieve(session_id="s1", query="revenue")
    only_b = await orchestrator.retrieve(session_id="s1", query="revenue", doc_ids=["b"])
    assert {c.doc_id for c in everything.citations} == {"a", "b"}
    assert {c.doc_id for c in only_b.citations} == {"b"}
