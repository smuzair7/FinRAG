"""Retrieval evaluation on a small fixed corpus of filing-style passages.

Indexes eval/corpus.jsonl into an in-memory Qdrant with the configured dense,
sparse and reranker models, runs every query in eval/queries.jsonl through the
production retrieval orchestrator and reports Recall@1, Recall@5 and MRR@10.

    python -m eval.run_retrieval_eval [--no-context] [--no-rerank]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from app.ingestion.chunking import contextualize
from app.retrieval import orchestrator, qdrant_client, reranker
from app.retrieval.embedder import embed_dense, embed_sparse
from app.retrieval.reranker import RerankResult
from app.settings import get_infra

EVAL_DIR = Path(__file__).parent
SESSION = "eval"
RECALL_DEPTH = 5
MRR_DEPTH = 10


def _load(name: str) -> list[dict]:
    with open(EVAL_DIR / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _no_context(text: str, **_: object) -> str:
    return text


def _passthrough_rerank(query: str, candidates: list, *, top_k: int = 6) -> list[RerankResult]:
    return [RerankResult(c.payload, c.score_before, c.score_before) for c in candidates][:top_k]


async def _index(corpus: list[dict], *, with_context: bool) -> None:
    texts = [
        contextualize(p["text"], doc_name=p["doc_name"], section_path=p["section_path"])
        if with_context
        else p["text"]
        for p in corpus
    ]
    await qdrant_client.upsert_chunks(
        SESSION,
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, p["id"])) for p in corpus],
        dense=embed_dense(texts),
        sparse=embed_sparse(texts),
        payloads=[
            {
                "passage_id": p["id"],
                "doc_id": p["doc_name"],
                "doc_name": p["doc_name"],
                "section_path": p["section_path"],
                "page": 1,
                "text": p["text"],
            }
            for p in corpus
        ],
    )


async def evaluate(*, with_context: bool, with_rerank: bool) -> dict[str, float]:
    get_infra().qdrant_mode = "memory"
    if not with_context:
        reranker.contextualize = _no_context
    if not with_rerank:
        orchestrator.rerank = _passthrough_rerank

    await _index(_load("corpus.jsonl"), with_context=with_context)
    queries = _load("queries.jsonl")

    hits_at_1 = hits_at_5 = reciprocal_ranks = 0.0
    for q in queries:
        result = await orchestrator.retrieve(session_id=SESSION, query=q["query"], top_k=MRR_DEPTH)
        ranked = [p["passage_id"] for p in result.raw_payloads]
        relevant = set(q["relevant"])
        rank = next((i for i, pid in enumerate(ranked, start=1) if pid in relevant), None)
        hits_at_1 += rank == 1
        hits_at_5 += rank is not None and rank <= RECALL_DEPTH
        reciprocal_ranks += 1 / rank if rank else 0.0

    n = len(queries)
    return {"recall@1": hits_at_1 / n, "recall@5": hits_at_5 / n, "mrr@10": reciprocal_ranks / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-context", action="store_true", help="index chunks without headers")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    args = parser.parse_args()
    scores = asyncio.run(evaluate(with_context=not args.no_context, with_rerank=not args.no_rerank))
    print(json.dumps({k: round(v, 3) for k, v in scores.items()}))


if __name__ == "__main__":
    main()
