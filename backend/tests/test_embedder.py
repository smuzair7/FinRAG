from __future__ import annotations

import pytest

from app.retrieval import embedder
from app.settings import get_infra


@pytest.mark.parametrize(
    ("model", "dim"),
    [("BAAI/bge-large-en-v1.5", 1024), ("BAAI/bge-small-en-v1.5", 384)],
)
def test_dense_dim_follows_configured_model(
    monkeypatch: pytest.MonkeyPatch, model: str, dim: int
) -> None:
    monkeypatch.setattr(get_infra(), "embedding_model", model)
    assert embedder.dense_dim() == dim
