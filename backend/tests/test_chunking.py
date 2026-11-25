from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from app.ingestion import chunking
from app.ingestion.chunking import chunk_text, contextualize, split_markdown_table

SECTION = " ".join(f"Revenue grew {i} percent in the quarter." for i in range(300))


def _word_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


@pytest.fixture(autouse=True)
def word_tokens(monkeypatch: pytest.MonkeyPatch):
    chunking._get_splitter.cache_clear()
    monkeypatch.setattr(chunking, "embedding_tokenizer", _word_tokenizer)
    yield
    chunking._get_splitter.cache_clear()


def test_chunk_text_honours_token_budget_per_call() -> None:
    chunk_text(SECTION, page=1)
    small = chunk_text(SECTION, page=1, max_tokens=60, overlap=6)
    assert len(small) > 1
    assert max(len(c.text.split()) for c in small) <= 60


def test_each_chunk_carries_only_its_own_text() -> None:
    chunks = chunk_text(SECTION, page=1, max_tokens=60, overlap=0)
    assert "".join(c.text for c in chunks).replace(" ", "") == SECTION.replace(" ", "")


def test_contextualize_prefixes_document_and_section() -> None:
    out = contextualize(
        "Up 12% year over year.", doc_name="acme-10k.pdf", section_path=("MD&A", "Revenue")
    )
    assert out == "Document: acme-10k.pdf\nSection: MD&A > Revenue\n\nUp 12% year over year."


def test_contextualize_without_metadata_is_identity() -> None:
    assert contextualize("plain", doc_name="") == "plain"


TABLE = "| Line item | FY2024 | FY2023 |\n| --- | --- | --- |\n" + "\n".join(
    f"| Item {i} | {i * 10} | {i * 9} |" for i in range(40)
)


def test_large_table_splits_into_row_groups_with_header() -> None:
    parts = split_markdown_table(TABLE, max_tokens=60)
    assert len(parts) > 1
    assert all(p.startswith("| Line item | FY2024 | FY2023 |\n| --- |") for p in parts)
    assert all(len(p.split()) <= 60 for p in parts)
    rows = [line for p in parts for line in p.splitlines()[2:]]
    assert rows == TABLE.splitlines()[2:]


def test_small_table_is_kept_whole() -> None:
    assert split_markdown_table(TABLE, max_tokens=10_000) == [TABLE]
