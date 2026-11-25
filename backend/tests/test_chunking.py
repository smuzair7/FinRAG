from __future__ import annotations

from app.ingestion.chunking import chunk_text, contextualize

SECTION = " ".join(f"Revenue grew {i} percent in the quarter." for i in range(300))


def test_chunk_text_honours_max_chars_per_call() -> None:
    chunk_text(SECTION, page=1)
    small = chunk_text(SECTION, page=1, max_chars=500, overlap=50)
    assert len(small) > 1
    assert max(len(c.text) for c in small) <= 500


def test_each_chunk_carries_only_its_own_text() -> None:
    chunks = chunk_text(SECTION, page=1, max_chars=500, overlap=0)
    assert "".join(c.text for c in chunks).replace(" ", "") == SECTION.replace(" ", "")


def test_contextualize_prefixes_document_and_section() -> None:
    out = contextualize(
        "Up 12% year over year.", doc_name="acme-10k.pdf", section_path=("MD&A", "Revenue")
    )
    assert out == "Document: acme-10k.pdf\nSection: MD&A > Revenue\n\nUp 12% year over year."


def test_contextualize_without_metadata_is_identity() -> None:
    assert contextualize("plain", doc_name="") == "plain"
