"""Tests for Phase 2 - chunking."""
from __future__ import annotations

import pytest

from src.chunker import (
    chunk_pages,
    split_sentences,
    split_text_into_chunks,
)


def test_chunk_creation_and_metadata():
    pages = [
        {"text": "First page content. It has two sentences.", "page": 1},
        {"text": "Second page content here.", "page": 3},  # note: page 2 skipped upstream
    ]
    chunks = chunk_pages(pages, document="example.pdf", chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c["document"] == "example.pdf" for c in chunks)
    assert chunks[0]["page"] == 1
    assert chunks[-1]["page"] == 3
    assert all(set(c.keys()) == {"chunk_id", "text", "page", "document"} for c in chunks)


def test_chunk_size_bounded():
    text = " ".join(f"Sentence number {i} about revenue and growth." for i in range(30))
    chunk_size, overlap = 200, 40
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    assert len(chunks) > 1
    assert all(len(c) <= chunk_size + overlap for c in chunks)


def test_overlap_shares_context():
    text = " ".join(f"Sentence number {i} with distinct content here." for i in range(20))
    chunks = split_text_into_chunks(text, chunk_size=150, chunk_overlap=50)
    assert len(chunks) >= 2
    # Tail of chunk 0 should appear at the start of chunk 1 (word-boundary adjusted).
    tail = chunks[0][-50:]
    assert any(word in chunks[1][:80] for word in tail.split() if len(word) > 4)


def test_no_overlap_mode():
    text = "Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta. Theta."
    chunks = split_text_into_chunks(text, chunk_size=20, chunk_overlap=0)
    assert len(chunks) >= 2
    # No forced sharing: first chunk must not leak into second beyond natural repeats.
    assert chunks[0] != chunks[1]


def test_sentence_awareness():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = split_text_into_chunks(text, chunk_size=40, chunk_overlap=0)
    # With short sentences and chunk_size=40, no chunk should cut mid-sentence.
    for c in chunks:
        stripped = c.strip()
        assert stripped.endswith((".", "!", "?")) or c == chunks[-1]


def test_long_sentence_hard_split():
    long_sentence = "Revenue was " + "very " * 100 + "high."
    chunks = split_text_into_chunks(long_sentence, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_empty_input_returns_empty():
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   ") == []
    assert chunk_pages([], document="x.pdf") == []


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        split_text_into_chunks("hello world, this is a test", 10, 5)
    with pytest.raises(ValueError):
        split_text_into_chunks("hello world, this is a test", 100, 100)
    with pytest.raises(ValueError):
        chunk_pages([{"text": "hi there hello world test", "page": 1}], document="")


def test_end_to_end_pdf_to_chunks():
    """Phase 1 -> Phase 2 wiring: chunk_id global, pages preserved."""
    pages = [
        {"text": "Revenue in 2024 was ten million dollars. Costs were low.", "page": 1},
        {"text": "Revenue in 2023 was eight million dollars. Growth continued.", "page": 2},
    ]
    chunks = chunk_pages(pages, document="annual.pdf", chunk_size=60, chunk_overlap=10)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert {c["page"] for c in chunks} == {1, 2}
