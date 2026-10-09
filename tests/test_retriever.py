"""Tests for Phase 5 - retrieval (real embeddings, model cached; no LLM)."""
from __future__ import annotations

import pytest

from src.chunker import chunk_pages
from src.embeddings import get_embedding_model
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


def _build_store() -> FaissVectorStore:
    """Index three topically distinct chunks for semantic tests."""
    model = get_embedding_model()
    pages = [
        {"text": "The company's revenue in 2024 was ten million dollars, up from eight million in 2023. Growth was driven by cloud subscriptions.", "page": 1},
        {"text": "Python is a high-level programming language known for readable syntax, dynamic typing, and a vast standard library.", "page": 2},
        {"text": "Sourdough bread requires flour, water, salt, and a live starter culture fermented slowly over twelve hours.", "page": 3},
    ]
    chunks = chunk_pages(pages, document="topics.pdf", chunk_size=500, chunk_overlap=0)
    texts = [str(c["text"]) for c in chunks]
    store = FaissVectorStore(dim=model.dim)
    store.add(model.embed_documents(texts), chunks)
    return store


def test_revenue_question_retrieves_revenue_chunk():
    retriever = Retriever(_build_store())
    hits = retriever.retrieve("What is the company's revenue?", top_k=3)
    assert len(hits) == 3
    assert "revenue" in str(hits[0]["text"]).lower()
    assert hits[0]["page"] == 1


def test_programming_question_retrieves_python_chunk():
    retriever = Retriever(_build_store())
    hits = retriever.retrieve("Which programming language has readable syntax?", top_k=3)
    assert "python" in str(hits[0]["text"]).lower()
    assert hits[0]["page"] == 2


def test_default_top_k_and_override():
    retriever = Retriever(_build_store(), default_top_k=2)
    assert retriever.default_top_k == 2
    assert len(retriever.retrieve("revenue?", top_k=None)) == 2
    assert len(retriever.retrieve("revenue?", top_k=1)) == 1


def test_empty_question_raises():
    retriever = Retriever(_build_store())
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve("   ")


def test_invalid_top_k_raises():
    retriever = Retriever(_build_store())
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("revenue?", top_k=0)
    with pytest.raises(ValueError, match="default_top_k"):
        Retriever(_build_store(), default_top_k=0)


def test_empty_store_raises():
    retriever = Retriever(FaissVectorStore(dim=get_embedding_model().dim))
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("revenue?")
