"""Tests for Phase 4 - FAISS vector store (synthetic vectors, no model needed)."""
from __future__ import annotations

import numpy as np
import pytest

from src.vector_store import FaissVectorStore


def _store(dim: int = 4) -> FaissVectorStore:
    return FaissVectorStore(dim=dim)


def _meta(i: int) -> dict:
    return {"chunk_id": i, "text": f"chunk {i}", "page": i + 1, "document": "d.pdf"}


def test_add_and_len():
    store = _store()
    vecs = np.eye(4, dtype=np.float32)
    store.add(vecs, [_meta(i) for i in range(4)])
    assert len(store) == 4
    assert store.dim == 4


def test_search_returns_most_similar_first():
    store = _store()
    vecs = np.eye(4, dtype=np.float32)  # orthogonal unit vectors
    store.add(vecs, [_meta(i) for i in range(4)])
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=2)
    assert hits[0]["chunk_id"] == 0
    assert hits[0]["score"] == pytest.approx(1.0)
    assert hits[0]["text"] == "chunk 0"
    assert hits[0]["page"] == 1
    assert hits[0]["document"] == "d.pdf"


def test_scores_ranked_best_first():
    store = _store(dim=2)
    vecs = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float32)
    store.add(vecs, [_meta(i) for i in range(3)])
    hits = store.search(np.array([1, 0], dtype=np.float32), k=3)
    assert [h["chunk_id"] for h in hits] == [0, 1, 2]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_clamped_when_larger_than_index():
    store = _store()
    store.add(np.eye(4, dtype=np.float32)[:2], [_meta(0), _meta(1)])
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=5)
    assert len(hits) == 2


def test_empty_index_search_raises():
    store = _store()
    with pytest.raises(ValueError, match="empty"):
        store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=3)


def test_dim_mismatch_raises():
    store = _store(dim=4)
    with pytest.raises(ValueError, match="shape"):
        store.add(np.ones((2, 3), dtype=np.float32), [_meta(0), _meta(1)])
    store.add(np.eye(4, dtype=np.float32)[:1], [_meta(0)])
    with pytest.raises(ValueError, match="dim"):
        store.search(np.ones(3, dtype=np.float32), k=1)


def test_metadata_length_mismatch_raises():
    store = _store()
    with pytest.raises(ValueError, match="1:1"):
        store.add(np.eye(4, dtype=np.float32)[:2], [_meta(0)])


def test_clear_resets_store():
    store = _store()
    store.add(np.eye(4, dtype=np.float32), [_meta(i) for i in range(4)])
    store.clear()
    assert len(store) == 0
    with pytest.raises(ValueError, match="empty"):
        store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=1)


def test_invalid_k_raises():
    store = _store()
    store.add(np.eye(4, dtype=np.float32)[:1], [_meta(0)])
    with pytest.raises(ValueError, match="k must be"):
        store.search(np.array([1, 0, 0, 0], dtype=np.float32), k=0)
