"""Tests for Phase 3 - FastEmbed embeddings."""
from __future__ import annotations

import numpy as np

from src import embeddings
from src.embeddings import EmbeddingModel, get_embedding_model


def test_document_embedding_shape_and_dtype():
    model = get_embedding_model()
    vecs = model.embed_documents(["Hello world", "Revenue was ten million dollars."])
    assert vecs.shape == (2, model.dim)
    assert vecs.dtype == np.float32
    assert model.dim == 384


def test_query_matches_document_dim():
    model = get_embedding_model()
    q = model.embed_query("What was the revenue?")
    assert q.shape == (model.dim,)
    assert q.dtype == np.float32


def test_singleton_reused():
    assert get_embedding_model() is get_embedding_model()


def test_empty_documents_returns_empty_matrix():
    model = get_embedding_model()
    vecs = model.embed_documents([])
    assert vecs.shape == (0, model.dim)


def test_empty_query_raises():
    model = get_embedding_model()
    try:
        model.embed_query("   ")
    except ValueError:
        return
    raise AssertionError("expected ValueError for blank query")


def test_convenience_wrappers_share_model():
    vecs = embeddings.embed_documents(["wrapper check"])
    q = embeddings.embed_query("wrapper check query")
    assert vecs.shape[1] == q.shape[0] == 384
