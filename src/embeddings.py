"""Embedding generation with FastEmbed (Phase 3).

Problem solved:
    Text chunks are strings; similarity search needs fixed-size numeric
    vectors where similar meanings sit close together. This module maps
    chunk text -> dense vectors.

Input:
    Document chunk strings (list) or a single user question (str).

Output:
    numpy float32 arrays: (n_chunks, dim) for documents, (dim,) for query.
    Model: BAAI/bge-small-en-v1.5, dim=384.

Why FastEmbed:
    Lightweight ONNX-based embeddings, no API key, runs locally, fast on
    CPU. Same model MUST encode docs and queries or similarity is
    meaningless -- enforced here by routing both through one singleton.

Why singleton:
    Loading the model downloads weights + builds an ONNX session
    (seconds + ~100MB). Recreating it per query would make every question
    slow. `get_embedding_model()` loads once and reuses it.

Normalization note:
    Raw vectors are returned here. Cosine similarity is achieved in
    Phase 4 (`vector_store.py`) via L2-normalize + IndexFlatIP, keeping
    this module pure (encode only).

Connection:
    Phase 2 chunks -> embed_documents -> Phase 4 FAISS index.
    User question -> embed_query -> Phase 4 search.
"""

from __future__ import annotations

import threading
from typing import List

import numpy as np
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EXPECTED_DIM = 384

_lock = threading.Lock()
_instance: "EmbeddingModel | None" = None


class EmbeddingModel:
    """Single-load FastEmbed wrapper for docs and queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name)
        # Cache dim from a probe embed so mismatches fail fast.
        probe = np.asarray(list(self._model.embed(["probe"])), dtype=np.float32)
        self.dim: int = int(probe.shape[1])

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed chunk texts -> (n, dim) float32, preserving input order.

        Empty/whitespace inputs are replaced with a single space so the
        output row count always matches the input (index <-> metadata
        alignment required by FAISS).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        cleaned = [t if t and t.strip() else " " for t in texts]
        vectors = list(self._model.embed(cleaned))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question -> (dim,) float32 with the SAME model."""
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty.")
        vectors = list(self._model.embed([text]))
        return np.asarray(vectors[0], dtype=np.float32)


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """Return the shared instance, creating it once (thread-safe)."""
    global _instance
    if _instance is not None and _instance.model_name == model_name:
        return _instance
    with _lock:
        if _instance is None or _instance.model_name != model_name:
            _instance = EmbeddingModel(model_name)
        return _instance


# Convenience wrappers used by pipeline/retriever (share the singleton).
def embed_documents(texts: List[str]) -> np.ndarray:
    return get_embedding_model().embed_documents(texts)


def embed_query(text: str) -> np.ndarray:
    return get_embedding_model().embed_query(text)
