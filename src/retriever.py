"""Document retrieval (Phase 5).

Problem solved:
    Turn a user's natural-language question into the handful of document
    chunks most likely to contain its answer -- without involving any LLM.
    Keeping retrieval LLM-independent means it can be tested, tuned
    (chunking, top_k), and debugged on its own (Phase 6 gate).

Pipeline:
    Question -> embed_query (SAME FastEmbed model as docs) -> FAISS
    cosine search -> top-K chunk dicts.

Input:
    question string + optional top_k override.

Output:
    [{chunk_id, text, page, document, score}] ranked best first.
    score = cosine similarity in [-1, 1].

Why a thin class:
    Retrieval is just embed + search; all index state lives in
    FaissVectorStore, all encoding in EmbeddingModel. Retriever only
    wires them with validation and a configurable default_top_k
    (spec suggests 3-5; default 5).

Connection:
    Phase 6 exercises this directly (print top-K, check relevance).
    Phase 8 pipeline calls retrieve() then passes hits to the LLM.
"""

from __future__ import annotations

from typing import Dict, List

from src.embeddings import EmbeddingModel, get_embedding_model
from src.vector_store import FaissVectorStore

Hit = Dict[str, object]  # {chunk_id, text, page, document, score}

DEFAULT_TOP_K = 5


class Retriever:
    """Question -> chunks via shared embedding model + FAISS store."""

    def __init__(
        self,
        store: FaissVectorStore,
        model: EmbeddingModel | None = None,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer.")
        self._store = store
        self._model = model or get_embedding_model()
        self._default_top_k = int(default_top_k)

    @property
    def default_top_k(self) -> int:
        return self._default_top_k

    def retrieve(self, question: str, top_k: int | None = None) -> List[Hit]:
        """Return top-K chunks relevant to `question` (best first)."""
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        k = self._default_top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k must be a positive integer.")
        query_vector = self._model.embed_query(question.strip())
        return self._store.search(query_vector, k=k)
