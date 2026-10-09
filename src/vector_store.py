"""FAISS vector index (Phase 4).

Problem solved:
    Given a query vector, find the most similar document vectors among
    thousands of chunks in milliseconds. Brute-force numpy comparison
    works for tiny demos but does not scale; FAISS is built for this.

Input:
    Document embeddings (n, dim) float32 from Phase 3 + one metadata dict
    per row (chunk_id, text, page, document).

Output:
    Top-K hits: [{chunk_id, text, page, document, score}] ranked best first.
    score = cosine similarity in [-1, 1] (higher = more similar).

Why IndexFlatIP + L2-normalize:
    Normalizing both sides turns inner product into cosine similarity,
    the standard metric for bge-style embeddings. IndexFlatIP is EXACT
    (no training, no approximation error) and ideal at this scale
    (hundreds-thousands of chunks). IVF/PQ quantization only matters at
    million-vector scale and would add complexity here.

What FAISS is NOT:
    A database. It stores bare float vectors with positional ids only.
    All metadata lives in `self._metadatas[i]` alongside row i. Search
    returns positions -> we map them back to chunk dicts + scores.

Connection:
    Phase 3 vectors -> add() once at ingest.
    Phase 5 retriever: embed_query -> search() -> chunks for the LLM.
"""

from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np

Metadata = Dict[str, object]  # {"chunk_id": int, "text": str, "page": int, "document": str}
Hit = Dict[str, object]  # Metadata + {"score": float}


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows (or a single vector) for cosine-via-IP search."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid div-by-zero for degenerate rows
    return (vectors / norms).astype(np.float32, copy=False)


class FaissVectorStore:
    """Exact cosine-similarity index with parallel metadata list."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        self._dim = int(dim)
        self._index = faiss.IndexFlatIP(self._dim)
        self._metadatas: List[Metadata] = []

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return self._index.ntotal

    def add(self, embeddings: np.ndarray, metadatas: List[Metadata]) -> None:
        """Add document vectors with their chunk metadata.

        Row i of `embeddings` corresponds to `metadatas[i]`; this 1:1
        alignment is what lets search positions map back to chunks.
        """
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(
                f"embeddings must have shape (n, {self._dim}); got {arr.shape}."
            )
        if len(metadatas) != arr.shape[0]:
            raise ValueError(
                f"{arr.shape[0]} vectors but {len(metadatas)} metadata dicts: "
                "must be 1:1."
            )
        if arr.shape[0] == 0:
            return
        self._index.add(_normalize(arr))
        self._metadatas.extend(dict(m) for m in metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Hit]:
        """Return top-K hits (best first) for one query vector."""
        if len(self) == 0:
            raise ValueError("Index is empty: ingest documents before searching.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(
                f"query dim {q.shape[1]} != index dim {self._dim}."
            )
        k = min(k, len(self))
        scores, positions = self._index.search(_normalize(q), k)
        hits: List[Hit] = []
        for score, pos in zip(scores[0], positions[0]):
            if int(pos) < 0:
                continue  # FAISS pads with -1 when k > ntotal (guarded above)
            hit = dict(self._metadatas[int(pos)])
            hit["score"] = float(score)
            hits.append(hit)
        return hits

    def clear(self) -> None:
        """Reset index + metadata (used by the UI reset button)."""
        self._index.reset()
        self._metadatas.clear()
