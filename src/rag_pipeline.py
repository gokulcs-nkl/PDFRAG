"""Central RAG pipeline (Phase 8).

Problem solved:
    Phases 1-7 built independent components; something must wire them in
    the right order with one ingest path and one query path, so that
    app.py (Phase 9) stays thin and the whole flow is testable as a unit.

Flow:
    ingest: PDF bytes -> extract_text_from_pdf -> chunk_pages (tagged
            with document name) -> embed_documents -> append to chunk /
            embedding memory -> rebuild FAISS index.
    query:  question -> Retriever (embed_query -> FAISS top-K) ->
            generate_answer -> {answer, sources}.

Input:
    ingest_documents([(document_name, pdf_source), ...], replace=False)
        pdf_source = path / bytes / file-like (anything pdf_processor
        accepts). Multi-PDF (Phase 10) falls out naturally: every chunk
        keeps its own document + page.
    query(question, top_k=None, history=None, documents=None)
        -> {"answer": str, "sources": [hits]}.
    query_stream(...) -> (hits, iterator of answer pieces) for UIs that
        show sources first and the answer as it is written.
    documents=[names] restricts retrieval to those documents.

Design notes:
    All chunks + their embeddings are kept in memory and the FlatIP
    index is rebuilt on each ingest. At this scale (hundreds-thousands
    of chunks) rebuild is milliseconds and it makes replace/clear
    trivially correct -- no fragile per-doc deletion inside FAISS.
    A persistent vector store is an explicit future improvement.

    Per-file failures (invalid PDF, no extractable text) are collected
    into the ingest report, not raised: one bad upload must not abort
    a multi-file batch (Phase 12 groundwork). Duplicate document names
    are skipped unless replace=True.

Conversation memory (Phase 11):
    The pipeline stays stateless about conversations: the caller owns
    the turn list and passes it to query(history=...). Retriever and
    index are untouched. The previous question is prepended to the
    retrieval query only, because a follow-up like "and the year
    before?" carries no searchable topic of its own; the LLM gets the
    recent turns as chat messages (see llm.py).

Connection:
    Phase 9 Streamlit calls only this class.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

from src.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    chunk_pages,
)
from src.embeddings import DEFAULT_MODEL as EMBED_MODEL
from src.embeddings import EmbeddingModel, get_embedding_model
from src.llm import Turn, generate_answer, stream_answer
from src.pdf_processor import PdfSource, extract_text_from_pdf
from src.retriever import DEFAULT_TOP_K, Hit, Retriever
from src.vector_store import FaissVectorStore

DocumentInput = Tuple[str, PdfSource]
IngestReport = Dict[str, object]  # {documents_added, skipped, chunks_added, total_chunks}
QueryResult = Dict[str, object]  # {"answer": str, "sources": List[Hit]}


class RagPipeline:
    """Single facade over extract -> chunk -> embed -> index -> retrieve -> LLM."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        top_k: int = DEFAULT_TOP_K,
        embedding_model: str = EMBED_MODEL,
        llm_model: str | None = None,
        llm_host: str | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = int(top_k)
        self._model: EmbeddingModel = get_embedding_model(embedding_model)
        self._llm_model = llm_model
        self._llm_host = llm_host
        self._chunks: List[Chunk] = []
        self._embeddings = np.zeros((0, self._model.dim), dtype=np.float32)
        self._store = FaissVectorStore(dim=self._model.dim)
        self._retriever = Retriever(self._store, self._model, default_top_k=self._top_k)
        self._documents: List[str] = []

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def documents(self) -> List[str]:
        return list(self._documents)

    def _rebuild_index(self) -> None:
        self._store.clear()
        if self._embeddings.shape[0]:
            self._store.add(self._embeddings, self._chunks)

    def ingest_documents(
        self, files: Iterable[DocumentInput], replace: bool = False
    ) -> IngestReport:
        """Ingest multiple PDFs; returns per-file outcome report."""
        documents_added: List[str] = []
        skipped: Dict[str, str] = {}
        chunks_added = 0
        for name, source in files:
            doc = (name or "").strip()
            if not doc:
                skipped[str(name)] = "Missing document name."
                continue
            if doc in self._documents and not replace:
                skipped[doc] = "Duplicate document: already indexed (use replace=True to re-index)."
                continue
            try:
                pages = extract_text_from_pdf(source)
            except (FileNotFoundError, ValueError) as exc:
                skipped[doc] = f"Invalid PDF: {exc}"
                continue
            if not pages:
                skipped[doc] = "No extractable text found (scanned/image-only PDF?)."
                continue
            new_chunks = chunk_pages(
                pages, doc, self._chunk_size, self._chunk_overlap
            )
            if not new_chunks:
                skipped[doc] = "No chunks produced (pages empty after cleaning?)."
                continue
            new_vecs = self._model.embed_documents([str(c["text"]) for c in new_chunks])
            # Drop the old copy only once the new one is ready to go in.
            if replace and doc in self._documents:
                mask = [str(c.get("document")) != doc for c in self._chunks]
                self._chunks = [c for c, keep in zip(self._chunks, mask) if keep]
                self._embeddings = self._embeddings[np.asarray(mask, dtype=bool)]
                self._documents = [d for d in self._documents if d != doc]
            self._chunks.extend(new_chunks)
            self._embeddings = (
                np.vstack([self._embeddings, new_vecs])
                if self._embeddings.shape[0]
                else new_vecs
            )
            self._documents.append(doc)
            documents_added.append(doc)
            chunks_added += len(new_chunks)
        self._rebuild_index()
        return {
            "documents_added": documents_added,
            "skipped": skipped,
            "chunks_added": chunks_added,
            "total_chunks": len(self._chunks),
        }

    @staticmethod
    def _retrieval_query(question: str, history: List[Turn] | None) -> str:
        """Prepend the previous question so follow-ups stay searchable."""
        if not history:
            return question
        previous = str(history[-1].get("question", "") or "").strip()
        return f"{previous} {question}" if previous else question

    def _retrieve(
        self,
        question: str,
        top_k: int | None,
        history: List[Turn] | None,
        documents: Sequence[str] | None,
    ) -> List[Hit]:
        if not self._chunks:
            raise ValueError("No documents indexed yet: upload PDFs before asking questions.")
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        search = self._retrieval_query(question, history)
        if not documents:
            return self._retriever.retrieve(search, top_k=top_k)
        k = self._top_k if top_k is None else top_k
        if k <= 0:
            raise ValueError("top_k must be a positive integer.")
        # Exact flat index: rank everything, then keep the wanted documents.
        wanted = set(documents)
        ranked = self._retriever.retrieve(search, top_k=len(self._chunks))
        return [h for h in ranked if h.get("document") in wanted][:k]

    def query(
        self,
        question: str,
        top_k: int | None = None,
        history: List[Turn] | None = None,
        documents: Sequence[str] | None = None,
    ) -> QueryResult:
        """Retrieve top-K chunks and generate a grounded answer.

        `history` is the caller-owned list of earlier
        {"question", "answer"} turns; None/[] means a standalone question.
        `documents` limits retrieval to those document names (None = all).
        """
        hits = self._retrieve(question, top_k, history, documents)
        answer = generate_answer(
            question,
            hits,
            model=self._llm_model,
            host=self._llm_host,
            history=history,
        )
        return {"answer": answer, "sources": hits}

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        history: List[Turn] | None = None,
        documents: Sequence[str] | None = None,
    ) -> Tuple[List[Hit], Iterator[str]]:
        """Like query(), but return (sources, answer pieces as they arrive)."""
        hits = self._retrieve(question, top_k, history, documents)
        pieces = stream_answer(
            question,
            hits,
            model=self._llm_model,
            host=self._llm_host,
            history=history,
        )
        return hits, pieces

    def document_stats(self) -> List[Dict[str, object]]:
        """Per-document [{"document", "pages", "chunks"}] in ingest order."""
        stats = []
        for doc in self._documents:
            own = [c for c in self._chunks if c.get("document") == doc]
            stats.append(
                {
                    "document": doc,
                    "pages": len({c.get("page") for c in own}),
                    "chunks": len(own),
                }
            )
        return stats

    def remove_document(self, document: str) -> bool:
        """Drop one document from the index; False if it was not indexed."""
        if document not in self._documents:
            return False
        mask = [str(c.get("document")) != document for c in self._chunks]
        self._chunks = [c for c, keep in zip(self._chunks, mask) if keep]
        self._embeddings = self._embeddings[np.asarray(mask, dtype=bool)]
        self._documents = [d for d in self._documents if d != document]
        self._rebuild_index()
        return True

    def clear(self) -> None:
        """Forget all documents (UI reset button)."""
        self._chunks.clear()
        self._embeddings = np.zeros((0, self._model.dim), dtype=np.float32)
        self._documents.clear()
        self._rebuild_index()
