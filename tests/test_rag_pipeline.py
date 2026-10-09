"""Tests for Phase 8 - RAG pipeline (stubbed LLM + one live e2e)."""
from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

import src.rag_pipeline as pipeline_mod
from src.rag_pipeline import RagPipeline


def _pdf_bytes(pages: list[str]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        if text:
            c.drawString(72, 750, text[:95])
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def _blank_pdf() -> bytes:
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = BytesIO()
    w.write(buf)
    return buf.getvalue()


@pytest.fixture()
def docs() -> list[tuple[str, bytes]]:
    return [
        ("annual.pdf", _pdf_bytes(["Revenue in 2024 was ten million dollars, driven by cloud subscriptions."])),
        ("notes.pdf", _pdf_bytes(["Python is a programming language with readable syntax."])),
    ]


def test_ingest_two_pdfs(docs):
    pipe = RagPipeline(chunk_size=500, chunk_overlap=0)
    report = pipe.ingest_documents(docs)
    assert report["documents_added"] == ["annual.pdf", "notes.pdf"]
    assert report["skipped"] == {}
    assert report["total_chunks"] == pipe.num_chunks >= 2
    assert pipe.documents == ["annual.pdf", "notes.pdf"]


def test_duplicate_skipped_then_replaced(docs):
    pipe = RagPipeline(chunk_size=500, chunk_overlap=0)
    pipe.ingest_documents(docs)
    before = pipe.num_chunks
    report = pipe.ingest_documents([docs[0]])
    assert "annual.pdf" in report["skipped"]
    assert "Duplicate" in report["skipped"]["annual.pdf"]
    assert pipe.num_chunks == before
    report = pipe.ingest_documents([docs[0]], replace=True)
    assert report["documents_added"] == ["annual.pdf"]
    assert pipe.num_chunks == before  # same content swapped back in
    assert pipe.documents == ["notes.pdf", "annual.pdf"]


def test_blank_and_invalid_pdfs_reported_not_raised():
    pipe = RagPipeline(chunk_size=500, chunk_overlap=0)
    report = pipe.ingest_documents([
        ("blank.pdf", _blank_pdf()),
        ("evil.txt", b"not a pdf at all"),
    ])
    assert report["documents_added"] == []
    assert "No extractable text" in report["skipped"]["blank.pdf"]
    assert "Invalid PDF" in report["skipped"]["evil.txt"]
    assert pipe.num_chunks == 0


def test_query_before_ingest_raises():
    pipe = RagPipeline()
    with pytest.raises(ValueError, match="No documents indexed"):
        pipe.query("Anything?")


def test_query_wiring_with_stubbed_llm(docs, monkeypatch):
    pipe = RagPipeline(chunk_size=500, chunk_overlap=0)
    pipe.ingest_documents(docs)
    seen: dict = {}

    def fake_generate(question, hits, **kwargs):
        seen["n_hits"] = len(hits)
        return f"STUB: {question} <- {hits[0]['document']} p{hits[0]['page']}"

    monkeypatch.setattr(pipeline_mod, "generate_answer", fake_generate)
    result = pipe.query("What was the revenue?", top_k=1)
    assert seen["n_hits"] == 1
    assert result["answer"].startswith("STUB:")
    assert "annual.pdf" in result["answer"]
    assert result["sources"][0]["page"] == 1


def test_clear_resets_everything(docs):
    pipe = RagPipeline(chunk_size=500, chunk_overlap=0)
    pipe.ingest_documents(docs)
    pipe.clear()
    assert pipe.num_chunks == 0
    assert pipe.documents == []
    with pytest.raises(ValueError, match="No documents indexed"):
        pipe.query("Q?")


def test_live_end_to_end(docs):
    """Full path with real Ollama. Skips if server/model unavailable."""
    pipe = RagPipeline(chunk_size=500, chunk_overlap=0)
    pipe.ingest_documents(docs)
    try:
        result = pipe.query("What was the revenue in 2024?", top_k=2)
    except RuntimeError as exc:
        pytest.skip(f"Ollama unavailable: {exc}")
    assert "ten million" in str(result["answer"]).lower()
    assert result["sources"][0]["document"] == "annual.pdf"
