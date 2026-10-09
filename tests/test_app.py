"""Tests for Phase 9 - Streamlit UI.

Strategy: AppTest in streamlit 1.54 cannot drive file_uploader
(renders as UnknownElement, no set_value API), so:
  - static UI (title, empty state, disabled Ask) is tested via AppTest;
  - the upload-filter policy is tested directly via pending_uploads();
  - the full ask flow is tested by injecting a pre-seeded pipeline
    through AppTest session_state, then driving question + Ask.
"""
from __future__ import annotations

import io

from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

from app import pending_uploads
from src.rag_pipeline import RagPipeline


def _pdf_bytes() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, "Revenue in 2024 was ten million dollars from cloud sales.")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


class _FakeUpload:
    def __init__(self, name: str):
        self.name = name

    def getvalue(self) -> bytes:
        return _pdf_bytes()


def _run(seeded: RagPipeline | None = None) -> AppTest:
    # First run loads the FastEmbed model: allow beyond the 3s default.
    at = AppTest.from_file("app.py", default_timeout=180)
    if seeded is not None:
        at.session_state["pipeline"] = seeded
    at.run(timeout=180)
    assert not at.exception, at.exception
    return at


def test_boot_shows_title_and_empty_state():
    at = _run()
    assert at.title[0].value == "PDF RAG Assistant"
    assert any("Upload one or more PDFs" in str(i.value) for i in at.info)


def test_ask_disabled_without_documents():
    at = _run()
    assert at.button[0].disabled is True


def test_pending_uploads_policy():
    pipe = RagPipeline()
    files = [_FakeUpload("a.pdf"), _FakeUpload("b.pdf")]
    assert [f.name for f in pending_uploads(pipe, files, False)] == ["a.pdf", "b.pdf"]
    assert pending_uploads(pipe, [], False) == []
    pipe.ingest_documents([("a.pdf", _pdf_bytes())])
    assert [f.name for f in pending_uploads(pipe, files, False)] == ["b.pdf"]
    assert [f.name for f in pending_uploads(pipe, files, True)] == ["a.pdf", "b.pdf"]


def test_ask_returns_grounded_answer_and_sources():
    seed = RagPipeline()
    seed.ingest_documents([("ui_demo.pdf", _pdf_bytes())])
    at = _run(seeded=seed)
    assert at.button[0].disabled is False
    assert any("ui_demo.pdf" in str(i.value) for i in at.info)
    at.text_input[0].set_value("What was the revenue in 2024?")
    at.run(timeout=180)
    at.button[0].click()
    at.run(timeout=180)
    assert not at.exception, at.exception
    full = " ".join(str(m.value) for m in at.markdown)
    assert "ten million" in full.lower()
    assert "ui_demo.pdf" in full
    assert "Page 1" in full
    assert any("snippet" in str(e.label).lower() for e in at.expander)
