"""Tests for Phase 9 - Streamlit chat UI.

Strategy: AppTest in streamlit 1.54 cannot drive file_uploader
(renders as UnknownElement, no set_value API), so:
  - static UI (title, empty state, disabled input) is tested via AppTest;
  - the upload policy is tested directly via pending_uploads() and
    ingest_upload();
  - the chat flow is tested by injecting a pre-seeded pipeline through
    AppTest session_state, then driving the chat input;
  - conversation memory (Phase 11), document scope and the buttons are
    tested with a stubbed LLM so what reaches the pipeline can be
    asserted exactly; one test runs the real LLM end to end.
"""
from __future__ import annotations

import io

import pytest
from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

import src.rag_pipeline as pipeline_mod
from app import (
    STARTER_QUESTIONS,
    ingest_upload,
    pending_uploads,
    transcript_markdown,
    upload_key,
)
from src.rag_pipeline import RagPipeline

REVENUE = "Revenue in 2024 was ten million dollars from cloud sales."
PYTHON = "Python is a programming language with readable syntax."


def _pdf_bytes(text: str = REVENUE) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 750, text)
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


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def _ask(at: AppTest, question: str) -> None:
    at.chat_input[0].set_value(question).run(timeout=180)
    assert not at.exception, at.exception


def _text(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def _seeded_with_stub(monkeypatch, two_docs: bool = False) -> tuple[AppTest, list]:
    """Seeded app whose LLM is a stub recording what the pipeline sends it."""
    calls: list = []

    def fake_stream(question, hits, history=None, **kwargs):
        calls.append(
            {
                "question": question,
                "history": [t["question"] for t in history or []],
                "documents": sorted({h["document"] for h in hits}),
            }
        )
        return iter(["STUB ", f"ANSWER {len(calls)}"])

    monkeypatch.setattr(pipeline_mod, "stream_answer", fake_stream)
    seed = RagPipeline()
    seed.ingest_documents([("ui_demo.pdf", _pdf_bytes())])
    if two_docs:
        seed.ingest_documents([("notes.pdf", _pdf_bytes(PYTHON))])
    return _run(seeded=seed), calls


def test_boot_shows_title_and_empty_state():
    at = _run()
    assert at.title[0].value == "PDF RAG Assistant"
    assert any("Upload one or more PDFs" in str(i.value) for i in at.info)


def test_chat_disabled_without_documents():
    at = _run()
    assert at.chat_input[0].disabled is True
    assert not any(b.key == "regenerate" for b in at.button)


def test_pending_uploads_policy():
    pipe = RagPipeline()
    a, b = _FakeUpload("a.pdf"), _FakeUpload("b.pdf")
    statuses: dict = {}
    assert pending_uploads([a, b], statuses, False) == [a, b]
    assert pending_uploads([], statuses, False) == []
    assert pending_uploads(None, statuses, False) == []

    statuses[upload_key(a)] = ingest_upload(pipe, a, False)
    assert statuses[upload_key(a)][0] == "indexed"
    # Already handled: not re-indexed on later reruns, even with replace on.
    assert pending_uploads([a, b], statuses, False) == [b]
    assert pending_uploads([a, b], statuses, True) == [b]

    # Removed by the user: stays out until re-indexing is ticked.
    statuses[upload_key(a)] = ("removed", "Removed a.pdf.")
    assert pending_uploads([a], statuses, False) == []
    assert pending_uploads([a], statuses, True) == [a]


def test_duplicate_name_reported_then_retried_with_replace():
    pipe = RagPipeline()
    first, again = _FakeUpload("a.pdf"), _FakeUpload("a.pdf")
    again.file_id = "second-upload"
    statuses = {upload_key(first): ingest_upload(pipe, first, False)}

    statuses[upload_key(again)] = ingest_upload(pipe, again, False)
    kind, message = statuses[upload_key(again)]
    assert kind == "duplicate" and "a.pdf" in message
    assert pending_uploads([again], statuses, False) == []
    assert pending_uploads([again], statuses, True) == [again]

    assert ingest_upload(pipe, again, True)[0] == "indexed"
    assert pipe.documents == ["a.pdf"]


def test_invalid_upload_is_skipped_with_reason():
    class _Bad(_FakeUpload):
        def getvalue(self) -> bytes:
            return b"not a pdf"

    kind, message = ingest_upload(RagPipeline(), _Bad("bad.pdf"), False)
    assert kind == "skipped"
    assert "bad.pdf" in message and "Invalid PDF" in message


def test_transcript_lists_questions_answers_and_sources():
    history = [
        {
            "question": "What was revenue?",
            "answer": "Ten million dollars.",
            "sources": [{"document": "a.pdf", "page": 3, "score": 0.71234}],
        }
    ]
    text = transcript_markdown(history)
    assert "## What was revenue?" in text
    assert "Ten million dollars." in text
    assert "- a.pdf - Page 3 (score 0.712)" in text
    assert transcript_markdown([]).startswith("# PDF RAG Assistant")


def test_live_chat_streams_grounded_answer_with_sources():
    """Real LLM end to end. Skips if the LLM backend is unavailable."""
    seed = RagPipeline()
    seed.ingest_documents([("ui_demo.pdf", _pdf_bytes())])
    at = _run(seeded=seed)
    assert at.chat_input[0].disabled is False
    _ask(at, "What was the revenue in 2024?")
    if at.error:
        pytest.skip(f"LLM unavailable: {at.error[0].value}")
    full = _text(at)
    assert "ten million" in full.lower()
    assert "ui_demo.pdf" in full
    assert "Page 1" in full
    assert any("Sources (1)" in str(e.label) for e in at.expander)
    assert [m.name for m in at.chat_message] == ["user", "assistant"]


def test_follow_up_receives_history_and_chat_persists(monkeypatch):
    at, calls = _seeded_with_stub(monkeypatch)
    _ask(at, "What was the revenue in 2024?")
    _ask(at, "Where did it come from?")
    assert [c["history"] for c in calls] == [[], ["What was the revenue in 2024?"]]

    full = _text(at)
    assert "STUB ANSWER 1" in full and "STUB ANSWER 2" in full
    assert len(at.chat_message) == 4

    # An unrelated rerun (slider move) must not wipe the chat or re-ask.
    at.slider[0].set_value(3).run(timeout=180)
    assert "STUB ANSWER 2" in _text(at)
    assert len(calls) == 2


def test_memory_toggle_off_sends_no_history(monkeypatch):
    at, calls = _seeded_with_stub(monkeypatch)
    _ask(at, "What was the revenue in 2024?")
    next(c for c in at.checkbox if "memory" in c.label).uncheck()
    _ask(at, "Where did it come from?")
    assert [c["history"] for c in calls] == [[], []]


def test_starter_question_and_regenerate(monkeypatch):
    at, calls = _seeded_with_stub(monkeypatch)
    at.button(key="starter_0").click().run(timeout=180)
    assert not at.exception, at.exception
    assert [c["question"] for c in calls] == [STARTER_QUESTIONS[0]]
    assert not any(str(b.key).startswith("starter_") for b in at.button)

    at.button(key="regenerate").click().run(timeout=180)
    assert not at.exception, at.exception
    # Same question asked again, replacing (not adding to) the last turn.
    assert [c["question"] for c in calls] == [STARTER_QUESTIONS[0]] * 2
    assert calls[1]["history"] == []
    assert len(at.session_state["history"]) == 1
    assert "STUB ANSWER 2" in _text(at)


def test_search_in_limits_sources_to_selected_documents(monkeypatch):
    at, calls = _seeded_with_stub(monkeypatch, two_docs=True)
    _ask(at, "What was the revenue in 2024?")
    assert calls[0]["documents"] == ["notes.pdf", "ui_demo.pdf"]

    at.multiselect[0].set_value(["notes.pdf"]).run(timeout=180)
    _ask(at, "What was the revenue in 2024?")
    assert calls[1]["documents"] == ["notes.pdf"]


def test_remove_document_button(monkeypatch):
    at, _ = _seeded_with_stub(monkeypatch, two_docs=True)
    at.button(key="rm_ui_demo.pdf").click().run(timeout=180)
    assert not at.exception, at.exception
    assert at.session_state["pipeline"].documents == ["notes.pdf"]
    assert len(at.multiselect) == 0  # filter hidden with a single document


def test_clear_buttons_reset_conversation_and_index(monkeypatch):
    at, _ = _seeded_with_stub(monkeypatch)
    _ask(at, "What was the revenue in 2024?")

    _button(at, "New conversation").click().run(timeout=180)
    assert at.session_state["history"] == []
    assert at.session_state["pipeline"].documents == ["ui_demo.pdf"]

    _ask(at, "What was the revenue in 2024?")
    generation = at.session_state["uploader_gen"]
    _button(at, "Clear index").click().run(timeout=180)
    assert not at.exception, at.exception
    assert at.session_state["pipeline"].documents == []
    assert at.session_state["history"] == []
    assert at.session_state["uploader_gen"] == generation + 1
    assert at.chat_input[0].disabled is True
