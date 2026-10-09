"""Tests for Phase 7 - LLM module (unit + optional live integration)."""
from __future__ import annotations

import pytest

from src.llm import (
    REFUSAL,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    generate_answer,
)

_HITS = [
    {"chunk_id": 0, "text": "Revenue in 2024 was ten million dollars.", "page": 1, "document": "annual.pdf", "score": 0.7},
    {"chunk_id": 1, "text": "Python is a programming language.", "page": 2, "document": "notes.pdf", "score": 0.4},
]


def test_context_labels_sources():
    ctx = build_context(_HITS)
    assert "[1] (annual.pdf - Page 1)" in ctx
    assert "ten million dollars" in ctx
    assert "[2] (notes.pdf - Page 2)" in ctx


def test_messages_ground_and_forbid_invention():
    msgs = build_messages("What was revenue?", _HITS)
    assert msgs[0]["role"] == "system"
    assert "ONLY the provided context" in msgs[0]["content"]
    assert "Do not invent facts" in msgs[0]["content"]
    assert "CONTEXT:" in msgs[1]["content"]
    assert "QUESTION:" in msgs[1]["content"]
    assert SYSTEM_PROMPT in msgs[0]["content"]


def test_empty_hits_refuses_without_server():
    # Must NOT touch the network: refusal is deterministic.
    assert generate_answer("Anything?", []) == REFUSAL


def test_empty_question_raises():
    with pytest.raises(ValueError, match="non-empty"):
        generate_answer("   ", _HITS)


def test_unreachable_host_raises_friendly():
    with pytest.raises(RuntimeError, match="Is Ollama running"):
        generate_answer("Q?", _HITS, host="http://127.0.0.1:1", timeout=5)


def test_unknown_model_raises_pull_hint():
    with pytest.raises(RuntimeError, match="ollama pull"):
        generate_answer("Q?", _HITS, model="no-such-model-xyz", timeout=30)


def test_live_grounded_answer():
    """Integration: real Ollama (gemma3:4b). Skips if server/model busy."""
    try:
        answer = generate_answer("What was the revenue in 2024?", _HITS, timeout=180)
    except RuntimeError as exc:
        pytest.skip(f"Ollama unavailable: {exc}")
    assert "ten million" in answer.lower()
    assert "python" not in answer.lower() or "ten million" in answer.lower()
