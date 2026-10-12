"""Tests for Phase 7 - LLM module (unit + optional live integration)."""
from __future__ import annotations

import httpx
import pytest

from src.llm import (
    DEFAULT_API_BASE,
    MAX_HISTORY_ANSWER_CHARS,
    MAX_HISTORY_TURNS,
    REFUSAL,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    generate_answer,
    llm_settings,
    stream_answer,
)

_HITS = [
    {"chunk_id": 0, "text": "Revenue in 2024 was ten million dollars.", "page": 1, "document": "annual.pdf", "score": 0.7},
    {"chunk_id": 1, "text": "Python is a programming language.", "page": 2, "document": "notes.pdf", "score": 0.4},
]


@pytest.fixture(autouse=True)
def _local_provider(monkeypatch):
    """Default every test to the local Ollama provider, whatever .env says."""
    for key in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


class _FakeResponse:
    def __init__(self, status_code: int, lines: list[str]):
        self.status_code = status_code
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter(self._lines)


def _fake_api(monkeypatch, status_code: int = 200, lines: list[str] | None = None) -> dict:
    """Route the OpenAI-compatible provider to a canned response."""
    seen: dict = {}

    def fake_stream(method, url, json=None, headers=None, timeout=None):
        seen.update(method=method, url=url, json=json, headers=headers)
        return _FakeResponse(status_code, lines or [])

    monkeypatch.setenv("LLM_API_KEY", "sk-test-secret")
    monkeypatch.setattr(httpx, "stream", fake_stream)
    return seen


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


def test_no_history_leaves_prompt_unchanged():
    assert build_messages("Q?", _HITS, history=[]) == build_messages("Q?", _HITS)
    assert build_messages("Q?", _HITS)[0]["content"] == SYSTEM_PROMPT


def test_history_replayed_before_current_context():
    history = [{"question": "What was revenue?", "answer": "Ten million dollars."}]
    msgs = build_messages("And who reported it?", _HITS, history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "What was revenue?"
    assert msgs[2]["content"] == "Ten million dollars."
    # Grounding rule survives; context is attached to the latest turn only.
    assert msgs[0]["content"].startswith(SYSTEM_PROMPT)
    assert "CONTEXT:" in msgs[3]["content"] and "And who reported it?" in msgs[3]["content"]
    assert "CONTEXT:" not in msgs[1]["content"]


def test_history_is_capped_and_truncated():
    history = [{"question": f"Q{i}", "answer": "x" * 5000} for i in range(10)]
    history.append({"question": "incomplete", "answer": ""})  # dropped
    msgs = build_messages("Now?", _HITS, history)
    past = msgs[1:-1]
    assert len(past) == 2 * (MAX_HISTORY_TURNS - 1)
    assert past[0]["content"] == f"Q{10 - MAX_HISTORY_TURNS + 1}"
    assert all(len(m["content"]) <= MAX_HISTORY_ANSWER_CHARS for m in past)


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


def test_settings_default_to_local_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "some-model")
    cfg = llm_settings()
    assert cfg["provider"] == "ollama"
    assert cfg["model"] == "some-model"
    assert cfg["api_key"] == ""


def test_api_key_selects_openai_compatible_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-secret")
    cfg = llm_settings()
    assert cfg["provider"] == "openai"
    assert cfg["host"] == DEFAULT_API_BASE
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    cfg = llm_settings()
    assert (cfg["host"], cfg["model"]) == ("https://example.test/v1", "my-model")
    # An explicit provider wins over key auto-detection.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert llm_settings()["provider"] == "ollama"


def test_stream_refuses_on_empty_hits_and_validates_question():
    assert list(stream_answer("Anything?", [])) == [REFUSAL]
    with pytest.raises(ValueError, match="non-empty"):
        stream_answer("  ", _HITS)  # validated before any iteration


def test_openai_provider_streams_sse_and_sends_key(monkeypatch):
    seen = _fake_api(monkeypatch, lines=[
        ": keep-alive",
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"delta": {"content": "Ten "}}]}',
        'data: {"choices": [{"delta": {"content": "million."}}]}',
        'data: {"choices": [], "usage": {"total_tokens": 9}}',
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "ignored"}}]}',
    ])
    assert list(stream_answer("What was revenue?", _HITS)) == ["Ten ", "million."]
    assert seen["url"] == f"{DEFAULT_API_BASE}/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer sk-test-secret"}
    assert seen["json"]["stream"] is True
    assert seen["json"]["messages"][0]["content"] == SYSTEM_PROMPT


def test_openai_provider_empty_reply_becomes_refusal(monkeypatch):
    _fake_api(monkeypatch, lines=["data: [DONE]"])
    assert generate_answer("What was revenue?", _HITS) == REFUSAL


@pytest.mark.parametrize(
    "status, expected",
    [(401, "LLM_API_KEY"), (404, "LLM_MODEL"), (429, "rate limit"), (500, "HTTP 500")],
)
def test_openai_provider_errors_are_friendly_and_keep_key_secret(monkeypatch, status, expected):
    _fake_api(monkeypatch, status_code=status)
    with pytest.raises(RuntimeError, match=expected) as err:
        generate_answer("Q?", _HITS)
    assert "sk-test-secret" not in str(err.value)


def test_openai_provider_unreachable_raises_friendly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(RuntimeError, match="Cannot reach the LLM API"):
        generate_answer("Q?", _HITS, host="http://127.0.0.1:1/v1", timeout=5)


def test_live_openai_compatible_endpoint(monkeypatch):
    """Real HTTP + SSE path, using the local Ollama server's /v1 endpoint."""
    local = llm_settings()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", f"{local['host']}/v1")
    monkeypatch.setenv("LLM_MODEL", local["model"])
    try:
        pieces = list(stream_answer("What was the revenue in 2024?", _HITS, timeout=180))
    except RuntimeError as exc:
        pytest.skip(f"OpenAI-compatible endpoint unavailable: {exc}")
    assert len(pieces) > 1  # genuinely streamed
    assert "ten million" in "".join(pieces).lower()
