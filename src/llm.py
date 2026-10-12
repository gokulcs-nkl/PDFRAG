"""LLM answer generation (Phase 7).

Problem solved:
    Retrieval finds relevant chunks, but users want a direct ANSWER, not
    a pile of excerpts. This module turns (question + chunks) into a
    grounded natural-language answer.

Input:
    question string + retriever hits [{text, page, document, score}].

Output:
    Answer string (generate_answer) or an iterator of text pieces
    (stream_answer) so the UI can show the answer as it is written.
    Empty hits -> deterministic refusal sentence (no LLM call wasted).

Prompt (per spec):
    SYSTEM: document QA assistant; answer ONLY from context; if absent,
    say the information is not available; do not invent facts.
    CONTEXT: numbered chunks with "document - Page N" labels.
    QUESTION: the user's question.

Providers (chosen from the environment at call time, see llm_settings):
    ollama  Local Ollama server. No API key, no cost, data stays
            on-machine. OLLAMA_HOST / OLLAMA_MODEL. Default.
    openai  Any OpenAI-compatible chat-completions API (Groq, Gemini,
            OpenRouter, ...). LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
            Needed for cloud hosting, where no local model can run.
            Selected by LLM_PROVIDER=openai, or automatically when
            LLM_API_KEY is set.
    Temperature is low (0.1) to favour faithful, extractive answers
    over creativity.

Conversation memory (Phase 11):
    Optional `history` = earlier [{"question", "answer"}] turns, replayed
    as chat messages BEFORE the current context + question so the model
    can resolve follow-ups ("and the year before?"). Only the last
    MAX_HISTORY_TURNS are sent, and earlier contexts are NOT replayed:
    facts must still come from the current CONTEXT block.

Errors (user-friendly, no secrets):
    Server down -> "Cannot reach Ollama at {host}. Is Ollama running?"
    Unknown model -> suggests `ollama pull {model}` / checking LLM_MODEL.
    Bad key / rate limit -> named plainly; the key is never echoed.

Connection:
    Phase 8 pipeline: retrieve() -> generate_answer() / stream_answer().
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

from dotenv import load_dotenv

load_dotenv()

Hit = Dict[str, object]
Turn = Dict[str, object]  # {"question": str, "answer": str}

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
# Hosted default: Groq's OpenAI-compatible endpoint (has a free tier).
DEFAULT_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_API_MODEL = "openai/gpt-oss-20b"
REFUSAL = "The information is not available in the provided documents."

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Answer using ONLY the provided context. "
    "If the answer cannot be found in the context, clearly say that the "
    "information is not available in the provided documents. "
    "Do not invent facts."
)

HISTORY_PROMPT = (
    " Earlier turns of the conversation are included only so you can "
    "understand follow-up questions. Facts in your answer must still come "
    "from the CONTEXT of the latest message."
)

MAX_CONTEXT_CHARS = 12_000
MAX_HISTORY_TURNS = 4
MAX_HISTORY_ANSWER_CHARS = 1_000
# Hosted models may spend part of the budget on hidden reasoning tokens.
MAX_TOKENS = {"ollama": 512, "openai": 1024}


def llm_settings() -> Dict[str, str]:
    """Resolve {provider, host, model, api_key} from the environment.

    Read at call time, not import time, so values injected after import
    (e.g. Streamlit secrets copied into os.environ) take effect.
    """
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "openai" if api_key else "ollama"
    if provider == "ollama":
        return {
            "provider": "ollama",
            "host": os.environ.get("OLLAMA_HOST", "").strip() or DEFAULT_OLLAMA_HOST,
            "model": os.environ.get("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL,
            "api_key": "",
        }
    return {
        "provider": "openai",
        "host": (os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_API_BASE).rstrip("/"),
        "model": os.environ.get("LLM_MODEL", "").strip() or DEFAULT_API_MODEL,
        "api_key": api_key,
    }


def build_context(hits: List[Hit], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format hits as numbered, source-labelled context within a char budget."""
    parts: List[str] = []
    used = 0
    for i, h in enumerate(hits, 1):
        label = f"[{i}] ({h.get('document', '?')} - Page {h.get('page', '?')}):"
        text = str(h.get("text", "") or "").strip()
        entry = f"{label}\n{text}"
        if used + len(entry) > max_chars:
            remaining = max_chars - used - len(label) - 5
            if remaining > 100:
                entry = f"{label}\n{text[:remaining]}...[truncated]"
            else:
                break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


def build_history_messages(
    history: List[Turn] | None, max_turns: int = MAX_HISTORY_TURNS
) -> List[Dict[str, str]]:
    """Replay the last `max_turns` complete turns as user/assistant messages."""
    messages: List[Dict[str, str]] = []
    for turn in (history or [])[-max_turns:]:
        asked = str(turn.get("question", "") or "").strip()
        answered = str(turn.get("answer", "") or "").strip()
        if not asked or not answered:
            continue
        messages.append({"role": "user", "content": asked})
        messages.append(
            {"role": "assistant", "content": answered[:MAX_HISTORY_ANSWER_CHARS]}
        )
    return messages


def build_messages(
    question: str, hits: List[Hit], history: List[Turn] | None = None
) -> List[Dict[str, str]]:
    """Build chat messages from question + hits (+ optional history)."""
    context = build_context(hits)
    past = build_history_messages(history)
    system = SYSTEM_PROMPT + HISTORY_PROMPT if past else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        *past,
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question.strip()}",
        },
    ]


class _ApiStatusError(Exception):
    """HTTP error status from an OpenAI-compatible API (body never kept)."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


def _stream_ollama(
    messages: List[Dict[str, str]],
    cfg: Dict[str, str],
    temperature: float,
    timeout: int,
    max_tokens: int,
) -> Iterator[str]:
    import ollama

    client = ollama.Client(host=cfg["host"], timeout=timeout)
    stream = client.chat(
        model=cfg["model"],
        messages=messages,
        options={"temperature": temperature, "num_predict": max_tokens},
        stream=True,
    )
    for part in stream:
        yield str(part["message"]["content"] or "")


def _stream_openai(
    messages: List[Dict[str, str]],
    cfg: Dict[str, str],
    temperature: float,
    timeout: int,
    max_tokens: int,
) -> Iterator[str]:
    import httpx

    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg["api_key"] else {}
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with httpx.stream(
        "POST",
        f"{cfg['host']}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    ) as response:
        if response.status_code >= 400:
            raise _ApiStatusError(response.status_code)
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                continue  # keep-alives / usage-only chunks carry no text
            if delta:
                yield str(delta)


def _friendly_error(exc: Exception, cfg: Dict[str, str]) -> RuntimeError:
    """Translate a provider failure into a user-facing message (no secrets)."""
    host, model = cfg["host"], cfg["model"]
    msg = str(exc).lower()
    if cfg["provider"] == "ollama":
        if (
            isinstance(exc, ConnectionError)
            or "connection" in msg
            or "refused" in msg
            or "failed to connect" in msg
        ):
            return RuntimeError(
                f"Cannot reach Ollama at {host}. Is Ollama running? "
                f"(Start it with `ollama serve`.)"
            )
        if "not found" in msg or "no such model" in msg or "404" in msg:
            return RuntimeError(
                f"Ollama model '{model}' not found. Pull it with "
                f"`ollama pull {model}`."
            )
        return RuntimeError(f"LLM request failed: {exc}")

    if isinstance(exc, _ApiStatusError):
        if exc.status in (401, 403):
            return RuntimeError(
                "The LLM API rejected the API key. Check LLM_API_KEY."
            )
        if exc.status == 404:
            return RuntimeError(
                f"Model '{model}' was not found at {host}. Check LLM_MODEL "
                f"and LLM_BASE_URL."
            )
        if exc.status == 429:
            return RuntimeError(
                "The LLM API rate limit was reached. Wait a moment and try again."
            )
        return RuntimeError(f"The LLM API returned an error (HTTP {exc.status}).")
    if "timed out" in msg or "timeout" in type(exc).__name__.lower():
        return RuntimeError(f"The LLM API at {host} timed out. Try again.")
    return RuntimeError(f"Cannot reach the LLM API at {host}.")


def _stream(
    messages: List[Dict[str, str]],
    cfg: Dict[str, str],
    temperature: float,
    timeout: int,
    max_tokens: int,
) -> Iterator[str]:
    backend = _stream_ollama if cfg["provider"] == "ollama" else _stream_openai
    produced = False
    try:
        for piece in backend(messages, cfg, temperature, timeout, max_tokens):
            if piece:
                produced = produced or bool(piece.strip())
                yield piece
    except Exception as exc:
        raise _friendly_error(exc, cfg) from exc
    if not produced:
        yield REFUSAL


def stream_answer(
    question: str,
    hits: List[Hit],
    model: str | None = None,
    host: str | None = None,
    temperature: float = 0.1,
    timeout: int = 120,
    max_tokens: int | None = None,
    history: List[Turn] | None = None,
) -> Iterator[str]:
    """Yield a grounded answer piece by piece; refuse on empty hits.

    The question is validated immediately; provider errors surface as
    RuntimeError while iterating.
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    if not hits:
        return iter([REFUSAL])
    cfg = llm_settings()
    cfg["model"] = model or cfg["model"]
    cfg["host"] = (host or cfg["host"]).rstrip("/")
    return _stream(
        build_messages(question, hits, history),
        cfg,
        temperature,
        timeout,
        max_tokens or MAX_TOKENS[cfg["provider"]],
    )


def generate_answer(
    question: str,
    hits: List[Hit],
    model: str | None = None,
    host: str | None = None,
    temperature: float = 0.1,
    timeout: int = 120,
    max_tokens: int | None = None,
    history: List[Turn] | None = None,
) -> str:
    """Generate a grounded answer; refuse deterministically on empty hits."""
    pieces = stream_answer(
        question,
        hits,
        model=model,
        host=host,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        history=history,
    )
    return "".join(pieces).strip() or REFUSAL
