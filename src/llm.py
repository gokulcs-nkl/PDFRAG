"""LLM answer generation via Ollama (Phase 7).

Problem solved:
    Retrieval finds relevant chunks, but users want a direct ANSWER, not
    a pile of excerpts. This module turns (question + chunks) into a
    grounded natural-language answer.

Input:
    question string + retriever hits [{text, page, document, score}].

Output:
    Answer string. Empty hits -> deterministic refusal sentence
    (no LLM call wasted).

Prompt (per spec):
    SYSTEM: document QA assistant; answer ONLY from context; if absent,
    say the information is not available; do not invent facts.
    CONTEXT: numbered chunks with "document - Page N" labels.
    QUESTION: the user's question.

Why Ollama (local):
    No API key, no cost, data stays on-machine. Provider stays
    configurable via OLLAMA_HOST / OLLAMA_MODEL env vars (.env), so a
    hosted API could replace the generate_answer() body later without
    touching retrieval or UI code. Temperature is low (0.1) to favour
    faithful, extractive answers over creativity.

Errors (user-friendly, no secrets):
    Server down -> "Cannot reach Ollama at {host}. Is Ollama running?"
    Unknown model -> suggests `ollama pull {model}`.
    Timeouts propagate as a clear message.

Connection:
    Phase 8 pipeline: retrieve() -> generate_answer() -> (answer, hits).
"""

from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

Hit = Dict[str, object]

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
REFUSAL = "The information is not available in the provided documents."

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. "
    "Answer using ONLY the provided context. "
    "If the answer cannot be found in the context, clearly say that the "
    "information is not available in the provided documents. "
    "Do not invent facts."
)

MAX_CONTEXT_CHARS = 12_000


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


def build_messages(question: str, hits: List[Hit]) -> List[Dict[str, str]]:
    """Build Ollama chat messages from question + hits."""
    context = build_context(hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question.strip()}",
        },
    ]


def generate_answer(
    question: str,
    hits: List[Hit],
    model: str | None = None,
    host: str | None = None,
    temperature: float = 0.1,
    timeout: int = 120,
    max_tokens: int = 512,
) -> str:
    """Generate a grounded answer; refuse deterministically on empty hits."""
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    if not hits:
        return REFUSAL
    model = model or DEFAULT_MODEL
    host = host or DEFAULT_HOST

    import ollama

    messages = build_messages(question, hits)
    try:
        client = ollama.Client(host=host, timeout=timeout)
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
    except ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}. Is Ollama running? "
            f"(Start it with `ollama serve`.)"
        ) from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "failed to connect" in msg:
            raise RuntimeError(
                f"Cannot reach Ollama at {host}. Is Ollama running?"
            ) from exc
        if "not found" in msg or "no such model" in msg or "404" in msg:
            raise RuntimeError(
                f"Ollama model '{model}' not found. Pull it with "
                f"`ollama pull {model}` (available here: gemma3:4b, gemma3, gemma4)."
            ) from exc
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        content = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("LLM returned an unexpected response shape.") from exc
    return str(content).strip() or REFUSAL
