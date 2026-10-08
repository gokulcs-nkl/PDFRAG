"""Document chunking (Phase 2).

Problem solved:
    Embeddings + FAISS work best on short, focused passages. A whole PDF
    page (or whole document) is too coarse: it dilutes similarity scores
    and blows past LLM context limits. Chunking splits pages into
    retrievable units.

Input:
    Phase 1 page list: [{"text": ..., "page": 1}, ...] plus a document name.

Output:
    Chunk list: [{"chunk_id": 0, "text": ..., "page": 3,
                  "document": "example.pdf"}, ...]
    chunk_id is globally sequential within one document.

Strategy (simple, better than blind char splits):
    1. Collapse PDF line-break noise to single spaces.
    2. Split into sentences on [.!?] boundaries (regex).
    3. Accumulate whole sentences up to chunk_size.
    4. Only oversize single sentences fall back to hard word-window splits.
    5. Overlap = tail of previous FINAL chunk prepended to the next base
       chunk (word-boundary adjusted), so consecutive chunks share context
       and information at boundaries is not lost.

Page policy:
    Chunking is done PER PAGE so every chunk has exactly one page number
    and citations stay unambiguous. A chunk never spans two pages.

Size semantics:
    Base content per chunk is <= chunk_size. With overlap>0 the final
    stored text may be up to chunk_size + chunk_overlap (overlap prefix
    is extra context). Tests assert against chunk_size + chunk_overlap.

Connection:
    Phase 3 embeds chunk["text"]; Phase 4 stores one vector per chunk_id
    with this metadata dict attached.
"""

from __future__ import annotations

import re
from typing import Dict, List

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

Chunk = Dict[str, object]  # {"chunk_id": int, "text": str, "page": int, "document": str}
PageText = Dict[str, object]  # {"text": str, "page": int} (from pdf_processor)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c\(\[])")


def _validate(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 20:
        raise ValueError("chunk_size must be >= 20 characters.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def normalize_text(text: str) -> str:
    """Collapse PDF line-break/whitespace noise to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> List[str]:
    """Split normalized text into sentences; fallback to whole text."""
    text = normalize_text(text)
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Sliding word-window split for a single oversize sentence."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - chunk_overlap) if chunk_overlap else chunk_size
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        add = len(word) + (1 if current else 0)
        if current and current_len + add > chunk_size:
            chunks.append(" ".join(current))
            # Overlap: keep trailing words totalling ~chunk_overlap chars.
            if chunk_overlap:
                tail: List[str] = []
                tail_len = 0
                for w in reversed(current):
                    tail_len += len(w) + (1 if tail else 0)
                    tail.append(w)
                    if tail_len >= chunk_overlap:
                        break
                current = list(reversed(tail))
                current_len = sum(len(w) for w in current) + max(0, len(current) - 1)
            else:
                current, current_len = [], 0
            add = len(word) + (1 if current else 0)
        current.append(word)
        current_len += add
        # Single word longer than chunk_size: hard cut it.
        if len(word) > chunk_size:
            joined = " ".join(current)
            chunks.extend(
                joined[i : i + chunk_size]
                for i in range(0, len(joined), step)
            )
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_prefix(previous: str, chunk_overlap: int) -> str:
    """Tail of previous chunk, backed off to a word boundary."""
    if chunk_overlap <= 0 or not previous:
        return ""
    tail = previous[-chunk_overlap:]
    space = tail.find(" ")
    if space != -1 and len(tail) - space > 10:
        # Drop partial leading word so next chunk starts cleanly.
        tail = tail[space + 1 :]
    return tail.strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split one page's text into sentence-aware overlapping chunk strings."""
    _validate(chunk_size, chunk_overlap)
    sentences = split_sentences(text)
    if not sentences:
        return []

    base: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        if len(sent) > chunk_size:
            if current:
                base.append(" ".join(current))
                current, current_len = [], 0
            base.extend(_hard_split(sent, chunk_size, chunk_overlap))
            continue
        add = len(sent) + (1 if current else 0)
        if current and current_len + add > chunk_size:
            base.append(" ".join(current))
            current, current_len = [], 0
            add = len(sent)
        current.append(sent)
        current_len += add
    if current:
        base.append(" ".join(current))

    # Second pass: prepend overlap prefix from previous FINAL chunk.
    final: List[str] = []
    for i, piece in enumerate(base):
        if i == 0 or chunk_overlap == 0:
            final.append(piece)
        else:
            prefix = _overlap_prefix(final[i - 1], chunk_overlap)
            merged = f"{prefix} {piece}" if prefix else piece
            final.append(merged.strip())
    return final


def chunk_pages(
    pages: List[PageText],
    document: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk Phase 1 pages, tagging each chunk with document + page.

    Args:
        pages: [{"text": str, "page": int}] from extract_text_from_pdf.
        document: Source filename, e.g. "example.pdf".
        chunk_size: Max base characters per chunk (see module docstring).
        chunk_overlap: Overlap characters carried from previous chunk.

    Returns:
        [{"chunk_id": int, "text": str, "page": int, "document": str}]
        with globally sequential chunk_id. Empty input -> [].
    """
    _validate(chunk_size, chunk_overlap)
    if not document or not str(document).strip():
        raise ValueError("document name must be a non-empty string.")
    chunks: List[Chunk] = []
    chunk_id = 0
    for page in pages:
        text = str(page.get("text", "") or "")
        page_num = int(page.get("page", 0))
        for piece in split_text_into_chunks(text, chunk_size, chunk_overlap):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "page": page_num,
                    "document": document,
                }
            )
            chunk_id += 1
    return chunks
