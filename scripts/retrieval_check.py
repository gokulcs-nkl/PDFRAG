"""Phase 6 gate: end-to-end retrieval check WITHOUT any LLM.

Builds a 3-page PDF in memory (revenue / python / sourdough), runs the
full path Extract -> Chunk -> Embed -> FAISS -> Retrieve, and prints the
top-K chunks for a demo question. Eyeball the output: the revenue page
must rank first for a revenue question. Do NOT proceed to LLM work if
it does not.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reportlab.pdfgen import canvas

from src.chunker import chunk_pages
from src.embeddings import get_embedding_model
from src.pdf_processor import extract_text_from_pdf
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


def make_demo_pdf() -> bytes:
    pages = [
        "Annual Report 2024. The company's revenue in 2024 was ten million dollars, "
        "up from eight million in 2023. Growth was driven by cloud subscriptions.",
        "Python is a high-level programming language. It is known for readable syntax, "
        "dynamic typing, and a vast standard library used in data science.",
        "Sourdough bread requires flour, water, salt, and a live starter culture "
        "fermented slowly over twelve hours before baking.",
    ]
    buf = BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        c.drawString(72, 750, text[:95])
        c.drawString(72, 735, text[95:190])
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def main(question: str = "What is the company's revenue?", top_k: int = 3, gate: bool = False) -> None:
    pdf_bytes = make_demo_pdf()
    pages = extract_text_from_pdf(pdf_bytes)
    print(f"Extracted {len(pages)} pages.")
    chunks = chunk_pages(pages, document="demo.pdf", chunk_size=500, chunk_overlap=50)
    print(f"Built {len(chunks)} chunks.")
    model = get_embedding_model()
    store = FaissVectorStore(dim=model.dim)
    store.add(
        model.embed_documents([str(c["text"]) for c in chunks]),
        chunks,
    )
    hits = Retriever(store).retrieve(question, top_k=top_k)
    print(f'\nQuestion: "{question}"\n')
    for i, h in enumerate(hits, 1):
        print(f"--- Retrieved Chunk {i} (score={h['score']:.3f}) ---")
        print(f"Source: {h['document']} - Page {h['page']}")
        print(str(h["text"])[:300])
        print()
    # Gate check only applies to the default revenue question; custom
    # questions are display-only (relevance judged by eye).
    if gate:
        ok = "revenue" in str(hits[0]["text"]).lower()
        print("GATE:", "PASS - top chunk is relevant." if ok else "FAIL - check chunking/embedding.")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) == 1:
        main(gate=True)
    else:
        main(_sys.argv[1])
