# PDF RAG Assistant

End-to-end PDF RAG: Upload PDFs -> Extract -> Chunk -> FastEmbed -> FAISS -> Retrieve -> Ollama LLM -> Answer + Sources (Streamlit UI).

## Status: Phase 8 complete (pipeline wired end-to-end, live answer verified)

- `src/pdf_processor.py`: `extract_text_from_pdf()` returns `[{"text": ..., "page": 1}]`, 1-indexed, skips empty pages.
- `src/chunker.py`: `chunk_pages()` returns `[{chunk_id, text, page, document}]`, sentence-aware, per-page, configurable size/overlap (defaults 1000/200).
- `src/embeddings.py`: `EmbeddingModel` singleton (`BAAI/bge-small-en-v1.5`, 384-dim) with `embed_documents()` / `embed_query()`; same model for docs + queries. First run downloads ~130MB (takes ~1-2 min), then cached.
- `src/vector_store.py`: `FaissVectorStore` (exact `IndexFlatIP` + L2-normalize = cosine); `add()` keeps 1:1 metadata alignment, `search()` returns top-K `[{..., score}]`, `clear()` for UI reset.
- `src/retriever.py`: `Retriever(store, default_top_k=5)` with `retrieve(question, top_k)` — LLM-independent. Gate: `python scripts/retrieval_check.py` (revenue→p.1 @0.71, python→p.2 @0.75).
- `src/llm.py`: `generate_answer(question, hits)` via Ollama (`OLLAMA_HOST`/`OLLAMA_MODEL`, default `gemma3:4b`); grounded system prompt + refuse-if-absent; empty hits short-circuit; low temp 0.1, `num_predict` cap. Live check: revenue Q → "ten million" in ~7-12s (warm).
- `src/rag_pipeline.py`: `RagPipeline` facade — `ingest_documents([(name, pdf), ...])` with per-file report (duplicates/blank/invalid collected, not raised) and `query(question)` → `{answer, sources}`. Multi-PDF native; in-memory chunks+embeddings with index rebuild on ingest.
- Choices: `pypdf` (pure-Python, simple install), Ollama local (`OLLAMA_HOST`/`OLLAMA_MODEL` in `.env`), FastEmbed `BAAI/bge-small-en-v1.5` (384-dim, Phase 3).

## Architecture (target)

PDF -> Extraction -> Chunking -> FastEmbed -> FAISS -> Retrieval -> LLM -> Answer

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
pytest -q
```

## Phase 1 interview notes

- Problem: PDFs are binary/layout-oriented; downstream needs clean page-aligned text.
- Input: path / bytes / file-like. Output: non-empty pages with original page numbers.
- Why pypdf: zero native deps; swappable behind `extract_text_from_pdf`.
- Next: chunker splits these pages into overlapping chunks with document + chunk_id.
