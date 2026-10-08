# PDF RAG Assistant

End-to-end PDF RAG: Upload PDFs -> Extract -> Chunk -> FastEmbed -> FAISS -> Retrieve -> Ollama LLM -> Answer + Sources (Streamlit UI).

## Status: Phase 1 complete (PDF text extraction)

- `src/pdf_processor.py`: `extract_text_from_pdf()` returns `[{"text": ..., "page": 1}]`, 1-indexed, skips empty pages.
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
