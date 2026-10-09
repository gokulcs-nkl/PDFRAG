# PDF RAG Assistant

Ask natural-language questions about your PDFs and get grounded answers with
page-level sources. Upload documents → the app extracts text, chunks it,
embeds it with FastEmbed, indexes it in FAISS, retrieves the most relevant
chunks for your question, and has a local LLM answer **only from that context**.

## Project Overview

**Problem:** Large language models hallucinate when asked about private
documents they were never trained on, and pasting whole PDFs into a prompt
doesn't scale (context limits, cost, noise).

**Solution:** Retrieval-Augmented Generation. Instead of hoping the model
"knows" your documents, the app *retrieves* the handful of relevant passages
first and *then* asks the model to answer strictly from them — refusing when
the answer isn't present. Every answer ships with its sources
(`document — Page N`, similarity score, text snippet) so nothing is a black box.

## Architecture

```text
PDF(s)
  → Extraction  (src/pdf_processor.py)   page-aligned text [{text, page}]
  → Chunking    (src/chunker.py)         overlapping chunks [{chunk_id, text, page, document}]
  → FastEmbed   (src/embeddings.py)      dense vectors (384-dim, one model for docs + queries)
  → FAISS       (src/vector_store.py)    exact cosine index + metadata mapping
  → User question → question vector → similarity search → Top-K chunks
  → Prompt      (src/llm.py)             grounded system prompt + labelled context
  → LLM (Ollama)                         faithful answer or explicit refusal
  → Streamlit UI (app.py)                answer + sources
```

`src/rag_pipeline.py` (`RagPipeline`) wires all of the above behind two calls —
`ingest_documents()` and `query()` — so `app.py` contains zero RAG logic.
`src/retriever.py` (`Retriever`) owns the LLM-independent
question → chunks path, which is how retrieval was verified *before* any LLM
was integrated (`scripts/retrieval_check.py`).

## Technologies

| Layer       | Choice | Why |
|-------------|--------|-----|
| Language    | Python 3.10 | Team runtime |
| UI          | Streamlit 1.54 | Fastest path to a usable multi-PDF frontend |
| PDF text    | pypdf 4.3.1 | Pure-Python, zero native deps; swappable behind one function |
| Embeddings  | FastEmbed 0.8.1, `BAAI/bge-small-en-v1.5` (384-dim) | Local ONNX, no API key, fast on CPU; **same** model encodes docs and queries |
| Vector search | FAISS 1.15.0 (`IndexFlatIP` + L2-normalize) | Exact cosine search, no training, ideal at this scale |
| LLM         | Ollama (local), default `gemma3:4b` | No key, no cost, data stays on-machine; temp 0.1 + token cap for faithful answers |
| Config      | python-dotenv | `OLLAMA_HOST` / `OLLAMA_MODEL` from `.env` |
| Tests       | pytest (55 tests) + reportlab fixtures | Every phase verified; UI tested headlessly via AppTest |

No LangChain — the pipeline is explicit so each component can be understood
and explained independently.

## Installation

```powershell
# 1. Python env
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Ollama (https://ollama.com), then in a separate terminal:
ollama serve
ollama pull gemma3:4b

# 3. Local config (gitignored; never commit secrets)
copy .env.example .env

# 4. Verify
python -m pytest tests/ -q
```

First run downloads the FastEmbed model (~130MB, ~1–2 min) and cold-loads the
LLM into the GPU/RAM (first answer can take minutes; warm answers ~7–12s on an
RTX 3050-class GPU).

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `gemma3:4b` | Model for answer generation |

Only `.env.example` is committed. Errors never print secrets — an unreachable
server yields "Is Ollama running?", an unknown model suggests the `ollama pull`
command.

## Usage

```powershell
streamlit run app.py
```

1. **Sidebar:** upload one or more PDFs, set Top-K (1–10), optionally enable
   re-indexing of same-named files. Duplicates, blank/scanned PDFs, and corrupt
   files are reported per-file without aborting the batch.
2. **Main:** type a question, press **Ask**.
3. Read the **Answer**, then expand **Sources** — each shows
   `document — Page N`, cosine score, and the exact retrieved snippet.
4. **Clear index** resets everything.

Pre-LLM retrieval check (the Phase 6 gate, still useful for debugging):

```powershell
python scripts/retrieval_check.py
```

## RAG Explanation

- **Embeddings** turn text into fixed-size numeric vectors (here 384 numbers)
  where similar *meanings* sit close together. Both chunks and the question go
  through the identical model — otherwise "closeness" would be meaningless.
- **Chunking is required** because whole pages/documents are too coarse: they
  dilute similarity scores and overflow LLM context. This project chunks
  **per page** (sentence-aware accumulation, ~1000 chars + 200 overlap) so every
  chunk carries exactly one page number for citations; only oversize single
  sentences fall back to hard splits.
- **FAISS** is a similarity-search index, not a database: it stores bare float
  vectors for millisecond top-K lookup. Metadata (`chunk_id/text/page/document`)
  lives in a parallel list; search positions map back to chunks plus a cosine
  **score** in [-1, 1].
- **Similarity search** = embed the question, L2-normalize, take the K highest
  inner products (= cosine similarities) from the index.
- **Retrieval happens before the LLM** so the model reasons over a small,
  relevant, citable evidence set instead of the whole corpus — cheaper, more
  accurate, and auditable. Retrieval is deliberately LLM-independent and was
  proven relevant (revenue Q → p.1 @0.71; python Q → p.2 @0.75) before any LLM
  call existed.
- **The LLM receives** a system instruction (answer *only* from context, refuse
  if absent, never invent) plus numbered source-labelled chunks plus the
  question, at low temperature with a token cap. Empty retrieval short-circuits
  to a refusal without spending an LLM call.

## Project Structure

```text
pdf-rag-assistant/
├── app.py                  # Thin Streamlit UI (Phase 9)
├── requirements.txt
├── .env.example            # OLLAMA_HOST / OLLAMA_MODEL (commit this, never .env)
├── data/.gitkeep
├── scripts/
│   └── retrieval_check.py  # Phase 6 gate: PDF → chunks → answer-free top-K printout
├── src/
│   ├── pdf_processor.py    # Phase 1: PDF → [{text, page}]
│   ├── chunker.py          # Phase 2: pages → [{chunk_id, text, page, document}]
│   ├── embeddings.py       # Phase 3: singleton FastEmbed wrapper
│   ├── vector_store.py     # Phase 4: FAISS exact-cosine index + metadata
│   ├── retriever.py        # Phase 5: question → top-K hits (no LLM)
│   ├── llm.py              # Phase 7: grounded prompt + Ollama generation
│   └── rag_pipeline.py     # Phase 8: ingest_documents() / query() facade
└── tests/                  # Phase 14: 55 tests, one module per phase + UI
    ├── test_pdf_processor.py (7)   ├── test_chunker.py (9)
    ├── test_embeddings.py (6)      ├── test_vector_store.py (9)
    ├── test_retriever.py (6)       ├── test_llm.py (7, incl. live)
    ├── test_rag_pipeline.py (7)    └── test_app.py (4, headless AppTest)
```

## Interview Guide (one line per component)

| Component | Problem → Input → Output → Why this tech |
|-----------|-------------------------------------------|
| PDF extraction | Binary layout → file/bytes → `[{text, page}]` → pypdf: zero-dep, swappable |
| Chunking | Pages too coarse → pages → sentence-aware overlapping chunks → per-page keeps citations exact |
| Embeddings | Text isn't searchable → strings → 384-dim vectors → FastEmbed: local, one shared model |
| FAISS | Slow brute force → vectors → index positions → FlatIP+normalize = exact cosine, no training |
| Retrieval | Need evidence, not excerpts → question → top-K scored chunks → LLM-free, independently testable |
| LLM | Chunks aren't answers → question+context → grounded answer/refusal → local Ollama, low temp |
| Pipeline | Wiring chaos → all components → two-call facade → UI stays thin, flow unit-testable |

## Future Improvements

- Hybrid search (dense + BM25 keyword) with reciprocal-rank fusion
- Cross-encoder reranking of top-K before generation
- Smarter chunking (layout-aware, sliding-window with section headers)
- Persistent vector storage (FAISS on-disk / vector DB) with incremental updates
- Conversation memory kept separate from retrieval (Phase 11)
- Finer citations (sentence-level spans, answer-to-source highlighting)
- Evaluation harness (golden Q&A set, recall@K, faithfulness scoring)
