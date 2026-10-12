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
  → LLM (Ollama or hosted API)           faithful answer or explicit refusal, streamed
  → Streamlit chat UI (app.py)           streamed answer + sources + conversation history
```

`src/rag_pipeline.py` (`RagPipeline`) wires all of the above behind two calls —
`ingest_documents()` and `query()` / `query_stream()` — so `app.py` contains
zero RAG logic. Conversation memory lives in the UI session and is handed to the
query call (`history=...`); the index and retriever never store it.
`src/retriever.py` (`Retriever`) owns the LLM-independent
question → chunks path, which is how retrieval was verified *before* any LLM
was integrated (`scripts/retrieval_check.py`).

## Technologies

| Layer       | Choice | Why |
|-------------|--------|-----|
| Language    | Python 3.10 | Team runtime |
| UI          | Streamlit 1.54 (chat elements, `write_stream`) | Chat frontend and backend in one deployable Python app |
| PDF text    | pypdf 4.3.1 | Pure-Python, zero native deps; swappable behind one function |
| Embeddings  | FastEmbed 0.8.1, `BAAI/bge-small-en-v1.5` (384-dim) | Local ONNX, no API key, fast on CPU; **same** model encodes docs and queries |
| Vector search | FAISS 1.15.0 (`IndexFlatIP` + L2-normalize) | Exact cosine search, no training, ideal at this scale |
| LLM         | Ollama (local, default `gemma3:4b`) **or** any OpenAI-compatible API | Local: no key, no cost, data stays on-machine. Hosted API: required in the cloud, where no local model can run. Temp 0.1 + token cap for faithful answers |
| Config      | python-dotenv / Streamlit secrets | Same keys from `.env` locally or app secrets when hosted |
| Tests       | pytest (83 tests) + reportlab fixtures | Every phase verified; UI tested headlessly via AppTest |

No LangChain — the pipeline is explicit so each component can be understood
and explained independently.

## Installation

```powershell
# 1. Python env
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # runtime + pytest/reportlab

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
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model for answer generation |
| `LLM_API_KEY` | *(unset)* | Key for a hosted OpenAI-compatible API. **Setting it switches the app from Ollama to that API** |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | API base URL (Groq, Gemini, OpenRouter, ...) |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Model id at that API |
| `LLM_PROVIDER` | *(auto)* | Force `ollama` or `openai` regardless of the key |

Only `.env.example` is committed. Errors never print secrets — an unreachable
server yields "Is Ollama running?", an unknown model suggests the `ollama pull`
command, and a rejected key or a rate limit is named without echoing the key.

## Usage

```powershell
streamlit run app.py
```

1. **Upload** one or more PDFs in the sidebar (25 MB each). Duplicates,
   blank/scanned PDFs, and corrupt files are reported per-file without aborting
   the batch. Each file is indexed once, not on every interaction.
2. **Manage documents:** every indexed document is listed with its page and
   chunk count and a **✕** to remove just that one. With two or more documents,
   **Search in** limits answers to the ones you pick.
3. **Chat:** type in the box at the bottom (or click a starter question). The
   answer streams in as it is written.
4. **Verify:** open **Sources** under any answer — one tab per retrieved chunk
   with `document — Page N`, a relevance bar, and the exact passage.
5. **Follow up** ("and the year before?"). With **Use conversation memory** on
   (Settings), earlier turns are used to interpret the question.
   **Regenerate last answer** asks the last question again.
6. **New conversation** forgets the turns but keeps the index. **Download
   transcript** saves the chat with its sources as Markdown. **Clear index**
   resets everything, including the uploader.

Settings also holds Top-K (1–10) and re-indexing of same-named files.

Pre-LLM retrieval check (the Phase 6 gate, still useful for debugging):

```powershell
python scripts/retrieval_check.py
```

## Deployment (free cloud hosting)

The app deploys as-is to **Streamlit Community Cloud**. Two things differ from
a local run: the LLM must be a hosted API (no free host can run `gemma3:4b`),
and configuration comes from app secrets instead of `.env`.

1. Get an API key from an OpenAI-compatible provider with a free tier, e.g.
   Groq (<https://console.groq.com>).
2. Push this repository to GitHub.
3. At <https://share.streamlit.io> choose **Create app**, pick the repo and
   branch, and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, paste the keys from
   `.streamlit/secrets.toml.example` with your own values.
5. Deploy. The first boot downloads the embedding model (~130 MB).

What to expect when hosted:

- Each visitor gets their own in-memory index; nothing is persisted, and the
  app sleeps after 12 hours without traffic (anyone can wake it).
- The retrieved passages and the question are sent to the LLM provider.
  Do not upload confidential PDFs to a public deployment.
- Free API tiers are rate-limited; the app reports a rate limit plainly.

Free hosts compared (checked October 2026 — limits change, verify before
relying on them):

| Host | Free for this app? | Limits | Verdict |
|------|--------------------|--------|---------|
| Streamlit Community Cloud | Yes, no card | ~0.7–2.7 GB RAM, sleeps after 12 h idle, public apps unlimited | **Recommended** — built for Streamlit, deploys from GitHub |
| Hugging Face Spaces | No longer | 2 vCPU / 16 GB hardware is free, but creating a Docker/Gradio Space now needs a paid PRO plan | Best hardware, but not free to create |
| Render | Yes, no card | 512 MB RAM, 0.1 CPU, sleeps after 15 min idle (~1 min wake) | Too little RAM for the embedding model + FAISS |
| Google Cloud Run | Free allowance, card required | 2M requests/month, scales to zero | Works via a container; more setup, card on file |
| Koyeb | Card required | 512 MB RAM, 0.1 vCPU, sleeps after 1 h | Same RAM problem as Render |
| Railway / Fly.io | No | Trial credit only | Not free |

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
- **Conversation memory** is kept out of the index. The last 4 question/answer
  turns are replayed to the LLM as chat messages so it can resolve references,
  but their old contexts are not: facts must come from the chunks retrieved for
  the *current* question. Because a follow-up like "and the year before?" has
  no searchable topic of its own, the previous question is prepended to the
  retrieval query (not to the question the LLM sees). Trade-off: right after a
  change of topic, the first retrieval is partly pulled toward the old topic —
  turn memory off or clear the conversation if that hurts.

## Project Structure

```text
pdf-rag-assistant/
├── app.py                  # Thin Streamlit chat UI (Phase 9) + conversation state (Phase 11)
├── requirements.txt        # Runtime only (what a cloud host installs)
├── requirements-dev.txt    # Runtime + pytest / reportlab
├── .env.example            # Local config template (commit this, never .env)
├── .streamlit/
│   ├── config.toml             # Upload cap, theme
│   └── secrets.toml.example    # Keys to paste into hosted app secrets
├── data/.gitkeep
├── scripts/
│   └── retrieval_check.py  # Phase 6 gate: PDF → chunks → answer-free top-K printout
├── src/
│   ├── pdf_processor.py    # Phase 1: PDF → [{text, page}]
│   ├── chunker.py          # Phase 2: pages → [{chunk_id, text, page, document}]
│   ├── embeddings.py       # Phase 3: singleton FastEmbed wrapper
│   ├── vector_store.py     # Phase 4: FAISS exact-cosine index + metadata
│   ├── retriever.py        # Phase 5: question → top-K hits (no LLM)
│   ├── llm.py              # Phase 7: grounded prompt (+ history turns), streaming, Ollama / hosted API
│   └── rag_pipeline.py     # Phase 8: ingest / query / query_stream / remove_document facade
└── tests/                  # Phase 14: 83 tests, one module per phase + UI
    ├── test_pdf_processor.py (7)   ├── test_chunker.py (9)
    ├── test_embeddings.py (6)      ├── test_vector_store.py (9)
    ├── test_retriever.py (6)       ├── test_llm.py (21, incl. live)
    ├── test_rag_pipeline.py (12)   └── test_app.py (13, headless AppTest)
```

## Interview Guide (one line per component)

| Component | Problem → Input → Output → Why this tech |
|-----------|-------------------------------------------|
| PDF extraction | Binary layout → file/bytes → `[{text, page}]` → pypdf: zero-dep, swappable |
| Chunking | Pages too coarse → pages → sentence-aware overlapping chunks → per-page keeps citations exact |
| Embeddings | Text isn't searchable → strings → 384-dim vectors → FastEmbed: local, one shared model |
| FAISS | Slow brute force → vectors → index positions → FlatIP+normalize = exact cosine, no training |
| Retrieval | Need evidence, not excerpts → question → top-K scored chunks → LLM-free, independently testable |
| LLM | Chunks aren't answers → question+context → grounded answer/refusal → local Ollama or hosted API behind one function, low temp |
| Pipeline | Wiring chaos → all components → two-call facade → UI stays thin, flow unit-testable |
| Memory | Follow-ups lack context → question+past turns → resolved answer → caller-owned turns, index untouched |

## Future Improvements

- Hybrid search (dense + BM25 keyword) with reciprocal-rank fusion
- Cross-encoder reranking of top-K before generation
- Smarter chunking (layout-aware, sliding-window with section headers)
- Persistent vector storage (FAISS on-disk / vector DB) with incremental updates
- LLM-based rewriting of follow-ups into standalone questions (replaces the
  previous-question heuristic used for retrieval today)
- Minimum-similarity cutoff so off-topic questions retrieve nothing and refuse
  without an LLM call
- Finer citations (sentence-level spans, answer-to-source highlighting)
- Evaluation harness (golden Q&A set, recall@K, faithfulness scoring)
