"""PDF RAG Assistant - Streamlit UI (Phase 9).

Thin frontend by design: ALL logic lives in RagPipeline (Phase 8).
This file only handles widgets, session state, and user-friendly
error display (Phases 12-13 groundwork: friendly errors, no secrets,
sources with document/page/score/snippet).

Layout (per spec):
    Sidebar: multi-PDF upload, top-K selector, re-index toggle,
             reset/clear button.
    Main: title, indexed-document info, question input, Ask button,
          answer, retrieved sources.
"""

from __future__ import annotations

import streamlit as st

from src.rag_pipeline import RagPipeline

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")


def get_pipeline() -> RagPipeline:
    """Session-persisted pipeline (one embedding-model load per session).

    Stored in session_state rather than cache_resource so tests can
    inject a pre-seeded pipeline via AppTest session_state, and so each
    user session owns a mutable index that Clear can reset.
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RagPipeline()
    return st.session_state.pipeline


def pending_uploads(pipeline: RagPipeline, uploads, replace: bool) -> list:
    """Filter sidebar uploads down to files needing indexing.

    Pure helper (no Streamlit calls) so the duplicate/replace policy is
    unit-testable without driving the file_uploader widget -- which
    AppTest in streamlit 1.54 does not support (renders UnknownElement).
    """
    if not uploads:
        return []
    indexed = set(pipeline.documents)
    return [f for f in uploads if replace or f.name not in indexed]


pipeline = get_pipeline()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    top_k = st.slider("Retrieved chunks (Top-K)", 1, 10, 5)
    replace = st.checkbox(
        "Re-index files with the same name", value=False,
        help="If off, already-indexed filenames are skipped as duplicates.",
    )
    if st.button("Clear index"):
        pipeline.clear()
        st.success("Index cleared.")
        st.rerun()

# ---------- Ingest new uploads ----------
if uploads:
    new_files = pending_uploads(pipeline, uploads, replace)
    if new_files:
        with st.spinner(f"Indexing {len(new_files)} PDF(s)..."):
            try:
                report = pipeline.ingest_documents(
                    [(f.name, f.getvalue()) for f in new_files],
                    replace=replace,
                )
            except Exception as exc:  # defensive: pipeline reports per-file issues itself
                st.error(f"Indexing failed: {exc}")
            else:
                for name in report["documents_added"]:
                    st.sidebar.success(f"Indexed: {name}")
                for name, reason in report["skipped"].items():
                    st.sidebar.warning(f"Skipped {name}: {reason}")

# ---------- Document info ----------
if pipeline.documents:
    st.info(
        f"Indexed {len(pipeline.documents)} document(s), "
        f"{pipeline.num_chunks} chunk(s): "
        + ", ".join(pipeline.documents)
    )
else:
    st.info("Upload one or more PDFs in the sidebar to get started.")

# ---------- Q&A ----------
question = st.text_input("Ask a question about your documents:")
ask = st.button("Ask", disabled=not pipeline.documents)

if ask:
    if not question or not question.strip():
        st.error("Please enter a question.")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = pipeline.query(question, top_k=top_k)
        except ValueError as exc:  # no docs, empty question, empty index
            st.error(str(exc))
        except RuntimeError as exc:  # Ollama down / model missing (already friendly)
            st.error(str(exc))
        except Exception:  # never leak internals or keys
            st.error("Something went wrong while answering. Please try again.")
        else:
            st.subheader("Answer")
            st.write(str(result["answer"]))
            st.subheader("Sources")
            sources = result["sources"]
            if not sources:
                st.write("No relevant chunks found.")
            for i, hit in enumerate(sources, 1):
                doc = hit.get("document", "?")
                page = hit.get("page", "?")
                score = float(hit.get("score", 0.0) or 0.0)
                st.markdown(f"**[{i}] {doc} — Page {page}** (score {score:.3f})")
                with st.expander(f"Chunk {i} snippet"):
                    st.write(str(hit.get("text", ""))[:1500])
