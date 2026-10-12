"""PDF RAG Assistant - Streamlit chat UI (Phase 9).

Thin frontend by design: ALL logic lives in RagPipeline (Phase 8).
This file only handles widgets, session state, and user-friendly
error display (friendly errors, no secrets, sources with
document/page/score/snippet).

Layout:
    Sidebar: multi-PDF upload, indexed-document list (page/chunk counts,
             per-document remove), "Search in" document filter, settings
             (top-K, re-index, conversation memory), new conversation,
             transcript download, clear index.
    Main: chat. Each answer streams in as it is written and carries its
          own expandable sources; starter questions when the chat is
          empty; regenerate for the last answer.

Session state:
    pipeline          RagPipeline (index + embedding model).
    history           Conversation turns [{question, answer, sources}]
                      (Phase 11). Owned here, passed to the pipeline.
    upload_status     {upload key: (kind, message)} so each uploaded file
                      is ingested once, not on every rerun.
    uploader_gen      Bumped by "Clear index" to empty the file uploader.
    pending_question  Question queued by a starter/regenerate button.

Configuration comes from the environment (.env locally). On Streamlit
Community Cloud the same keys are set as app secrets and copied into
the environment by apply_secrets().
"""

from __future__ import annotations

import os

import streamlit as st

from src.llm import llm_settings
from src.rag_pipeline import RagPipeline

CONFIG_KEYS = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
)
STARTER_QUESTIONS = (
    "What is this document about?",
    "What are the main conclusions?",
    "Which key numbers or dates are mentioned?",
)
RETRY_KINDS = ("duplicate", "removed")  # re-attempted once re-indexing is ticked

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")


def apply_secrets() -> None:
    """Copy hosted app secrets into os.environ (existing env vars win)."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets file: local run configured via .env
        return
    for key in CONFIG_KEYS:
        if key in secrets and not os.environ.get(key):
            os.environ[key] = str(secrets[key])


def get_pipeline() -> RagPipeline:
    """Session-persisted pipeline (one embedding-model load per session).

    Stored in session_state rather than cache_resource so tests can
    inject a pre-seeded pipeline via AppTest session_state, and so each
    user session owns a mutable index that Clear can reset.
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RagPipeline()
    return st.session_state.pipeline


def upload_key(upload) -> str:
    """Identity of one uploaded file (Streamlit's file_id when present)."""
    file_id = getattr(upload, "file_id", None)
    return str(file_id) if file_id else f"{upload.name}:{getattr(upload, 'size', 0)}"


def pending_uploads(uploads, statuses: dict, replace: bool) -> list:
    """Filter sidebar uploads down to files needing indexing.

    Streamlit reruns the script on every interaction while the files stay
    in the uploader, so a file is pending only if it has no recorded
    outcome yet -- or was skipped as a duplicate name / removed by the
    user, and re-indexing has since been enabled.

    Pure helper (no Streamlit calls) so the policy is unit-testable
    without driving the file_uploader widget -- which AppTest in
    streamlit 1.54 does not support (renders UnknownElement).
    """
    pending = []
    for upload in uploads or []:
        status = statuses.get(upload_key(upload))
        if status is None or (replace and status[0] in RETRY_KINDS):
            pending.append(upload)
    return pending


def ingest_upload(pipeline: RagPipeline, upload, replace: bool) -> tuple:
    """Index one upload; return its (kind, message) status."""
    try:
        report = pipeline.ingest_documents(
            [(upload.name, upload.getvalue())], replace=replace
        )
    except Exception as exc:  # defensive: pipeline reports per-file issues itself
        return ("skipped", f"Indexing failed for {upload.name}: {exc}")
    if report["documents_added"]:
        return ("indexed", f"Indexed: {upload.name}")
    reason = next(iter(report["skipped"].values()), "Not indexed.")
    kind = "duplicate" if reason.startswith("Duplicate") else "skipped"
    return (kind, f"Skipped {upload.name}: {reason}")


def transcript_markdown(history: list) -> str:
    """Conversation as a Markdown document (for the download button)."""
    lines = ["# PDF RAG Assistant - conversation", ""]
    for turn in history:
        lines += [f"## {turn['question']}", "", str(turn["answer"]), ""]
        cited = [
            f"- {h.get('document', '?')} - Page {h.get('page', '?')} "
            f"(score {float(h.get('score', 0.0) or 0.0):.3f})"
            for h in turn["sources"]
        ]
        if cited:
            lines += ["**Sources**", "", *cited, ""]
    return "\n".join(lines)


def render_sources(sources: list) -> None:
    """Expandable sources: one tab per retrieved chunk."""
    if not sources:
        st.caption("No relevant passages found.")
        return
    with st.expander(f"Sources ({len(sources)})"):
        tabs = st.tabs(
            [f"{i}. {h.get('document', '?')} p.{h.get('page', '?')}"
             for i, h in enumerate(sources, 1)]
        )
        for tab, hit in zip(tabs, sources):
            score = float(hit.get("score", 0.0) or 0.0)
            with tab:
                st.markdown(
                    f"**{hit.get('document', '?')} — Page {hit.get('page', '?')}**"
                )
                st.progress(min(max(score, 0.0), 1.0), text=f"Relevance {score:.3f}")
                st.write(str(hit.get("text", ""))[:1500])


def answer_question(
    pipeline: RagPipeline,
    history: list,
    question: str,
    top_k: int,
    use_memory: bool,
    scope: list,
) -> None:
    """Show the question, stream the answer, then record the turn."""
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                sources, pieces = pipeline.query_stream(
                    question,
                    top_k=top_k,
                    history=history if use_memory else None,
                    documents=scope or None,
                )
            answer = str(st.write_stream(pieces)).strip()
        except (ValueError, RuntimeError) as exc:  # already user-friendly
            st.error(str(exc))
            return
        except Exception:  # never leak internals or keys
            st.error("Something went wrong while answering. Please try again.")
            return
        render_sources(sources)
    history.append({"question": question, "answer": answer, "sources": sources})


apply_secrets()
pipeline = get_pipeline()
st.session_state.setdefault("history", [])
st.session_state.setdefault("upload_status", {})
st.session_state.setdefault("uploader_gen", 0)
history: list = st.session_state.history
statuses: dict = st.session_state.upload_status

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_gen}",
    )
    replace = bool(st.session_state.get("replace", False))
    new_files = pending_uploads(uploads, statuses, replace)
    if new_files:
        with st.spinner(f"Indexing {len(new_files)} PDF(s)..."):
            for upload in new_files:
                statuses[upload_key(upload)] = ingest_upload(pipeline, upload, replace)
    for upload in uploads or []:
        kind, message = statuses.get(upload_key(upload), ("indexed", ""))
        if kind == "removed":
            st.caption(message)
        elif kind != "indexed":
            st.warning(message)

    for stat in pipeline.document_stats():
        name = str(stat["document"])
        label, action = st.columns([5, 1], vertical_alignment="center")
        label.markdown(
            f"**{name}**  \n{stat['pages']} page(s) · {stat['chunks']} chunk(s)"
        )
        if action.button("✕", key=f"rm_{name}", help=f"Remove {name} from the index"):
            pipeline.remove_document(name)
            for upload in uploads or []:
                if upload.name == name:
                    statuses[upload_key(upload)] = (
                        "removed",
                        f"Removed {name}. Tick re-indexing in Settings to add it back.",
                    )
            st.rerun()

    scope: list = []
    if len(pipeline.documents) > 1:
        scope = st.multiselect(
            "Search in",
            pipeline.documents,
            placeholder="All documents",
            help="Limit answers to the selected documents.",
        )

    with st.expander("Settings"):
        top_k = st.slider("Retrieved chunks (Top-K)", 1, 10, 5)
        st.checkbox(
            "Re-index files with the same name", value=False, key="replace",
            help="If off, already-indexed filenames are skipped as duplicates.",
        )
        use_memory = st.checkbox(
            "Use conversation memory", value=True,
            help="Lets follow-up questions refer to earlier ones in this conversation.",
        )

    if st.button("New conversation", disabled=not history, use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.download_button(
        "Download transcript",
        transcript_markdown(history),
        file_name="conversation.md",
        mime="text/markdown",
        disabled=not history,
        use_container_width=True,
    )
    if st.button("Clear index", use_container_width=True):
        pipeline.clear()
        st.session_state.history = []
        st.session_state.upload_status = {}
        # A new widget key empties the uploader; otherwise the files still
        # sitting in it would be re-indexed on the very next run.
        st.session_state.uploader_gen += 1
        st.rerun()

    llm = llm_settings()
    where = "local Ollama" if llm["provider"] == "ollama" else llm["host"].split("//")[-1]
    st.caption(f"Answers by `{llm['model']}` via {where}")

# ---------- Main ----------
st.title("PDF RAG Assistant")
st.caption("Ask questions about your PDFs. Answers come only from your documents, with page-level sources.")

prompt = st.chat_input(
    "Ask a question about your documents"
    if pipeline.documents
    else "Upload a PDF in the sidebar to start",
    disabled=not pipeline.documents,
)
question = (prompt or st.session_state.pop("pending_question", None) or "").strip()

if not pipeline.documents:
    st.info("Upload one or more PDFs in the sidebar to get started.")
    for column, (step, text) in zip(
        st.columns(3),
        (
            ("1. Upload", "Add PDFs in the sidebar. They are indexed in this session only."),
            ("2. Ask", "Type a question, then follow up as in a normal chat."),
            ("3. Verify", "Open Sources under any answer to see the exact passages used."),
        ),
    ):
        column.markdown(f"**{step}**  \n{text}")
elif not history and not question:
    st.markdown("**Try asking**")
    for i, (column, starter) in enumerate(zip(st.columns(3), STARTER_QUESTIONS)):
        if column.button(starter, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending_question = starter
            st.rerun()

for turn in history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        render_sources(turn["sources"])

if question and pipeline.documents:
    answer_question(pipeline, history, question, top_k, use_memory, scope)

if history and pipeline.documents:
    if st.button("Regenerate last answer", key="regenerate"):
        st.session_state.pending_question = history.pop()["question"]
        st.rerun()
