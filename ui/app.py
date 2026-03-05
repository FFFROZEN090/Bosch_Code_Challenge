"""ME Engineering Assistant — Streamlit demo UI.

Launch with:
    streamlit run ui/app.py --server.port 8501
"""

import sys
import uuid
from pathlib import Path

# Ensure both src/ and project root are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402
from langgraph.types import Command  # noqa: E402

from me_assistant.ingest.indexer import load_faiss_index  # noqa: E402
from me_assistant.ingest.loader import load_document  # noqa: E402
from me_assistant.ingest.splitter import create_full_doc_chunk  # noqa: E402
from me_assistant.agent.graph import build_graph  # noqa: E402
from me_assistant.config import DOCS_DIR  # noqa: E402

from ui.config import PAGE_TITLE, PAGE_ICON, LAYOUT, WELCOME_MESSAGE  # noqa: E402
from ui.styles.theme import inject_theme  # noqa: E402
from ui.components.header import render_header  # noqa: E402
from ui.components.sidebar import render_sidebar  # noqa: E402
from ui.components.metadata import render_metadata  # noqa: E402
from ui.components.pipeline_status import render_pipeline_steps  # noqa: E402

# ── Page config (must be the first Streamlit call) ──────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
inject_theme()


# ── Load pipeline once (with HITL enabled) ──────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def _load_pipeline():
    """Load FAISS index and build the LangGraph agent (cached)."""
    index = load_faiss_index()
    full_doc_chunks = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        loaded = load_document(md_file)
        full_doc_chunks.append(create_full_doc_chunk(loaded))
    return build_graph(index, full_doc_chunks, enable_hitl=True)


graph = _load_pipeline()

# ── Session state ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_review" not in st.session_state:
    st.session_state.pending_review = None

# ── Layout ──────────────────────────────────────────────────────────
render_header()
render_sidebar()

# ── Chat history ────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("state"):
            render_pipeline_steps(msg["state"])
            render_metadata(msg["state"])


def _run_graph(question, thread_id, resume_value=None):
    """Invoke the graph and return (state, interrupt_payload_or_None)."""
    config = {"configurable": {"thread_id": thread_id}}

    if resume_value is not None:
        # Resume from interrupt with human input
        state = graph.invoke(Command(resume=resume_value), config)
    else:
        state = graph.invoke({"question": question}, config)

    # Check if the graph was interrupted (not yet at END)
    snapshot = graph.get_state(config)
    if snapshot.next:  # graph paused at a node, not finished
        # Extract interrupt payload from the snapshot
        interrupt_payload = None
        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_payload = task.interrupts[0].value
                    break
        return state, interrupt_payload
    return state, None


# ── Human-in-the-loop review panel ─────────────────────────────────
if st.session_state.pending_review:
    review = st.session_state.pending_review
    with st.chat_message("assistant"):
        st.warning(
            f"**Human Review Required**\n\n"
            f"**Reason:** {review['reason']}\n\n"
            f"**Detected route:** `{review['route']}`\n\n"
            f"**Confidence:** {review['confidence']:.0%}\n\n"
            f"**Context preview:** {review['context_preview'][:150]}..."
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Approve & Continue", type="primary", use_container_width=True):
                thread_id = review["thread_id"]
                with st.spinner("Resuming pipeline..."):
                    state, new_interrupt = _run_graph(
                        None, thread_id, resume_value={"action": "approve"}
                    )
                st.session_state.pending_review = None
                answer = state.get("answer", "Sorry, I could not generate an answer.")
                st.markdown(answer)
                render_pipeline_steps(state)
                render_metadata(state)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "state": state,
                })
                st.rerun()

        with col2:
            if st.button("Correct Route", use_container_width=True):
                st.session_state.show_route_picker = True
                st.rerun()

        # Route correction picker
        if st.session_state.get("show_route_picker"):
            new_route = st.selectbox(
                "Select correct route:",
                ["ECU_700", "ECU_800", "COMPARE"],
                key="route_correction",
            )
            if st.button("Submit Correction", type="primary"):
                thread_id = review["thread_id"]
                with st.spinner("Resuming with corrected route..."):
                    state, new_interrupt = _run_graph(
                        None, thread_id, resume_value={"route": new_route}
                    )
                st.session_state.pending_review = None
                st.session_state.show_route_picker = False
                answer = state.get("answer", "Sorry, I could not generate an answer.")
                st.markdown(answer)
                render_pipeline_steps(state)
                render_metadata(state)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "state": state,
                })
                st.rerun()


# ── Handle input ────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about ECU specifications...")

# Sidebar demo-question buttons set this value and call st.rerun()
if st.session_state.get("pending_question"):
    user_input = st.session_state.pop("pending_question")

if user_input and not st.session_state.pending_review:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run pipeline with unique thread_id for HITL checkpointer
    thread_id = str(uuid.uuid4())

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state, interrupt_payload = _run_graph(user_input, thread_id)

        if interrupt_payload:
            # Pipeline paused — show review panel
            interrupt_payload["thread_id"] = thread_id
            st.session_state.pending_review = interrupt_payload
            st.rerun()
        else:
            # Normal completion
            answer = state.get("answer", "Sorry, I could not generate an answer.")
            st.markdown(answer)
            render_pipeline_steps(state)
            render_metadata(state)

    # Persist to history (only if completed, not interrupted)
    if not st.session_state.pending_review:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "state": state,
        })
