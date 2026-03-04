"""ME Engineering Assistant — Streamlit demo UI.

Launch with:
    streamlit run ui/app.py --server.port 8501
"""

import sys
from pathlib import Path

# Ensure both src/ and project root are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

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


# ── Load pipeline once ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline...")
def _load_pipeline():
    """Load FAISS index and build the LangGraph agent (cached)."""
    index = load_faiss_index()
    full_doc_chunks = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        loaded = load_document(md_file)
        full_doc_chunks.append(create_full_doc_chunk(loaded))
    return build_graph(index, full_doc_chunks)


graph = _load_pipeline()

# ── Session state ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

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

# ── Handle input ────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about ECU specifications...")

# Sidebar demo-question buttons set this value and call st.rerun()
if st.session_state.get("pending_question"):
    user_input = st.session_state.pop("pending_question")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            state = graph.invoke({"question": user_input})

        # Answer
        answer = state.get("answer", "Sorry, I could not generate an answer.")
        st.markdown(answer)

        # Pipeline steps + metadata
        render_pipeline_steps(state)
        render_metadata(state)

    # Persist to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "state": state,
    })
