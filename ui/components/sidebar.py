"""Sidebar component: system info, demo questions, and document browser."""

import streamlit as st

from me_assistant.config import DOCS_DIR
from ui.config import DEMO_QUESTIONS


def render_sidebar():
    """Render the sidebar with system info, demo questions, and docs."""
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "Multi-source RAG agent that answers technical questions "
            "about ECU product specifications using a LangGraph pipeline."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", "3")
        col2.metric("ECU Models", "3")
        col3.metric("Accuracy", "9/10")

        st.divider()
        st.markdown("### Quick Demo")

        for item in DEMO_QUESTIONS:
            if st.button(item["label"], key=item["question"], use_container_width=True):
                st.session_state.pending_question = item["question"]
                st.rerun()

        st.divider()
        st.markdown("### Source Documents")
        doc_names = [
            "ECU-700_Series_Manual.md",
            "ECU-800_Series_Base.md",
            "ECU-800_Series_Plus.md",
        ]
        for name in doc_names:
            path = DOCS_DIR / name
            if path.exists():
                with st.expander(name.replace(".md", "").replace("_", " ")):
                    st.markdown(path.read_text(encoding="utf-8"))
