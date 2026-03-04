"""Header banner component."""

import streamlit as st


def render_header():
    """Render the branded header banner."""
    st.markdown(
        """
        <div class="main-header">
            <h1>\u2699\ufe0f ME Engineering Assistant</h1>
            <p>Intelligent ECU Technical Specification Agent
               &nbsp;\u2022&nbsp; LangGraph + FAISS + Mistral 7B</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
