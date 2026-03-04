"""Blue-and-white theme for the ME Engineering Assistant UI."""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* ── Header banner ── */
    .main-header {
        background: linear-gradient(135deg, #1565C0 0%, #1E88E5 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(21, 101, 192, 0.25);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #BBDEFB;
        margin: 0.25rem 0 0 0;
        font-size: 0.95rem;
    }

    /* ── Route badges ── */
    .route-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .route-ecu700 { background: #E3F2FD; color: #1565C0; }
    .route-ecu800 { background: #E8F5E9; color: #2E7D32; }
    .route-compare { background: #F3E5F5; color: #7B1FA2; }
    .route-unknown { background: #F5F5F5; color: #616161; }

    /* ── Metadata card ── */
    .metadata-card {
        background: #F5F7FA;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #F5F7FA;
    }
    [data-testid="stSidebar"] hr {
        border-color: #E0E0E0;
    }

    /* ── Demo question buttons ── */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: white;
        border: 1px solid #BBDEFB;
        color: #1565C0;
        text-align: left;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #E3F2FD;
        border-color: #1E88E5;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
    }

    /* ── Progress bar color ── */
    .stProgress > div > div > div {
        background-color: #1565C0;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Expander styling ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1565C0;
    }
</style>
"""


def inject_theme():
    """Inject the custom CSS theme into the Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
