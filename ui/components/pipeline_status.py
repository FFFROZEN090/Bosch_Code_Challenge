"""Pipeline step-by-step visualization component."""

import streamlit as st


def render_pipeline_steps(state: dict):
    """Show pipeline steps as a collapsible status expander."""
    route = state.get("route", "UNKNOWN")
    models = state.get("matched_models", [])
    sources = state.get("sources", [])
    evidence_ok = state.get("evidence_sufficient", True)
    attempts = state.get("retrieval_attempts", 1)
    confidence = state.get("confidence", 0.0)
    needs_review = state.get("needs_human_review", False)
    latency_ms = state.get("latency_ms", 0.0)

    with st.status("Pipeline Complete", expanded=False, state="complete"):
        # Step 1: Classify
        model_info = f" ({', '.join(models)})" if models else ""
        st.write(f"**Classify** \u2192 `{route}`{model_info}")

        # Step 2: Retrieve
        st.write(f"**Retrieve** \u2192 {len(sources)} document chunks")

        # Step 3: Evidence check
        ev_label = "Sufficient" if evidence_ok else "Insufficient"
        retry_note = f" (attempts: {attempts})" if attempts > 1 else ""
        st.write(f"**Check Evidence** \u2192 {ev_label}{retry_note}")

        # Step 4: Confidence
        review_note = " \u26a0 flagged for review" if needs_review else ""
        st.write(f"**Validate Confidence** \u2192 {confidence:.0%}{review_note}")

        # Step 5: Synthesize
        st.write(f"**Synthesize** \u2192 completed in {latency_ms:.0f} ms")
