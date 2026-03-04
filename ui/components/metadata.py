"""Answer metadata panel: route badge, confidence bar, latency, sources."""

import streamlit as st

_BADGE_CLASS = {
    "ECU_700": "route-ecu700",
    "ECU_800": "route-ecu800",
    "COMPARE": "route-compare",
    "UNKNOWN": "route-unknown",
}


def render_metadata(state: dict):
    """Render a metadata card below each answer."""
    route = state.get("route", "UNKNOWN")
    confidence = state.get("confidence", 0.0)
    latency_ms = state.get("latency_ms", 0.0)
    sources = state.get("sources", [])
    route_reason = state.get("route_reason", "")

    st.markdown('<div class="metadata-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        badge = _BADGE_CLASS.get(route, "route-unknown")
        st.markdown(
            f'**Route:** <span class="route-badge {badge}">{route}</span>',
            unsafe_allow_html=True,
        )
        if route_reason:
            st.caption(route_reason)

    with col2:
        st.markdown(f"**Confidence:** {confidence:.0%}")
        st.progress(min(confidence, 1.0))

    with col3:
        latency_s = latency_ms / 1000 if latency_ms else 0
        st.markdown(f"**Latency:** {latency_s:.1f}s")

    if sources:
        unique = list({s.get("source_file", "") for s in sources if s.get("source_file")})
        if unique:
            st.markdown("**Sources:** " + " &bull; ".join(f"`{s}`" for s in sorted(unique)),
                        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
