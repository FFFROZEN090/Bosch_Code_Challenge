"""LangGraph agent state definition."""

from typing import TypedDict


class AgentState(TypedDict):
    """State that flows through the LangGraph agent pipeline.

    Fields:
        question:       The user's original question.
        route:          Classification result: ECU_700 | ECU_800 | COMPARE | UNKNOWN.
        matched_models: Specific models detected in the query (e.g. ["ECU-850b"]).
        route_reason:   Human-readable explanation of the routing decision.
        context:        Retrieved document text to feed into the LLM.
        answer:         The final synthesized answer.
        sources:        List of source references (dicts with source_file, chunk_id).
        confidence:     Confidence score (0.0 - 1.0) for the answer.
        latency_ms:     Total processing time in milliseconds.
    """
    question: str
    route: str
    matched_models: list[str]
    route_reason: str
    context: str
    answer: str
    sources: list[dict]
    confidence: float
    latency_ms: float
