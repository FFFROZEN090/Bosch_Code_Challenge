"""LangGraph agent state definition."""

from typing import TypedDict


class AgentState(TypedDict):
    """State that flows through the LangGraph agent pipeline.

    Fields:
        question:           The user's original question.
        route:              Classification result: ECU_700 | ECU_800 | COMPARE | UNKNOWN.
        matched_models:     Specific models detected in the query (e.g. ["ECU-850b"]).
        route_reason:       Human-readable explanation of the routing decision.
        context:            Retrieved document text to feed into the LLM.
        answer:             The final synthesized answer.
        sources:            List of source references (dicts with source_file, chunk_id).
        confidence:         Confidence score (0.0 - 1.0) for the answer.
        latency_ms:         Total processing time in milliseconds.
        needs_human_review: Whether the query requires human review before synthesis.
        review_reason:      Why human review is needed (empty if not needed).
        search_query:       Rewritten query for retrieval (defaults to question).
        retrieval_attempts: Number of retrieval attempts (max 2 before forcing synthesis).
        evidence_sufficient: Whether retrieved evidence is sufficient.
        evidence_gap:       Description of what's missing from the evidence.
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
    needs_human_review: bool
    review_reason: str
    search_query: str
    retrieval_attempts: int
    evidence_sufficient: bool
    evidence_gap: str
