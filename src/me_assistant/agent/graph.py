"""Build and compile the LangGraph StateGraph for the ME Engineering Assistant.

Graph structure:
    START → classify → conditional_edge:
      ├─ ECU_700  → retrieve_single  → check_evidence → conditional_edge:
      │     ├─ sufficient   → validate_confidence → synthesize → END
      │     └─ insufficient → rewrite_query → retrieve_single (loop, max 2x)
      ├─ ECU_800  → retrieve_single  → check_evidence → (same as above)
      ├─ COMPARE  → retrieve_compare → check_evidence → validate_confidence → synthesize → END
      └─ UNKNOWN  → retrieve_compare → check_evidence → validate_confidence → synthesize → END

The check_evidence node enables multi-step retrieval: if the initial retrieval
returns insufficient context, the query is rewritten and retrieval retried.

The validate_confidence node may trigger a LangGraph interrupt for
human-in-the-loop review when confidence is low.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from me_assistant.agent.state import AgentState
from me_assistant.agent.nodes import (
    classify_node,
    make_retrieve_single_node,
    make_retrieve_compare_node,
    check_evidence_node,
    rewrite_query_node,
    validate_confidence_node,
    make_synthesize_node,
)


def _route_after_classify(state: dict) -> str:
    """Conditional edge: pick retrieval node based on route."""
    route = state["route"]
    if route in ("ECU_700", "ECU_800"):
        return "retrieve_single"
    return "retrieve_compare"


def _route_after_evidence(state: dict) -> str:
    """Conditional edge: proceed to synthesis or retry retrieval."""
    if state.get("evidence_sufficient", True):
        return "validate_confidence"
    return "rewrite_query"


def build_graph(
    index: FAISS,
    full_doc_chunks: list[Document],
    enable_hitl: bool = False,
) -> StateGraph:
    """Build and compile the agent graph.

    Args:
        index: FAISS vector store for single-source retrieval.
        full_doc_chunks: Full-document chunks for comparison retrieval.
        enable_hitl: If True, enable human-in-the-loop with a checkpointer
                     so interrupt() can pause the graph.

    Returns:
        A compiled LangGraph ready for .invoke().
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve_single", make_retrieve_single_node(index))
    graph.add_node("retrieve_compare", make_retrieve_compare_node(full_doc_chunks))
    graph.add_node("check_evidence", check_evidence_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("validate_confidence", validate_confidence_node)
    graph.add_node("synthesize", make_synthesize_node())

    # Edges: START → classify → retrieval
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _route_after_classify)

    # Both retrieval paths feed into evidence checking
    graph.add_edge("retrieve_single", "check_evidence")
    graph.add_edge("retrieve_compare", "check_evidence")

    # Evidence check: proceed or retry
    graph.add_conditional_edges("check_evidence", _route_after_evidence)

    # Retry loop: rewrite → re-retrieve → check again
    graph.add_edge("rewrite_query", "retrieve_single")

    # Final pipeline
    graph.add_edge("validate_confidence", "synthesize")
    graph.add_edge("synthesize", END)

    # Checkpointer is required for interrupt() to work in HITL mode
    checkpointer = MemorySaver() if enable_hitl else None
    return graph.compile(checkpointer=checkpointer)
