"""Build and compile the LangGraph StateGraph for the ME Engineering Assistant.

Graph structure:
    START → classify → conditional_edge:
      ├─ ECU_700  → retrieve_single  → synthesize → END
      ├─ ECU_800  → retrieve_single  → synthesize → END
      ├─ COMPARE  → retrieve_compare → synthesize → END
      └─ UNKNOWN  → retrieve_compare → synthesize → END
"""

from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from me_assistant.agent.state import AgentState
from me_assistant.agent.nodes import (
    classify_node,
    make_retrieve_single_node,
    make_retrieve_compare_node,
    make_synthesize_node,
)


def _route_after_classify(state: dict) -> str:
    """Conditional edge: pick retrieval node based on route."""
    route = state["route"]
    if route in ("ECU_700", "ECU_800"):
        return "retrieve_single"
    return "retrieve_compare"


def build_graph(
    index: FAISS,
    full_doc_chunks: list[Document],
) -> StateGraph:
    """Build and compile the agent graph.

    Args:
        index: FAISS vector store for single-source retrieval.
        full_doc_chunks: Full-document chunks for comparison retrieval.

    Returns:
        A compiled LangGraph ready for .invoke().
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve_single", make_retrieve_single_node(index))
    graph.add_node("retrieve_compare", make_retrieve_compare_node(full_doc_chunks))
    graph.add_node("synthesize", make_synthesize_node())

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", _route_after_classify)
    graph.add_edge("retrieve_single", "synthesize")
    graph.add_edge("retrieve_compare", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()
