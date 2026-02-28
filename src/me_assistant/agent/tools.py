"""LangChain tools for ECU documentation retrieval.

These tools provide an autonomous tool-based interface for the agent,
wrapping core retrieval functions so the LLM can select retrieval
strategies dynamically.
"""

import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool

from me_assistant.retrieval.retriever import (
    retrieve_by_series,
    retrieve_by_model,
    retrieve_all_docs,
)

logger = logging.getLogger(__name__)


def make_tools(index: FAISS, full_doc_chunks: list[Document]) -> list:
    """Create LangChain tools with injected dependencies.

    Uses closures to inject the FAISS index and document chunks
    so tools can be used in a LangGraph ToolNode.

    Args:
        index: FAISS vector store for similarity search.
        full_doc_chunks: Full-document chunks for comparison retrieval.

    Returns:
        List of LangChain tool functions.
    """

    @tool
    def search_ecu700(query: str) -> str:
        """Search ECU-700 series documentation for technical specifications.

        Use this tool when the question is about ECU-750 or the 700 series.
        Returns relevant document chunks from ECU-700 series manuals.
        """
        docs = retrieve_by_series(index, query, "700")
        if not docs:
            return "No results found in ECU-700 series documentation."
        logger.info("search_ecu700 returned %d chunks", len(docs))
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def search_ecu800(query: str) -> str:
        """Search ECU-800 series documentation for technical specifications.

        Use this tool when the question is about ECU-850, ECU-850b,
        or the 800 series.
        Returns relevant document chunks from ECU-800 series manuals.
        """
        docs = retrieve_by_series(index, query, "800")
        if not docs:
            return "No results found in ECU-800 series documentation."
        logger.info("search_ecu800 returned %d chunks", len(docs))
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def compare_models(feature: str) -> str:  # pylint: disable=unused-argument
        """Compare a specific feature across all ECU models.

        Use this tool for comparison queries that span multiple models
        or series. Returns complete documentation from all ECU models
        for cross-referencing.

        Args:
            feature: The feature or specification to compare across models.
        """
        text = retrieve_all_docs(full_doc_chunks)
        logger.info("compare_models returned %d chars of context", len(text))
        return text

    @tool
    def get_full_specs(model: str) -> str:
        """Get the complete specification table for a specific ECU model.

        Args:
            model: Model identifier, e.g. "ECU-750", "ECU-850", "ECU-850b".

        Returns the full specification data for the requested model.
        """
        docs = retrieve_by_model(index, "specifications features", model)
        if not docs:
            return f"No specifications found for model {model}."
        logger.info("get_full_specs(%s) returned %d chunks", model, len(docs))
        return "\n\n".join(doc.page_content for doc in docs)

    return [search_ecu700, search_ecu800, compare_models, get_full_specs]
