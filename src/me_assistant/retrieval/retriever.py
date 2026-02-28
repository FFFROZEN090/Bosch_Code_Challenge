"""Retrieval functions for querying ECU documentation."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from me_assistant.config import FAISS_TOP_K


def retrieve_by_series(
    index: FAISS,
    query: str,
    series: str,
    top_k: int = FAISS_TOP_K,
) -> list[Document]:
    """Retrieve chunks filtered by ECU series (700 or 800).

    Args:
        index: The FAISS vector store.
        query: User's question text.
        series: "700" or "800".
        top_k: Number of results to return.

    Returns:
        List of matching Document chunks, sorted by relevance.
    """
    results = index.similarity_search_with_score(query, k=top_k * 3)
    filtered = [
        doc for doc, _score in results
        if doc.metadata.get("series") == series
    ]
    return filtered[:top_k]


def retrieve_by_model(
    index: FAISS,
    query: str,
    model: str,
    top_k: int = FAISS_TOP_K,
) -> list[Document]:
    """Retrieve chunks filtered by specific ECU model.

    Args:
        index: The FAISS vector store.
        query: User's question text.
        model: e.g. "ECU-750", "ECU-850", "ECU-850b".
        top_k: Number of results to return.

    Returns:
        List of matching Document chunks, sorted by relevance.
    """
    results = index.similarity_search_with_score(query, k=top_k * 3)
    filtered = [
        doc for doc, _score in results
        if doc.metadata.get("model") == model
    ]
    return filtered[:top_k]


def retrieve_all_docs(full_doc_chunks: list[Document]) -> str:
    """Return the concatenated full text of all documents.

    Used by the COMPARE retrieval path — no similarity search needed
    since the docs are small enough to fit entirely in LLM context.

    Args:
        full_doc_chunks: List of full-document chunks from splitter.

    Returns:
        Concatenated text of all documents, separated by dividers.
    """
    parts = []
    for doc in full_doc_chunks:
        source = doc.metadata.get("source_file", "unknown")
        parts.append(f"--- Source: {source} ---\n{doc.page_content}")
    return "\n\n".join(parts)
