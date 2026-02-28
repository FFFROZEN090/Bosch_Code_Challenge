"""LangGraph node functions for the agent pipeline.

Each node takes AgentState as input and returns a partial state update dict.
Nodes: classify → retrieve_single / retrieve_compare → synthesize
"""

import json
import logging
import time
import urllib.request

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from me_assistant.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from me_assistant.agent.router import route_query
from me_assistant.agent.prompts import format_prompt
from me_assistant.retrieval.retriever import (
    retrieve_by_series,
    retrieve_by_model,
    retrieve_all_docs,
)

logger = logging.getLogger(__name__)


def classify_node(state: dict) -> dict:
    """Route the user's question to the appropriate retrieval strategy."""
    question = state["question"]
    result = route_query(question)

    logger.info(
        "Route: %s | models: %s | reason: %s",
        result.route, result.matched_models, result.reason,
    )

    return {
        "route": result.route,
        "matched_models": result.matched_models,
        "route_reason": result.reason,
    }


def make_retrieve_single_node(index: FAISS):
    """Create a retrieve node for single-source queries (ECU_700 or ECU_800).

    Uses closure to inject the FAISS index dependency.
    """
    def retrieve_single_node(state: dict) -> dict:
        question = state["question"]
        route = state["route"]
        matched_models = state.get("matched_models", [])

        # If a specific model was matched, filter by model; otherwise by series
        if matched_models:
            model = matched_models[0]
            docs = retrieve_by_model(index, question, model)
            logger.info("Retrieved %d chunks for model=%s", len(docs), model)
        else:
            series = route.split("_")[1]  # "ECU_700" -> "700"
            docs = retrieve_by_series(index, question, series)
            logger.info("Retrieved %d chunks for series=%s", len(docs), series)

        context = "\n\n".join(doc.page_content for doc in docs)
        sources = [
            {
                "source_file": doc.metadata.get("source_file", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
            for doc in docs
        ]

        return {"context": context, "sources": sources}

    return retrieve_single_node


def make_retrieve_compare_node(full_doc_chunks: list[Document]):
    """Create a retrieve node for comparison queries.

    Injects all documents as context (docs are small enough).
    """
    def retrieve_compare_node(state: dict) -> dict:  # pylint: disable=unused-argument
        context = retrieve_all_docs(full_doc_chunks)
        sources = [
            {
                "source_file": doc.metadata.get("source_file", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
            }
            for doc in full_doc_chunks
        ]

        logger.info(
            "COMPARE retrieval: injecting all %d documents (%d chars)",
            len(full_doc_chunks), len(context),
        )

        return {"context": context, "sources": sources}

    return retrieve_compare_node


def _call_ollama(prompt: str) -> str:
    """Call Ollama chat API using urllib (bypasses httpx 503 issue).

    Args:
        prompt: The formatted prompt string.

    Returns:
        The LLM response text.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return data["message"]["content"]


def make_synthesize_node():
    """Create the synthesis node that calls the LLM."""

    def synthesize_node(state: dict) -> dict:
        question = state["question"]
        context = state["context"]
        route = state["route"]

        prompt = format_prompt(question, context, route)

        logger.info("Calling LLM for synthesis (route=%s)...", route)
        start = time.time()
        answer = _call_ollama(prompt)
        elapsed_ms = (time.time() - start) * 1000

        logger.info("LLM responded in %.0fms (%d chars)", elapsed_ms, len(answer))

        return {
            "answer": answer,
            "confidence": 1.0,
            "latency_ms": elapsed_ms,
        }

    return synthesize_node
