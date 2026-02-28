"""LangGraph node functions for the agent pipeline.

Each node takes AgentState as input and returns a partial state update dict.
Nodes: classify → retrieve → check_evidence → validate_confidence → synthesize

The check_evidence node enables multi-step retrieval: if retrieved evidence
is insufficient, the query is rewritten and retrieval retried (max 2 attempts).
"""

import json
import logging
import re
import time
import urllib.request

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.types import interrupt

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
        # Use rewritten query if available (from multi-step retrieval)
        query = state.get("search_query") or state["question"]
        route = state["route"]
        matched_models = state.get("matched_models", [])

        attempt = state.get("retrieval_attempts", 1)
        logger.info("Retrieval attempt %d with query: %s", attempt, query[:80])

        # If a specific model was matched, filter by model; otherwise by series
        if matched_models:
            model = matched_models[0]
            docs = retrieve_by_model(index, query, model)
            logger.info("Retrieved %d chunks for model=%s", len(docs), model)
        else:
            series = route.split("_")[1]  # "ECU_700" -> "700"
            docs = retrieve_by_series(index, query, series)
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


def validate_confidence_node(state: dict) -> dict:
    """Check retrieval confidence and flag low-confidence queries for human review.

    Confidence is LOW when:
    - route == UNKNOWN (no model/series detected)
    - Query mentions a model name not in our database (e.g., "ECU-900")
    - Context is empty (retrieval returned nothing)

    When confidence is low, the node triggers a LangGraph interrupt to pause
    the pipeline and request human input before proceeding to synthesis.
    """
    route = state.get("route", "")
    question = state.get("question", "")
    context = state.get("context", "")
    review_reasons = []

    # Check 1: UNKNOWN route
    if route == "UNKNOWN":
        review_reasons.append("No ECU model or series detected in query")

    # Check 2: Query mentions an unknown model (e.g., "ECU-900", "ECU-650")
    ecu_mentions = re.findall(r"\bECU[-\s]?(\d+\w*)\b", question, re.IGNORECASE)
    for mention in ecu_mentions:
        normalized = mention.lower().rstrip()
        if normalized not in ("750", "850", "850b", "700", "800"):
            review_reasons.append(f"Unknown model referenced: ECU-{mention}")

    # Check 3: Empty context (retrieval failure)
    if not context.strip():
        review_reasons.append("Retrieval returned no context")

    if review_reasons:
        reason = "; ".join(review_reasons)
        logger.warning("Low confidence — needs human review: %s", reason)

        # Interrupt the graph: pause and wait for human input
        human_input = interrupt({
            "reason": reason,
            "question": question,
            "route": route,
            "context_preview": context[:200] if context else "(empty)",
        })

        # If human provides a corrected route, update state
        if isinstance(human_input, dict):
            if "route" in human_input:
                logger.info("Human corrected route to: %s", human_input["route"])
                return {
                    "needs_human_review": False,
                    "review_reason": f"Human corrected: {reason}",
                    "route": human_input["route"],
                }

        return {
            "needs_human_review": False,
            "review_reason": f"Human approved: {reason}",
        }

    return {
        "needs_human_review": False,
        "review_reason": "",
    }


def check_evidence_node(state: dict) -> dict:
    """Check if retrieved evidence is sufficient for answering the query.

    For COMPARE/UNKNOWN routes, evidence is always sufficient because
    all documents are injected as context.

    For single-source routes (ECU_700/ECU_800), evidence is insufficient when:
    - Context is empty (retrieval returned nothing)
    - Context is very short (< 50 chars, likely a fragment)

    When insufficient, the graph loops back through rewrite_query for
    a second retrieval attempt (max 2 total).
    """
    route = state.get("route", "")
    context = state.get("context", "")
    retrieval_attempts = state.get("retrieval_attempts", 1)

    # COMPARE/UNKNOWN: all docs injected, always sufficient
    if route in ("COMPARE", "UNKNOWN"):
        logger.info("Evidence check: COMPARE route, always sufficient")
        return {"evidence_sufficient": True, "evidence_gap": ""}

    # Max attempts reached — proceed with whatever we have
    if retrieval_attempts >= 2:
        logger.info(
            "Evidence check: max attempts reached (%d), proceeding",
            retrieval_attempts,
        )
        return {"evidence_sufficient": True, "evidence_gap": ""}

    # Check for empty or very short context
    if not context.strip():
        logger.info("Evidence check: empty context, will retry")
        return {
            "evidence_sufficient": False,
            "evidence_gap": "no context retrieved",
        }

    if len(context) < 50:
        logger.info(
            "Evidence check: context too short (%d chars), will retry",
            len(context),
        )
        return {
            "evidence_sufficient": False,
            "evidence_gap": f"context too short ({len(context)} chars)",
        }

    logger.info(
        "Evidence check: sufficient (%d chars, attempt %d)",
        len(context), retrieval_attempts,
    )
    return {"evidence_sufficient": True, "evidence_gap": ""}


def rewrite_query_node(state: dict) -> dict:
    """Rewrite the search query for improved retrieval on retry.

    Expands the original question with model-specific context and
    technical keywords to improve FAISS similarity matching.
    """
    question = state["question"]
    matched_models = state.get("matched_models", [])
    evidence_gap = state.get("evidence_gap", "")
    attempt = state.get("retrieval_attempts", 1)

    parts = [question]

    # Add model-specific context to improve embedding similarity
    if matched_models:
        model = matched_models[0]
        parts.append(f"{model} specifications technical data")
        if "750" in model:
            parts.append("ECU-700 series automotive controller")
        elif "850" in model:
            parts.append("ECU-800 series automotive controller")

    # Add generic technical terms to broaden retrieval
    parts.append("features parameters performance specifications")

    rewritten = " ".join(parts)
    new_attempt = attempt + 1

    logger.info(
        "Query rewrite (attempt %d→%d, gap=%s): '%s' → '%s'",
        attempt, new_attempt, evidence_gap, question[:50], rewritten[:80],
    )

    return {
        "search_query": rewritten,
        "retrieval_attempts": new_attempt,
    }


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
