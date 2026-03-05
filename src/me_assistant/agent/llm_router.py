"""LLM-based query router for benchmarking comparison.

Uses Mistral 7B via Ollama to classify queries into routing categories.
This exists solely for A/B comparison against the deterministic regex router.
"""

import json
import time
import urllib.request

from me_assistant.agent.router import RouteResult
from me_assistant.config import OLLAMA_BASE_URL, OLLAMA_MODEL

_VALID_ROUTES = {"ECU_700", "ECU_800", "COMPARE", "UNKNOWN"}

_LLM_ROUTER_PROMPT = """\
Classify this question into exactly one category:
- ECU_700: about ECU-700 series or ECU-750 only
- ECU_800: about ECU-800 series, ECU-850, or ECU-850b only
- COMPARE: comparing multiple models or asking about all models
- UNKNOWN: cannot determine or not about ECU specifications

Question: {question}

Respond with ONLY the category name, nothing else."""


def _call_ollama(prompt: str) -> str:
    """Call Ollama chat API for routing classification."""
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    return data["message"]["content"]


def _parse_route(raw: str) -> str:
    """Extract a valid route from LLM response text.

    Handles common LLM response quirks: extra whitespace, punctuation,
    explanatory text around the route name, markdown formatting.
    """
    cleaned = raw.strip().strip("`").strip("*").strip()

    # Try exact match first
    upper = cleaned.upper()
    if upper in _VALID_ROUTES:
        return upper

    # Try to find a valid route anywhere in the response
    for route in _VALID_ROUTES:
        if route in upper:
            return route

    return "UNKNOWN"


def llm_route_query(question: str) -> tuple[RouteResult, float]:
    """Route a query using LLM classification.

    Args:
        question: The user's question string.

    Returns:
        Tuple of (RouteResult, latency_ms).
    """
    prompt = _LLM_ROUTER_PROMPT.format(question=question)

    start = time.time()
    raw_response = _call_ollama(prompt)
    latency_ms = (time.time() - start) * 1000

    route = _parse_route(raw_response)

    return RouteResult(
        route=route,
        matched_models=[],
        reason=f"LLM classified as {route} (raw: {raw_response.strip()[:80]})",
    ), latency_ms
