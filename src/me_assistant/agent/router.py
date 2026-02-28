"""Rule-based query router for ECU document retrieval.

Routes queries to the correct retrieval strategy:
- ECU_700:  Single-source retrieval from ECU-700 series docs
- ECU_800:  Single-source retrieval from ECU-800 series docs
- COMPARE:  All-docs retrieval for cross-model comparisons
- UNKNOWN:  Fallback — treated as COMPARE (search everything)
"""

import re
from typing import NamedTuple


class RouteResult(NamedTuple):
    """Result of query routing."""
    route: str
    matched_models: list[str]
    reason: str


# --- Pattern definitions ---

# Models by series
_700_MODELS = {"ECU-750", "750"}
_800_MODELS = {"ECU-850b", "850b", "ECU-850", "850"}

# Must check 850b BEFORE 850 to avoid partial match
_MODEL_PATTERNS = [
    (re.compile(r"\bECU[-\s]?850b\b", re.IGNORECASE), "ECU-850b", "800"),
    (re.compile(r"\b850b\b", re.IGNORECASE), "ECU-850b", "800"),
    (re.compile(r"\bECU[-\s]?850\b", re.IGNORECASE), "ECU-850", "800"),
    (re.compile(r"\b850\b(?!b)", re.IGNORECASE), "ECU-850", "800"),
    (re.compile(r"\bECU[-\s]?750\b", re.IGNORECASE), "ECU-750", "700"),
    (re.compile(r"\b750\b", re.IGNORECASE), "ECU-750", "700"),
]

_SERIES_PATTERNS = [
    (re.compile(r"\bECU[-\s]?700\b|\b700\s*series\b", re.IGNORECASE), "700"),
    (re.compile(r"\bECU[-\s]?800\b|\b800\s*series\b", re.IGNORECASE), "800"),
]

# Compare triggers — checked FIRST (highest priority)
_COMPARE_EXPLICIT = re.compile(
    r"\b(compare|vs\.?|versus|differenc|differ|contrast)\b",
    re.IGNORECASE,
)
_COMPARE_MULTI_MODEL = re.compile(
    r"\b(all\s+(ECU|model)|each\s+model|every\s+model|across\s+.*(model|ECU))\b",
    re.IGNORECASE,
)
_COMPARE_FEATURE_SCAN = re.compile(
    r"\bwhich\s+(ECU\s*models?|models?|ECUs?)\s+(support|have|offer|provide|include)\b",
    re.IGNORECASE,
)
_COMPARE_SUPERLATIVE = re.compile(
    r"\b(harshest|highest|lowest|best|worst|most|least|fastest|"
    r"slowest|largest|smallest|biggest|maximum|minimum)\b",
    re.IGNORECASE,
)


def _find_models(query: str) -> list[tuple[str, str]]:
    """Find all ECU model mentions in the query.

    Returns list of (model_name, series) tuples, deduplicated.
    """
    found = []
    seen = set()
    for pattern, model, series in _MODEL_PATTERNS:
        if pattern.search(query) and model not in seen:
            found.append((model, series))
            seen.add(model)
    return found


def _is_compare_query(query: str, models: list[tuple[str, str]]) -> str | None:
    """Check if the query is a comparison. Returns reason string or None."""
    if _COMPARE_EXPLICIT.search(query):
        return "explicit comparison keyword"

    if _COMPARE_MULTI_MODEL.search(query):
        return "multi-model keyword (all/each/every/across)"

    if _COMPARE_FEATURE_SCAN.search(query):
        return "feature availability scan (which models support...)"

    # Superlative only triggers COMPARE when no single model is specified.
    # "maximum temperature for ECU-750" is a single-source query,
    # but "harshest temperature conditions" (no model) is a comparison.
    if _COMPARE_SUPERLATIVE.search(query) and len(models) != 1:
        return "superlative implies comparison across models"

    # Multiple models from different series mentioned
    if models:
        series_set = {s for _, s in models}
        if len(series_set) > 1:
            return "models from multiple series mentioned"
        if len(models) > 1:
            return "multiple models from same series mentioned"

    return None


def route_query(query: str) -> RouteResult:
    """Classify a user query into a retrieval route.

    Priority order:
    1. COMPARE triggers (explicit keywords, multi-model, superlatives)
    2. Model-specific match (ECU-750, ECU-850, ECU-850b)
    3. Series-level match (700 series, 800 series)
    4. UNKNOWN fallback (treated as COMPARE)

    Args:
        query: The user's question string.

    Returns:
        RouteResult with route, matched_models, and reason.
    """
    models = _find_models(query)
    model_names = [m for m, _ in models]

    # Priority 1: Compare triggers
    compare_reason = _is_compare_query(query, models)
    if compare_reason:
        return RouteResult(
            route="COMPARE",
            matched_models=model_names,
            reason=compare_reason,
        )

    # Priority 2: Single model match
    if len(models) == 1:
        model_name, series = models[0]
        return RouteResult(
            route=f"ECU_{series}",
            matched_models=[model_name],
            reason=f"matched model {model_name}",
        )

    # Priority 3: Series-level match
    for pattern, series in _SERIES_PATTERNS:
        if pattern.search(query):
            return RouteResult(
                route=f"ECU_{series}",
                matched_models=[],
                reason=f"matched {series} series keyword",
            )

    # Priority 4: Fallback
    return RouteResult(
        route="UNKNOWN",
        matched_models=[],
        reason="no model or series detected, will search all docs",
    )
