"""Custom evaluation metrics for ME Engineering Assistant."""

# Key facts that MUST appear in a correct answer, per question ID.
# Each entry is a list of required keywords/phrases (case-insensitive).
# A question passes if ALL required keywords are found in the answer.
EXPECTED_KEYWORDS = {
    1: ["+85°C", "-40°C"],
    2: ["2 GB", "LPDDR4"],
    3: ["NPU", "5 TOPS"],
    4: ["NPU", "4 GB", "2 GB", "1.5 GHz", "1.2 GHz"],
    5: ["single channel", "dual channel"],
    6: ["1.7A", "550mA"],
    7: ["OTA"],
    8: ["2 MB", "16 GB", "32 GB"],
    9: ["+105°C", "+85°C"],
    10: ["me-driver-ctl", "--enable-npu"],
}

# Expected route for each question ID.
EXPECTED_ROUTES = {
    1: "ECU_700",
    2: "ECU_800",
    3: "ECU_800",
    4: "COMPARE",
    5: "COMPARE",
    6: "ECU_800",
    7: "COMPARE",
    8: "COMPARE",
    9: "COMPARE",
    10: "ECU_800",
}

# Expected source documents for each question ID.
EXPECTED_SOURCES = {
    1: ["ECU-700_Series_Manual.md"],
    2: ["ECU-800_Series_Base.md"],
    3: ["ECU-800_Series_Plus.md"],
    4: ["ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"],
    5: ["ECU-700_Series_Manual.md", "ECU-800_Series_Base.md"],
    6: ["ECU-800_Series_Plus.md"],
    7: ["ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"],
    8: ["ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"],
    9: ["ECU-700_Series_Manual.md", "ECU-800_Series_Base.md", "ECU-800_Series_Plus.md"],
    10: ["ECU-800_Series_Plus.md"],
}


def check_answer_accuracy(question_id: int, answer: str) -> bool:
    """Check if the answer contains all required keywords for the question.

    Args:
        question_id: 1-based question ID.
        answer: The model's answer text.

    Returns:
        True if all required keywords are found.
    """
    keywords = EXPECTED_KEYWORDS.get(question_id, [])
    answer_lower = answer.lower()
    return all(kw.lower() in answer_lower for kw in keywords)


def check_routing_correctness(question_id: int, route: str) -> bool:
    """Check if the route matches the expected route for the question."""
    expected = EXPECTED_ROUTES.get(question_id)
    return route == expected


def check_source_correctness(question_id: int, sources_str: str) -> bool:
    """Check if the sources reference the expected documents.

    Args:
        question_id: 1-based question ID.
        sources_str: JSON string of sources list from model output.

    Returns:
        True if all expected source files appear in the sources.
    """
    expected = EXPECTED_SOURCES.get(question_id, [])
    return all(src in sources_str for src in expected)


def compute_overall_scores(results: list[dict]) -> dict:
    """Compute aggregate metrics from per-question results.

    Args:
        results: List of dicts with keys: question_id, answer_correct,
                 route_correct, source_correct, latency_ms.

    Returns:
        Dict with accuracy, routing_accuracy, source_accuracy,
        avg_latency_ms, max_latency_ms, pass_count, total.
    """
    total = len(results)
    pass_count = sum(1 for r in results if r["answer_correct"])
    route_count = sum(1 for r in results if r["route_correct"])
    source_count = sum(1 for r in results if r["source_correct"])
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]

    return {
        "accuracy": pass_count / total if total else 0.0,
        "routing_accuracy": route_count / total if total else 0.0,
        "source_accuracy": source_count / total if total else 0.0,
        "pass_count": pass_count,
        "total": total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }
