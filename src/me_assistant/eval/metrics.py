"""Custom evaluation metrics for ME Engineering Assistant.

Provides three interfaces:
- Standalone functions: check_answer_accuracy(), check_routing_correctness(), etc.
- LLM-as-Judge: llm_judge_answer() uses Mistral 7B to score answers against criteria.
- MLflow-compatible metrics: make_mlflow_metrics() returns metrics for mlflow.evaluate().
"""

import json
import logging
import re
import urllib.request

from mlflow.metrics import MetricValue, make_metric

from me_assistant.config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

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


# Edge cases for routing benchmark (not part of the standard eval suite).
BENCHMARK_EDGE_CASES = [
    {"id": 11, "question": "Tell me about automotive systems", "expected_route": "UNKNOWN"},
    {"id": 12, "question": "ecu-850 ram", "expected_route": "ECU_800"},
    {"id": 13, "question": "What is the maximum temperature for ECU-750?",
     "expected_route": "ECU_700"},
    {"id": 14, "question": "How does ECU-850b compare to 850?", "expected_route": "COMPARE"},
    {"id": 15, "question": "What are the specs of the ECU-900?", "expected_route": "UNKNOWN"},
]


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


# ---------------------------------------------------------------------------
# LLM-as-Judge evaluation
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert evaluator for a technical documentation QA system about ECU specifications.

Score the following answer on a scale of 1-5:
  5 = Perfect: accurate, complete, well-structured
  4 = Good: mostly accurate, minor omissions
  3 = Acceptable: partially correct but missing key details
  2 = Poor: significant errors or omissions
  1 = Wrong: incorrect or irrelevant

Question: {question}
Reference Answer: {expected_answer}
Evaluation Criteria: {criteria}
Actual Answer: {actual_answer}

Respond in this exact format:
Score: <number>
Reason: <one sentence explanation>"""


def _call_ollama_judge(prompt: str) -> str:
    """Call Ollama chat API for LLM judge evaluation."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def llm_judge_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    criteria: str,
) -> dict:
    """Score an answer using LLM-as-Judge.

    Args:
        question: The original question.
        expected_answer: Reference answer from test data.
        actual_answer: The model's generated answer.
        criteria: Evaluation criteria from test-questions.csv.

    Returns:
        Dict with 'score' (1-5, or 0 on error) and 'reason' string.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        criteria=criteria,
        actual_answer=actual_answer,
    )
    try:
        raw = _call_ollama_judge(prompt)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("LLM judge call failed: %s", exc)
        return {"score": 0, "reason": f"LLM call failed: {exc}"}

    # Parse "Score: N"
    score_match = re.search(r"Score:\s*([1-5])", raw)
    score = int(score_match.group(1)) if score_match else 0

    # Parse "Reason: ..."
    reason_match = re.search(r"Reason:\s*(.+)", raw)
    reason = reason_match.group(1).strip() if reason_match else raw.strip()[:120]

    if not score_match:
        logger.warning("Could not parse judge score from: %s", raw[:120])
        reason = f"Parse error: {raw.strip()[:120]}"

    return {"score": score, "reason": reason}


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

    sorted_lat = sorted(latencies)
    p95_idx = int(len(sorted_lat) * 0.95)
    p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0

    scores = {
        "accuracy": pass_count / total if total else 0.0,
        "routing_accuracy": route_count / total if total else 0.0,
        "source_accuracy": source_count / total if total else 0.0,
        "pass_count": pass_count,
        "total": total,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": p95,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }

    # Include LLM judge average if present
    judge_scores = [r["judge_score"] for r in results if "judge_score" in r and r["judge_score"] > 0]
    if judge_scores:
        scores["avg_judge_score"] = sum(judge_scores) / len(judge_scores)

    return scores


# ---------------------------------------------------------------------------
# MLflow-compatible custom metrics for use with mlflow.evaluate()
# ---------------------------------------------------------------------------

def make_answer_accuracy_metric(question_ids: list[int]):
    """Create an MLflow metric that checks keyword-based answer accuracy.

    Uses closure to inject question_ids so the eval_fn can map each
    prediction row to its expected keywords.

    Args:
        question_ids: Ordered list of question IDs matching the eval data rows.
    """
    def eval_fn(predictions, targets, metrics):  # pylint: disable=unused-argument
        scores = []
        for i, pred in enumerate(predictions):
            qid = question_ids[i]
            score = 1.0 if check_answer_accuracy(qid, str(pred)) else 0.0
            scores.append(score)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return MetricValue(
            scores=scores,
            aggregate_results={"mean": mean_score},
        )

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="answer_accuracy",
    )


def make_routing_correctness_metric(question_ids: list[int], routes: list[str]):
    """Create an MLflow metric that checks routing correctness.

    Args:
        question_ids: Ordered list of question IDs.
        routes: Ordered list of actual routes from model output.
    """
    def eval_fn(predictions, targets, metrics):  # pylint: disable=unused-argument
        scores = []
        for i, _ in enumerate(predictions):
            qid = question_ids[i]
            route = routes[i]
            score = 1.0 if check_routing_correctness(qid, route) else 0.0
            scores.append(score)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return MetricValue(
            scores=scores,
            aggregate_results={"mean": mean_score},
        )

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="routing_correctness",
    )


def make_source_correctness_metric(question_ids: list[int], sources_list: list[str]):
    """Create an MLflow metric that checks source document correctness.

    Args:
        question_ids: Ordered list of question IDs.
        sources_list: Ordered list of JSON source strings from model output.
    """
    def eval_fn(predictions, targets, metrics):  # pylint: disable=unused-argument
        scores = []
        for i, _ in enumerate(predictions):
            qid = question_ids[i]
            sources_str = sources_list[i]
            score = 1.0 if check_source_correctness(qid, sources_str) else 0.0
            scores.append(score)
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return MetricValue(
            scores=scores,
            aggregate_results={"mean": mean_score},
        )

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="source_correctness",
    )


def make_latency_metric(latencies: list[float]):
    """Create an MLflow metric that reports response latency statistics.

    Args:
        latencies: Ordered list of latency_ms values from model output.
    """
    def eval_fn(predictions, targets, metrics):  # pylint: disable=unused-argument
        valid = [lat for lat in latencies if lat > 0]
        avg = sum(valid) / len(valid) if valid else 0.0
        p95_idx = int(len(valid) * 0.95)
        sorted_lat = sorted(valid)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)] if sorted_lat else 0.0
        return MetricValue(
            scores=list(latencies),
            aggregate_results={"mean": avg, "p95": p95, "max": max(valid) if valid else 0.0},
        )

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=False,
        name="response_latency_ms",
    )


def make_llm_judge_metric(judge_results: list[dict]):
    """Create an MLflow metric from pre-computed LLM judge scores.

    Args:
        judge_results: List of dicts with 'score' (1-5) and 'reason'.
    """
    def eval_fn(predictions, targets, metrics):  # pylint: disable=unused-argument
        # Normalize 1-5 scores to 0.0-1.0 for MLflow
        scores = [(r["score"] / 5.0) if r["score"] > 0 else 0.0 for r in judge_results]
        raw_scores = [r["score"] for r in judge_results]
        valid = [s for s in raw_scores if s > 0]
        return MetricValue(
            scores=scores,
            aggregate_results={
                "mean": sum(valid) / len(valid) if valid else 0.0,
                "min": min(valid) if valid else 0,
                "max": max(valid) if valid else 0,
            },
        )

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=True,
        name="llm_judge_score",
    )


def make_mlflow_metrics(per_question_results: list[dict]) -> list:
    """Build all MLflow custom metrics from evaluation results.

    Args:
        per_question_results: List of result dicts from _run_questions(),
            each containing question_id, route, sources, latency_ms,
            and optionally judge_score/judge_reason.

    Returns:
        List of MLflow metrics for mlflow.evaluate().
    """
    question_ids = [r["question_id"] for r in per_question_results]
    routes = [r["route"] for r in per_question_results]
    sources_list = [r["sources"] for r in per_question_results]
    latencies = [r["latency_ms"] for r in per_question_results]

    metrics = [
        make_answer_accuracy_metric(question_ids),
        make_routing_correctness_metric(question_ids, routes),
        make_source_correctness_metric(question_ids, sources_list),
        make_latency_metric(latencies),
    ]

    # Add LLM judge metric if judge results are present
    if per_question_results and "judge_score" in per_question_results[0]:
        judge_results = [
            {"score": r.get("judge_score", 0), "reason": r.get("judge_reason", "")}
            for r in per_question_results
        ]
        metrics.append(make_llm_judge_metric(judge_results))

    return metrics
