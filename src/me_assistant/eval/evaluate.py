"""Evaluation runner: run all test questions through the agent and score results.

Supports two modes:
- run_evaluation(): Direct evaluation, returns results dict.
- run_mlflow_evaluation(): MLflow-tracked evaluation with logged metrics and artifacts.
"""

import csv
import json
import logging
import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd

from me_assistant.config import TEST_QUESTIONS_PATH, OLLAMA_MODEL, EMBEDDING_MODEL
from me_assistant.ingest.indexer import load_faiss_index
from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.agent.graph import build_graph
from me_assistant.eval.metrics import (
    check_answer_accuracy,
    check_routing_correctness,
    check_source_correctness,
    compute_overall_scores,
    make_mlflow_metrics,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "me-assistant-evaluation"


def load_test_questions() -> list[dict]:
    """Load test questions from CSV."""
    questions = []
    with open(TEST_QUESTIONS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                "question_id": int(row["Question_ID"]),
                "category": row["Category"],
                "question": row["Question"],
                "expected_answer": row["Expected_Answer"],
                "criteria": row["Evaluation_Criteria"],
            })
    return questions


def _build_graph():
    """Build the agent graph from saved index and documents."""
    index = load_faiss_index()
    loaded_docs = load_all_documents()
    _, full_doc_chunks = split_all_documents(loaded_docs)
    return build_graph(index, full_doc_chunks)


def _run_questions(graph, questions):
    """Run all questions through the graph, return per-question results."""
    per_question = []

    for q in questions:
        qid = q["question_id"]
        logger.info("Q%d: %s", qid, q["question"])

        start = time.time()
        state = graph.invoke({"question": q["question"]})
        elapsed = (time.time() - start) * 1000

        answer = state.get("answer", "")
        route = state.get("route", "")
        sources = json.dumps(state.get("sources", []))

        answer_ok = check_answer_accuracy(qid, answer)
        route_ok = check_routing_correctness(qid, route)
        source_ok = check_source_correctness(qid, sources)

        status = "PASS" if answer_ok else "FAIL"
        logger.info(
            "  Q%d %s | route=%s (correct=%s) | %.0fms",
            qid, status, route, route_ok, elapsed,
        )

        per_question.append({
            "question_id": qid,
            "question": q["question"],
            "category": q["category"],
            "expected_answer": q["expected_answer"],
            "answer": answer,
            "route": route,
            "sources": sources,
            "answer_correct": answer_ok,
            "route_correct": route_ok,
            "source_correct": source_ok,
            "latency_ms": elapsed,
        })

    return per_question


def run_evaluation() -> dict:
    """Run all test questions through the agent pipeline and evaluate.

    Returns:
        Dict with per_question results and overall scores.
    """
    graph = _build_graph()
    questions = load_test_questions()
    per_question = _run_questions(graph, questions)
    overall = compute_overall_scores(per_question)
    return {"per_question": per_question, "overall": overall}


def run_mlflow_evaluation() -> dict:
    """Run evaluation with full MLflow tracking and mlflow.evaluate().

    Pipeline:
    1. Run predictions through the agent graph
    2. Build custom MLflow metrics (answer_accuracy, routing, source, latency)
    3. Call mlflow.evaluate() with predictions and custom metrics
    4. Log additional artifacts (per-question CSV, summary JSON)

    Returns:
        Dict with per_question results, overall scores, and mlflow run_id.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    graph = _build_graph()
    questions = load_test_questions()

    with mlflow.start_run(run_name="full-evaluation") as run:
        # Log evaluation parameters
        mlflow.log_params({
            "llm_model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "num_questions": len(questions),
            "evaluation_type": "keyword_matching",
        })

        # Run all questions through the graph
        per_question = _run_questions(graph, questions)
        overall = compute_overall_scores(per_question)

        # Build eval DataFrame with predictions for mlflow.evaluate()
        eval_df = pd.DataFrame({
            "question": [r["question"] for r in per_question],
            "expected_answer": [r["expected_answer"] for r in per_question],
            "answer": [r["answer"] for r in per_question],
        })

        # Create custom MLflow metrics from results
        custom_metrics = make_mlflow_metrics(per_question)

        # Run mlflow.evaluate() with custom metrics
        eval_result = mlflow.evaluate(
            data=eval_df,
            predictions="answer",
            targets="expected_answer",
            extra_metrics=custom_metrics,
        )
        logger.info("mlflow.evaluate() metrics: %s", eval_result.metrics)

        # Log our aggregate metrics (supplements mlflow.evaluate output)
        mlflow.log_metrics({
            "overall_accuracy": overall["accuracy"],
            "overall_pass_count": overall["pass_count"],
            "overall_routing_accuracy": overall["routing_accuracy"],
            "overall_source_accuracy": overall["source_accuracy"],
            "overall_avg_latency_ms": overall["avg_latency_ms"],
            "overall_p95_latency_ms": overall["p95_latency_ms"],
            "overall_max_latency_ms": overall["max_latency_ms"],
        })

        # Save per-question results as CSV artifact
        results_df = pd.DataFrame(per_question)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "evaluation_results.csv"
            results_df.to_csv(csv_path, index=False)

            summary = {
                "overall": overall,
                "per_question_summary": [
                    {
                        "question_id": r["question_id"],
                        "category": r["category"],
                        "pass": r["answer_correct"],
                        "route_correct": r["route_correct"],
                        "latency_ms": round(r["latency_ms"], 1),
                    }
                    for r in per_question
                ],
            }
            summary_path = Path(tmpdir) / "evaluation_summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )

            mlflow.log_artifacts(tmpdir, artifact_path="evaluation")

        run_id = run.info.run_id
        logger.info("Evaluation logged to MLflow run: %s", run_id)

    return {
        "per_question": per_question,
        "overall": overall,
        "mlflow_run_id": run_id,
    }
