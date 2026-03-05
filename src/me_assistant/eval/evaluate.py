"""Evaluation runner: run all test questions through the agent and score results.

Supports two modes:
- run_evaluation(): Direct evaluation, returns results dict.
- run_mlflow_evaluation(): MLflow-tracked evaluation with logged metrics and artifacts.

Multi-run mode (num_runs > 1):
- Each question is run N times through the pipeline.
- Results include per-question pass rate (e.g. "2/3"), average latency,
  route consistency, and majority-vote accuracy.
- LLM Judge scores only the first run's answer to avoid N× judge overhead.
"""

import csv
import json
import logging
import tempfile
import time
from pathlib import Path
from statistics import mean

import mlflow
import pandas as pd

from me_assistant.config import (
    TEST_QUESTIONS_PATH, OLLAMA_MODEL, EMBEDDING_MODEL, ROUTING_STRATEGY,
)
from me_assistant.ingest.indexer import load_faiss_index
from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.agent.graph import build_graph
from me_assistant.eval.metrics import (
    check_answer_accuracy,
    check_routing_correctness,
    check_source_correctness,
    compute_overall_scores,
    llm_judge_answer,
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


def _run_questions(graph, questions, num_runs=1):
    """Run all questions through the graph, return per-question results.

    When num_runs > 1, each question is executed N times. Results include
    per-run details and aggregated metrics (pass rate, avg latency,
    route consistency).
    """
    per_question = []

    for q in questions:
        qid = q["question_id"]
        logger.info("Q%d: %s", qid, q["question"])

        run_results = []
        for run_idx in range(num_runs):
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
            if num_runs > 1:
                logger.info(
                    "  Q%d run %d/%d %s | route=%s (correct=%s) | %.0fms",
                    qid, run_idx + 1, num_runs, status, route, route_ok, elapsed,
                )
            else:
                logger.info(
                    "  Q%d %s | route=%s (correct=%s) | %.0fms",
                    qid, status, route, route_ok, elapsed,
                )

            run_results.append({
                "answer": answer,
                "route": route,
                "sources": sources,
                "answer_correct": answer_ok,
                "route_correct": route_ok,
                "source_correct": source_ok,
                "latency_ms": elapsed,
            })

        # Aggregate across runs
        pass_count = sum(1 for r in run_results if r["answer_correct"])
        route_correct_count = sum(1 for r in run_results if r["route_correct"])
        source_correct_count = sum(1 for r in run_results if r["source_correct"])
        all_routes = [r["route"] for r in run_results]
        route_consistent = len(set(all_routes)) == 1
        avg_latency = mean([r["latency_ms"] for r in run_results])

        # Use majority vote for pass/fail when num_runs > 1
        answer_correct = pass_count > num_runs // 2 if num_runs > 1 else run_results[0]["answer_correct"]

        entry = {
            "question_id": qid,
            "question": q["question"],
            "category": q["category"],
            "expected_answer": q["expected_answer"],
            "criteria": q["criteria"],
            # Use first run's answer/route/sources for judge scoring
            "answer": run_results[0]["answer"],
            "route": run_results[0]["route"],
            "sources": run_results[0]["sources"],
            "answer_correct": answer_correct,
            "route_correct": route_correct_count == num_runs,
            "source_correct": source_correct_count == num_runs,
            "latency_ms": avg_latency,
        }

        # Multi-run specific fields
        if num_runs > 1:
            entry.update({
                "pass_rate": f"{pass_count}/{num_runs}",
                "route_correct_rate": f"{route_correct_count}/{num_runs}",
                "route_consistent": route_consistent,
                "all_routes": all_routes,
                "all_latencies": [r["latency_ms"] for r in run_results],
                "num_runs": num_runs,
            })
            logger.info(
                "  Q%d aggregate: %s pass, routes=%s (consistent=%s), avg=%.0fms",
                qid, entry["pass_rate"], all_routes, route_consistent, avg_latency,
            )

        per_question.append(entry)

    return per_question


def _run_llm_judge(per_question: list[dict]) -> None:
    """Run LLM-as-Judge on all results, adding judge_score/judge_reason in-place."""
    logger.info("Running LLM-as-Judge on %d answers...", len(per_question))
    for r in per_question:
        judge = llm_judge_answer(
            question=r["question"],
            expected_answer=r["expected_answer"],
            actual_answer=r["answer"],
            criteria=r["criteria"],
        )
        r["judge_score"] = judge["score"]
        r["judge_reason"] = judge["reason"]
        logger.info(
            "  Q%d judge=%d/5: %s",
            r["question_id"], judge["score"], judge["reason"][:80],
        )


def run_evaluation(num_runs: int = 1) -> dict:
    """Run all test questions through the agent pipeline and evaluate.

    Args:
        num_runs: Number of times to run each question (default 1).
            When > 1, results include pass rate, route consistency,
            and averaged latency across runs.

    Returns:
        Dict with per_question results, overall scores, and eval config.
    """
    graph = _build_graph()
    questions = load_test_questions()

    logger.info(
        "Evaluation config: model=%s, routing=%s, num_runs=%d, questions=%d",
        OLLAMA_MODEL, ROUTING_STRATEGY, num_runs, len(questions),
    )

    per_question = _run_questions(graph, questions, num_runs=num_runs)
    _run_llm_judge(per_question)
    overall = compute_overall_scores(per_question)

    # Add multi-run aggregate stats
    if num_runs > 1:
        all_latencies = []
        for r in per_question:
            all_latencies.extend(r.get("all_latencies", [r["latency_ms"]]))
        total_runs = len(per_question) * num_runs
        total_pass = sum(
            int(r["pass_rate"].split("/")[0]) for r in per_question
        )
        route_consistent_count = sum(
            1 for r in per_question if r.get("route_consistent", True)
        )
        overall["all_runs_pass_rate"] = f"{total_pass}/{total_runs}"
        overall["all_runs_pass_pct"] = total_pass / total_runs if total_runs else 0.0
        overall["route_consistency"] = route_consistent_count / len(per_question)
        overall["num_runs"] = num_runs

    return {
        "per_question": per_question,
        "overall": overall,
        "config": {
            "model": OLLAMA_MODEL,
            "routing_strategy": ROUTING_STRATEGY,
            "num_runs": num_runs,
        },
    }


def run_mlflow_evaluation(num_runs: int = 1) -> dict:
    """Run evaluation with full MLflow tracking and mlflow.evaluate().

    Pipeline:
    1. Run predictions through the agent graph (N times per question)
    2. Build custom MLflow metrics (answer_accuracy, routing, source, latency)
    3. Call mlflow.evaluate() with predictions and custom metrics
    4. Log additional artifacts (per-question CSV, summary JSON)

    Args:
        num_runs: Number of times to run each question (default 1).

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
            "routing_strategy": ROUTING_STRATEGY,
            "num_questions": len(questions),
            "num_runs": num_runs,
            "evaluation_type": "keyword_matching + llm_judge",
        })

        # Run all questions through the graph, then score with LLM judge
        per_question = _run_questions(graph, questions, num_runs=num_runs)
        _run_llm_judge(per_question)
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
        aggregate = {
            "overall_accuracy": overall["accuracy"],
            "overall_pass_count": overall["pass_count"],
            "overall_routing_accuracy": overall["routing_accuracy"],
            "overall_source_accuracy": overall["source_accuracy"],
            "overall_avg_latency_ms": overall["avg_latency_ms"],
            "overall_p95_latency_ms": overall["p95_latency_ms"],
            "overall_max_latency_ms": overall["max_latency_ms"],
        }
        if "avg_judge_score" in overall:
            aggregate["overall_avg_judge_score"] = overall["avg_judge_score"]
        if num_runs > 1:
            aggregate["overall_all_runs_pass_pct"] = overall.get("all_runs_pass_pct", 0.0)
            aggregate["overall_route_consistency"] = overall.get("route_consistency", 0.0)
        mlflow.log_metrics(aggregate)

        # Save per-question results as CSV artifact
        results_df = pd.DataFrame(per_question)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "evaluation_results.csv"
            results_df.to_csv(csv_path, index=False)

            per_q_summary = []
            for r in per_question:
                entry = {
                    "question_id": r["question_id"],
                    "category": r["category"],
                    "pass": r["answer_correct"],
                    "route_correct": r["route_correct"],
                    "latency_ms": round(r["latency_ms"], 1),
                }
                if "judge_score" in r:
                    entry["judge_score"] = r["judge_score"]
                    entry["judge_reason"] = r.get("judge_reason", "")
                per_q_summary.append(entry)

            summary = {
                "overall": overall,
                "per_question_summary": per_q_summary,
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
        "config": {
            "model": OLLAMA_MODEL,
            "routing_strategy": ROUTING_STRATEGY,
            "num_runs": num_runs,
        },
    }
