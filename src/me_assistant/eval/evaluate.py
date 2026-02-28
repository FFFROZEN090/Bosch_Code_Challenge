"""Evaluation runner: run all test questions through the agent and score results."""

import csv
import json
import logging
import time

from me_assistant.config import TEST_QUESTIONS_PATH
from me_assistant.ingest.indexer import load_faiss_index
from me_assistant.ingest.loader import load_all_documents
from me_assistant.ingest.splitter import split_all_documents
from me_assistant.agent.graph import build_graph
from me_assistant.eval.metrics import (
    check_answer_accuracy,
    check_routing_correctness,
    check_source_correctness,
    compute_overall_scores,
)

logger = logging.getLogger(__name__)


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


def run_evaluation() -> dict:
    """Run all test questions through the agent pipeline and evaluate.

    Returns:
        Dict with per_question results and overall scores.
    """
    # Load index and docs
    index = load_faiss_index()
    loaded_docs = load_all_documents()
    _, full_doc_chunks = split_all_documents(loaded_docs)
    graph = build_graph(index, full_doc_chunks)

    questions = load_test_questions()
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
            "answer": answer,
            "route": route,
            "sources": sources,
            "answer_correct": answer_ok,
            "route_correct": route_ok,
            "source_correct": source_ok,
            "latency_ms": elapsed,
        })

    overall = compute_overall_scores(per_question)
    return {"per_question": per_question, "overall": overall}
