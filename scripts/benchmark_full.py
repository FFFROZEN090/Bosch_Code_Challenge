#!/usr/bin/env python3
"""Full Pipeline Benchmark: Regex vs LLM for Routing and Query Rewrite.

Compares four strategy combinations on the 10 standard test questions:
  A) Regex routing + keyword rewrite  (current system)
  B) LLM routing   + keyword rewrite
  C) Regex routing + LLM rewrite
  D) LLM routing   + LLM rewrite

Measures per-stage latency, accuracy, and total pipeline time.

Usage:
    python scripts/benchmark_full.py
"""

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from me_assistant.agent.router import route_query
from me_assistant.agent.llm_router import llm_route_query
from me_assistant.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, TEST_QUESTIONS_PATH, FAISS_INDEX_DIR, DOCS_DIR,
)
from me_assistant.eval.metrics import EXPECTED_ROUTES, EXPECTED_KEYWORDS
from me_assistant.ingest.indexer import load_faiss_index
from me_assistant.ingest.loader import load_document
from me_assistant.ingest.splitter import create_full_doc_chunk
from me_assistant.retrieval.retriever import (
    retrieve_by_series, retrieve_by_model, retrieve_all_docs,
)
from me_assistant.agent.prompts import format_prompt


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    """Call Ollama chat API."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


_LLM_REWRITE_PROMPT = """\
Rewrite this question to improve retrieval from technical ECU specification documents.
Add relevant technical terms, model numbers, and specification keywords.
Return ONLY the rewritten query, nothing else.

Original question: {question}
Detected model: {model}

Rewritten query:"""


def llm_rewrite_query(question: str, matched_models: list[str]) -> tuple[str, float]:
    """Rewrite query using LLM. Returns (rewritten_query, latency_ms)."""
    model = matched_models[0] if matched_models else "unknown"
    prompt = _LLM_REWRITE_PROMPT.format(question=question, model=model)

    start = time.time()
    rewritten = _call_ollama(prompt)
    latency_ms = (time.time() - start) * 1000

    return rewritten.strip(), latency_ms


def keyword_rewrite_query(question: str, matched_models: list[str]) -> tuple[str, float]:
    """Rewrite query using keyword expansion (current approach)."""
    start = time.time()

    parts = [question]
    if matched_models:
        model = matched_models[0]
        parts.append(f"{model} specifications technical data")
        if "750" in model:
            parts.append("ECU-700 series automotive controller")
        elif "850" in model:
            parts.append("ECU-800 series automotive controller")
    parts.append("features parameters performance specifications")

    rewritten = " ".join(parts)
    latency_ms = (time.time() - start) * 1000
    return rewritten, latency_ms


# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

def load_resources():
    """Load FAISS index and full doc chunks."""
    index = load_faiss_index()
    full_doc_chunks = [
        create_full_doc_chunk(load_document(f))
        for f in sorted(DOCS_DIR.glob("*.md"))
    ]
    return index, full_doc_chunks


def load_questions() -> list[dict]:
    """Load the 10 standard test questions."""
    questions = []
    with open(TEST_QUESTIONS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["Question_ID"])
            questions.append({
                "id": qid,
                "question": row["Question"],
                "expected_route": EXPECTED_ROUTES[qid],
            })
    return questions


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def do_retrieve(index, full_doc_chunks, route, matched_models, query):
    """Run retrieval based on route. Returns context string."""
    if route in ("COMPARE", "UNKNOWN"):
        return retrieve_all_docs(full_doc_chunks)

    series = route.split("_")[1]  # "ECU_700" -> "700"
    if matched_models:
        docs = retrieve_by_model(index, query, matched_models[0])
    else:
        docs = retrieve_by_series(index, query, series)

    return "\n\n".join(doc.page_content for doc in docs) if docs else ""


# ---------------------------------------------------------------------------
# Run a single pipeline combination
# ---------------------------------------------------------------------------

def run_pipeline(question: str, qid: int, index, full_doc_chunks,
                 use_llm_router: bool, use_llm_rewrite: bool):
    """Run full pipeline and return detailed timing results."""

    # --- Stage 1: Routing ---
    if use_llm_router:
        route_result, route_ms = llm_route_query(question)
        route = route_result.route
        matched_models = route_result.matched_models
    else:
        start = time.time()
        route_result = route_query(question)
        route_ms = (time.time() - start) * 1000
        route = route_result.route
        matched_models = route_result.matched_models

    route_correct = (route == EXPECTED_ROUTES.get(qid, ""))

    # --- Stage 2: Retrieval ---
    start = time.time()
    context = do_retrieve(index, full_doc_chunks, route, matched_models, question)
    retrieve_ms = (time.time() - start) * 1000

    # --- Stage 3: Rewrite (simulate: rewrite the original query) ---
    if use_llm_rewrite:
        rewritten, rewrite_ms = llm_rewrite_query(question, matched_models)
    else:
        rewritten, rewrite_ms = keyword_rewrite_query(question, matched_models)

    # --- Stage 4: Synthesis ---
    prompt = format_prompt(question, context, route)
    start = time.time()
    answer = _call_ollama(prompt)
    synth_ms = (time.time() - start) * 1000

    # --- Check answer ---
    keywords = EXPECTED_KEYWORDS.get(qid, [])
    answer_lower = answer.lower()
    answer_correct = all(kw.lower() in answer_lower for kw in keywords)

    total_ms = route_ms + retrieve_ms + rewrite_ms + synth_ms

    return {
        "qid": qid,
        "route": route,
        "route_correct": route_correct,
        "answer_correct": answer_correct,
        "route_ms": route_ms,
        "retrieve_ms": retrieve_ms,
        "rewrite_ms": rewrite_ms,
        "synth_ms": synth_ms,
        "total_ms": total_ms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = [
    ("A", "Regex + Keyword", False, False),
    ("B", "LLM + Keyword",   True,  False),
    ("C", "Regex + LLM",     False, True),
    ("D", "LLM + LLM",       True,  True),
]


def main():
    print("Loading FAISS index and documents...")
    index, full_doc_chunks = load_resources()

    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    print(f"Running 4 strategies x {len(questions)} questions = "
          f"{4 * len(questions)} pipeline runs")
    print(f"Estimated time: ~{4 * len(questions) * 5 // 60} minutes\n")

    all_results = {}

    for label, name, use_llm_router, use_llm_rewrite in STRATEGIES:
        print(f"{'='*60}")
        print(f"  Strategy {label}: {name}")
        print(f"    Router: {'LLM' if use_llm_router else 'Regex'}  |  "
              f"Rewrite: {'LLM' if use_llm_rewrite else 'Keyword'}")
        print(f"{'='*60}")

        results = []
        for q in questions:
            print(f"  Q{q['id']:>2}...", end="", flush=True)
            r = run_pipeline(
                q["question"], q["id"], index, full_doc_chunks,
                use_llm_router, use_llm_rewrite,
            )
            results.append(r)
            mark = "ok" if r["answer_correct"] else "FAIL"
            route_mark = "ok" if r["route_correct"] else "WRONG"
            print(f" route={r['route']:<8}[{route_mark}] "
                  f"answer[{mark}] "
                  f"({r['route_ms']:.0f} + {r['retrieve_ms']:.0f} + "
                  f"{r['rewrite_ms']:.0f} + {r['synth_ms']:.0f} = "
                  f"{r['total_ms']:.0f}ms)")

        all_results[label] = results
        print()

    # --- Print summary tables ---
    print_summary(all_results)


def print_summary(all_results: dict):
    """Print comprehensive comparison tables."""

    print()
    print("=" * 90)
    print("  FULL PIPELINE BENCHMARK RESULTS")
    print("=" * 90)

    # Table 1: Per-question accuracy
    print()
    print("  Table 1: Routing & Answer Accuracy")
    print()
    header = f"  {'Q#':<4}"
    for label, name, _, _ in STRATEGIES:
        header += f" {label}: {name:<18}"
    print(header)
    print(f"  {'--':<4}" + f" {'------------------':<20}" * 4)

    for i in range(len(list(all_results.values())[0])):
        qid = list(all_results.values())[0][i]["qid"]
        line = f"  Q{qid:<3}"
        for label, _, _, _ in STRATEGIES:
            r = all_results[label][i]
            r_mark = "R:ok" if r["route_correct"] else "R:X "
            a_mark = "A:ok" if r["answer_correct"] else "A:X "
            line += f" {r_mark} {a_mark}{'':>10}"
        print(line)

    # Table 2: Latency breakdown
    print()
    print("  Table 2: Average Latency Breakdown (ms)")
    print()
    print(f"  {'Stage':<16}", end="")
    for label, name, _, _ in STRATEGIES:
        print(f" {label}: {name:<16}", end="")
    print()
    print(f"  {'-----':<16}" + f" {'------------------':<18}" * 4)

    stages = [
        ("Route", "route_ms"),
        ("Retrieve", "retrieve_ms"),
        ("Rewrite", "rewrite_ms"),
        ("Synthesize", "synth_ms"),
        ("TOTAL", "total_ms"),
    ]

    for stage_name, key in stages:
        prefix = "  " if stage_name != "TOTAL" else "  "
        bold = "**" if stage_name == "TOTAL" else ""
        line = f"{prefix}{bold}{stage_name:<14}{bold}"
        for label, _, _, _ in STRATEGIES:
            results = all_results[label]
            avg = sum(r[key] for r in results) / len(results)
            if avg < 1:
                line += f" {'<1':>8} ms{'':>8}"
            else:
                line += f" {avg:>8.0f} ms{'':>8}"
        print(line)

    # Table 3: Summary metrics
    print()
    print("  Table 3: Summary")
    print()
    print(f"  {'Metric':<24}", end="")
    for label, name, _, _ in STRATEGIES:
        print(f" {label}: {name:<16}", end="")
    print()
    print(f"  {'------':<24}" + f" {'------------------':<18}" * 4)

    for metric_name, key in [("Route Accuracy", "route_correct"),
                              ("Answer Accuracy", "answer_correct")]:
        line = f"  {metric_name:<24}"
        for label, _, _, _ in STRATEGIES:
            results = all_results[label]
            correct = sum(1 for r in results if r[key])
            total = len(results)
            line += f" {correct}/{total:<17}"
        print(line)

    # Avg total latency
    line = f"  {'Avg Total Latency':<24}"
    for label, _, _, _ in STRATEGIES:
        results = all_results[label]
        avg = sum(r["total_ms"] for r in results) / len(results)
        line += f" {avg:>6.0f} ms{'':>9}"
    print(line)

    # Overhead vs Strategy A
    baseline_avg = sum(r["total_ms"] for r in all_results["A"]) / len(all_results["A"])
    line = f"  {'Overhead vs A':<24}"
    for label, _, _, _ in STRATEGIES:
        results = all_results[label]
        avg = sum(r["total_ms"] for r in results) / len(results)
        overhead = avg - baseline_avg
        if label == "A":
            line += f" {'baseline':>8}{'':>10}"
        else:
            line += f" {'+' if overhead > 0 else ''}{overhead:>6.0f} ms{'':>9}"

    print(line)

    print()
    print("-" * 90)
    print()
    print("  Strategy A = Current system (Regex routing + Keyword rewrite)")
    print("  Strategy B = LLM routing + Keyword rewrite")
    print("  Strategy C = Regex routing + LLM query rewrite")
    print("  Strategy D = LLM routing + LLM query rewrite (full LLM pipeline)")
    print()


if __name__ == "__main__":
    main()
