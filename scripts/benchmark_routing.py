#!/usr/bin/env python3
"""Benchmark: Regex Router vs LLM Router.

Runs both routing strategies against the same test questions and edge cases,
measuring accuracy, latency, and consistency (multi-run stability).

Usage:
    python scripts/benchmark_routing.py              # 1 run, quick check
    python scripts/benchmark_routing.py --runs 3     # 3 runs for consistency
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from me_assistant.agent.router import route_query
from me_assistant.agent.llm_router import llm_route_query
from me_assistant.config import TEST_QUESTIONS_PATH
from me_assistant.eval.metrics import EXPECTED_ROUTES, BENCHMARK_EDGE_CASES


def load_test_questions() -> list[dict]:
    """Load the 10 standard test questions from CSV."""
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


def build_test_set() -> list[dict]:
    """Combine standard questions + edge cases."""
    questions = load_test_questions()
    for case in BENCHMARK_EDGE_CASES:
        questions.append({
            "id": case["id"],
            "question": case["question"],
            "expected_route": case["expected_route"],
        })
    return questions


def run_regex_router(questions: list[dict]) -> list[dict]:
    """Run regex router on all questions. Returns per-question results."""
    results = []
    for q in questions:
        start = time.time()
        result = route_query(q["question"])
        latency_ms = (time.time() - start) * 1000

        results.append({
            "id": q["id"],
            "question": q["question"],
            "expected": q["expected_route"],
            "predicted": result.route,
            "correct": result.route == q["expected_route"],
            "latency_ms": latency_ms,
            "reason": result.reason,
        })
    return results


def run_llm_router(questions: list[dict], num_runs: int) -> list[dict]:
    """Run LLM router on all questions, multiple times for consistency.

    Returns per-question results with consistency info.
    """
    results = []
    total = len(questions) * num_runs
    current = 0

    for q in questions:
        run_routes = []
        run_latencies = []

        for run_idx in range(num_runs):
            current += 1
            print(f"  LLM routing [{current}/{total}] "
                  f"Q{q['id']} run {run_idx + 1}/{num_runs}...",
                  end="", flush=True)

            route_result, latency_ms = llm_route_query(q["question"])
            run_routes.append(route_result.route)
            run_latencies.append(latency_ms)

            mark = "ok" if route_result.route == q["expected_route"] else "WRONG"
            print(f" {route_result.route} ({latency_ms:.0f}ms) [{mark}]")

        # Use first run as the "predicted" result
        predicted = run_routes[0]
        consistent = len(set(run_routes)) == 1

        results.append({
            "id": q["id"],
            "question": q["question"],
            "expected": q["expected_route"],
            "predicted": predicted,
            "correct": predicted == q["expected_route"],
            "latency_ms": sum(run_latencies) / len(run_latencies),
            "all_routes": run_routes,
            "consistent": consistent,
        })

    return results


def print_results(regex_results: list[dict], llm_results: list[dict],
                  num_runs: int) -> None:
    """Print formatted comparison table."""
    print()
    print("=" * 78)
    print("  ROUTING STRATEGY BENCHMARK RESULTS")
    print("=" * 78)

    # Per-question table
    print()
    print(f"  {'Q#':<4} {'Question':<40} {'Regex':<12} {'LLM':<12}")
    print(f"  {'--':<4} {'--------':<40} {'-----':<12} {'---':<12}")

    for rr, lr in zip(regex_results, llm_results):
        q_text = rr["question"][:38] + ".." if len(rr["question"]) > 40 else rr["question"]
        r_mark = "pass" if rr["correct"] else "FAIL"
        l_mark = "pass" if lr["correct"] else "FAIL"
        r_str = f"{rr['predicted']:<8} {r_mark}"
        l_str = f"{lr['predicted']:<8} {l_mark}"
        print(f"  {rr['id']:<4} {q_text:<40} {r_str:<12} {l_str:<12}")

    # Summary
    regex_correct = sum(1 for r in regex_results if r["correct"])
    llm_correct = sum(1 for r in llm_results if r["correct"])
    total = len(regex_results)

    regex_latencies = [r["latency_ms"] for r in regex_results]
    llm_latencies = [r["latency_ms"] for r in llm_results]

    sorted_llm_lat = sorted(llm_latencies)
    p95_idx = min(int(len(sorted_llm_lat) * 0.95), len(sorted_llm_lat) - 1)
    llm_p95 = sorted_llm_lat[p95_idx] if sorted_llm_lat else 0

    llm_consistent = sum(1 for r in llm_results if r["consistent"])

    print()
    print("-" * 78)
    print(f"  {'Metric':<24} {'Regex':<20} {'LLM':<20}")
    print(f"  {'------':<24} {'-----':<20} {'---':<20}")
    avg_regex_lat = sum(regex_latencies) / len(regex_latencies)
    avg_llm_lat = sum(llm_latencies) / len(llm_latencies)
    regex_p95 = sorted(regex_latencies)[min(p95_idx, len(regex_latencies) - 1)]
    consistency_label = f"Consistency ({num_runs}x)"

    print(f"  {'Accuracy':<24} {regex_correct}/{total:<19} {llm_correct}/{total}")
    print(f"  {'Avg Latency':<24} {avg_regex_lat:.1f} ms{'':<15} {avg_llm_lat:.0f} ms")
    print(f"  {'P95 Latency':<24} {regex_p95:.1f} ms{'':<15} {llm_p95:.0f} ms")
    print(f"  {consistency_label:<24} {total}/{total} (100%)"
          f"{'':>8}{llm_consistent}/{total} ({llm_consistent*100//total}%)")
    print("-" * 78)

    # Speedup
    avg_regex = sum(regex_latencies) / len(regex_latencies)
    avg_llm = sum(llm_latencies) / len(llm_latencies)
    if avg_regex > 0:
        speedup = avg_llm / avg_regex
        print(f"\n  Regex router is {speedup:,.0f}x faster than LLM router.")

    # Disagreements
    disagreements = [(rr, lr) for rr, lr in zip(regex_results, llm_results)
                     if rr["predicted"] != lr["predicted"]]
    if disagreements:
        print(f"\n  Routing disagreements ({len(disagreements)}):")
        for rr, lr in disagreements:
            q_text = rr["question"][:60]
            print(f"    Q{rr['id']}: Regex={rr['predicted']}, LLM={lr['predicted']}  "
                  f"(expected: {rr['expected']})")
            print(f"          \"{q_text}\"")

    # LLM inconsistencies
    inconsistencies = [r for r in llm_results if not r["consistent"]]
    if inconsistencies:
        print(f"\n  LLM routing inconsistencies ({len(inconsistencies)}):")
        for r in inconsistencies:
            print(f"    Q{r['id']}: {r['all_routes']}  "
                  f"\"{r['question'][:60]}\"")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark regex vs LLM routing strategies")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of LLM runs per question (default: 1)")
    args = parser.parse_args()

    questions = build_test_set()
    print(f"Loaded {len(questions)} questions ({10} standard + "
          f"{len(BENCHMARK_EDGE_CASES)} edge cases)")
    print(f"LLM runs per question: {args.runs}")
    print(f"Estimated time: ~{len(questions) * args.runs * 15 // 60} minutes\n")

    # Regex router (instant)
    print("Running regex router...")
    regex_results = run_regex_router(questions)
    print(f"  Done. All {len(regex_results)} questions routed in "
          f"{sum(r['latency_ms'] for r in regex_results):.1f}ms total.\n")

    # LLM router (slow)
    print("Running LLM router...")
    llm_results = run_llm_router(questions, num_runs=args.runs)

    # Print comparison
    print_results(regex_results, llm_results, args.runs)


if __name__ == "__main__":
    main()
