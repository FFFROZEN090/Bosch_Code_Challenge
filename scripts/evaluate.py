"""CLI script: Run evaluation on all test questions with MLflow tracking."""

import argparse
import logging

from me_assistant.eval.evaluate import run_evaluation, run_mlflow_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _print_results(results):
    """Print evaluation results table to stdout."""
    overall = results["overall"]
    config = results.get("config", {})
    num_runs = config.get("num_runs", 1)
    multi = num_runs > 1

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    if config:
        print(f"  Model: {config.get('model', '?')}  |  "
              f"Routing: {config.get('routing_strategy', '?')}  |  "
              f"Runs per question: {num_runs}")
    print("=" * 70)

    for r in results["per_question"]:
        if multi:
            pass_rate = r.get("pass_rate", "?/?")
            consistent = r.get("route_consistent", True)
            routes = r.get("all_routes", [r["route"]])
            consistency_str = f"{len(routes)}/{len(routes)}" if consistent else f"INCONSISTENT {routes}"
            judge = f"  judge={r['judge_score']}/5" if "judge_score" in r else ""
            print(
                f"  Q{r['question_id']:2d} [{pass_rate} PASS]  "
                f"route={r['route']:<10s} ({consistency_str})  "
                f"avg={r['latency_ms']:.0f}ms{judge}"
            )
        else:
            status = "PASS" if r["answer_correct"] else "FAIL"
            route_ok = "ok" if r["route_correct"] else "WRONG"
            judge = f"  judge={r['judge_score']}/5" if "judge_score" in r else ""
            print(
                f"  Q{r['question_id']:2d} [{status}]  route={r['route']:<10s} "
                f"({route_ok})  {r['latency_ms']:.0f}ms{judge}"
            )
        if "judge_reason" in r and r.get("judge_score", 0) > 0:
            print(f"         {r['judge_reason'][:72]}")

    print("-" * 70)
    print(f"  Accuracy (majority): {overall['pass_count']}/{overall['total']} "
          f"({overall['accuracy']:.0%})")
    if multi:
        print(f"  Pass rate (all runs):{overall.get('all_runs_pass_rate', '?')} "
              f"({overall.get('all_runs_pass_pct', 0):.0%})")
    print(f"  Routing accuracy:    {overall['routing_accuracy']:.0%}")
    if multi and "route_consistency" in overall:
        print(f"  Route consistency:   {overall['route_consistency']:.0%}")
    print(f"  Source accuracy:     {overall['source_accuracy']:.0%}")
    if "avg_judge_score" in overall:
        print(f"  LLM Judge (avg):     {overall['avg_judge_score']:.1f}/5")
    print(f"  Avg latency:         {overall['avg_latency_ms']:.0f}ms")
    print(f"  P95 latency:         {overall['p95_latency_ms']:.0f}ms")
    print(f"  Max latency:         {overall['max_latency_ms']:.0f}ms")
    print("=" * 70)

    if "mlflow_run_id" in results:
        print(f"\n  MLflow run ID: {results['mlflow_run_id']}")
        print("  View in UI:    mlflow ui --port 5000")


def main():
    parser = argparse.ArgumentParser(description="Run ME Assistant evaluation")
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow tracking (logs metrics and artifacts)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per question for statistical robustness (default: 3)",
    )
    args = parser.parse_args()

    if args.mlflow:
        logger.info("Starting MLflow-tracked evaluation (runs=%d)...", args.runs)
        results = run_mlflow_evaluation(num_runs=args.runs)
    else:
        logger.info("Starting evaluation (runs=%d)...", args.runs)
        results = run_evaluation(num_runs=args.runs)

    _print_results(results)


if __name__ == "__main__":
    main()
