"""CLI script: Run evaluation on all test questions with MLflow tracking."""

import argparse
import logging

from me_assistant.eval.evaluate import run_evaluation, run_mlflow_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _print_results(results):
    """Print evaluation results table to stdout."""
    overall = results["overall"]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for r in results["per_question"]:
        status = "PASS" if r["answer_correct"] else "FAIL"
        route_ok = "ok" if r["route_correct"] else "WRONG"
        print(
            f"  Q{r['question_id']:2d} [{status}]  route={r['route']:<10s} "
            f"({route_ok})  {r['latency_ms']:.0f}ms"
        )

    print("-" * 60)
    print(f"  Accuracy:          {overall['pass_count']}/{overall['total']} "
          f"({overall['accuracy']:.0%})")
    print(f"  Routing accuracy:  {overall['routing_accuracy']:.0%}")
    print(f"  Source accuracy:   {overall['source_accuracy']:.0%}")
    print(f"  Avg latency:       {overall['avg_latency_ms']:.0f}ms")
    print(f"  Max latency:       {overall['max_latency_ms']:.0f}ms")
    print("=" * 60)

    if "mlflow_run_id" in results:
        print(f"\n  MLflow run ID: {results['mlflow_run_id']}")
        print("  View in UI:    mlflow ui --port 5000")


def main():
    parser = argparse.ArgumentParser(description="Run ME Assistant evaluation")
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow tracking (logs metrics and artifacts)",
    )
    args = parser.parse_args()

    if args.mlflow:
        logger.info("Starting MLflow-tracked evaluation...")
        results = run_mlflow_evaluation()
    else:
        logger.info("Starting evaluation...")
        results = run_evaluation()

    _print_results(results)


if __name__ == "__main__":
    main()
