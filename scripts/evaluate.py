"""CLI script: Run evaluation on all test questions."""

import json
import logging

from me_assistant.eval.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting evaluation...")
    results = run_evaluation()

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


if __name__ == "__main__":
    main()
