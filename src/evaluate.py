"""Evaluation entry point for final reported metrics."""

import json
from pathlib import Path


def main():
    metrics_path = Path("results/recommended_test_metrics.json")

    if not metrics_path.exists():
        raise FileNotFoundError("Missing results/recommended_test_metrics.json")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    print("Final test metrics")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
