"""Write GradLens results as JSON and, when possible, a matplotlib chart."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _write_csv(results: dict, path: Path) -> None:
    rows = []
    for mode, metrics in results.get("modes", {}).items():
        conflict = metrics.get("gradient_conflict") or {}
        test = metrics.get("test_accuracy") or {}
        rows.append(
            {
                "mode": mode,
                "conflict_frequency": conflict.get("conflict_frequency"),
                "mean_cosine_similarity": conflict.get("mean_cosine_similarity"),
                "task_a_accuracy": test.get("task_a"),
                "task_b_accuracy": test.get("task_b"),
                "mean_accuracy": test.get("mean"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "conflict_frequency",
                "mean_cosine_similarity",
                "task_a_accuracy",
                "task_b_accuracy",
                "mean_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(results: dict, output_dir: str | Path) -> dict[str, str]:
    """Write results.json plus a PNG chart, or a CSV when matplotlib is absent."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    written = {"json": str(json_path)}

    modes = results.get("modes", {})
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        csv_path = output / "results.csv"
        _write_csv(results, csv_path)
        written["csv"] = str(csv_path)
        return written

    labels = list(modes)
    conflict_frequency = []
    for label in labels:
        conflict = modes[label].get("gradient_conflict") or {}
        value = conflict.get("conflict_frequency")
        conflict_frequency.append(0.0 if value is None else value)

    task_a = [modes[label]["test_accuracy"]["task_a"] for label in labels]
    task_b = [modes[label]["test_accuracy"]["task_b"] for label in labels]

    x_positions = range(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, conflict_frequency, color="#7c3aed")
    axes[0].set_title("Gradient-conflict frequency")
    axes[0].set_ylabel("Fraction of steps with cosine < 0")
    axes[0].set_ylim(0, 1)

    axes[1].bar(
        [position - width / 2 for position in x_positions], task_a, width=width, label="Task A"
    )
    axes[1].bar(
        [position + width / 2 for position in x_positions], task_b, width=width, label="Task B"
    )
    axes[1].set_xticks(list(x_positions), labels)
    axes[1].set_title("Held-out task accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    figure.tight_layout()
    chart_path = output / "conflict_chart.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    written["chart"] = str(chart_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate GradLens report outputs.")
    parser.add_argument("--results", type=Path, default=Path("results/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    written = write_outputs(results, args.output_dir)
    print(written)


if __name__ == "__main__":
    main()
