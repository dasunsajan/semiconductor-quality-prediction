from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

SUMMARY_PATH = RESULTS_DIR / "cross_validation_summary.csv"


# --------------------------------------------------
# Load cross-validation summary
# --------------------------------------------------

def load_results():

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {SUMMARY_PATH}"
        )

    results = pd.read_csv(SUMMARY_PATH)

    print("Loaded cross-validation summary:")
    print(results.head())

    return results


# --------------------------------------------------
# Chart 1: Balanced Accuracy
# --------------------------------------------------

def plot_balanced_accuracy(results):

    models = results["Model"]
    means = results["Balanced_Accuracy_mean"]
    stds = results["Balanced_Accuracy_std"]

    plt.figure(figsize=(10, 6))

    plt.bar(
        models,
        means,
        yerr=stds,
        capsize=5
    )

    plt.title("Model Comparison - Balanced Accuracy")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Balanced Accuracy")

    plt.ylim(0, 0.25)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = RESULTS_DIR / "cv_balanced_accuracy.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


# --------------------------------------------------
# Chart 2: Fail Class Metrics
# --------------------------------------------------

def plot_fail_metrics(results):

    models = results["Model"].tolist()

    metrics = {
        "Precision": (
            "Fail_Precision_mean",
            "Fail_Precision_std"
        ),
        "Recall": (
            "Fail_Recall_mean",
            "Fail_Recall_std"
        ),
        "F1-score": (
            "Fail_F1_mean",
            "Fail_F1_std"
        )
    }

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(12, 6))

    for i, (metric_name, columns) in enumerate(metrics.items()):

        mean_column, std_column = columns

        means = results[mean_column]
        stds = results[std_column]

        plt.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=metric_name
        )

    plt.title("Fail Class Performance Comparison")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Score")

    plt.xticks(
        x,
        models,
        rotation=20,
        ha="right"
    )

    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = RESULTS_DIR / "cv_fail_class_metrics.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


# --------------------------------------------------
# Chart 3: Average Precision
# --------------------------------------------------

def plot_average_precision(results):

    models = results["Model"]
    means = results["Average_Precision_mean"]
    stds = results["Average_Precision_std"]

    plt.figure(figsize=(10, 6))

    plt.bar(
        models,
        means,
        yerr=stds,
        capsize=5
    )

    plt.title("Model Comparison - Average Precision")
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Average Precision")

    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = RESULTS_DIR / "cv_average_precision.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


# --------------------------------------------------
# Main function
# --------------------------------------------------

def main():

    results = load_results()

    plot_balanced_accuracy(results)

    plot_fail_metrics(results)

    plot_average_precision(results)

    print("\nAll cross-validation charts generated successfully!")


if __name__ == "__main__":
    main()
