from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

CV_SUMMARY_PATH = RESULTS_DIR / "cross_validation_summary.csv"


# --------------------------------------------------
# Load cross-validation results
# --------------------------------------------------


def load_cv_results():
    if not CV_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Cross-validation summary not found: {CV_SUMMARY_PATH}"
        )

    df = pd.read_csv(CV_SUMMARY_PATH)

    # Standardize column names for visualization
    df = df.rename(columns={
        "Model": "model",
        "Balanced_Accuracy_mean": "balanced_accuracy",
        "Balanced_Accuracy_std": "balanced_accuracy_std",
        "Fail_Precision_mean": "fail_precision",
        "Fail_Precision_std": "fail_precision_std",
        "Fail_Recall_mean": "fail_recall",
        "Fail_Recall_std": "fail_recall_std",
        "Fail_F1_mean": "fail_f1",
        "Fail_F1_std": "fail_f1_std",
        "Average_Precision_mean": "average_precision",
        "Average_Precision_std": "average_precision_std"
    })

    print("\nRepeated Cross-Validation Results:")
    print(df.to_string(index=False))

    return df



# --------------------------------------------------
# Chart 1: Balanced Accuracy and Average Precision
# --------------------------------------------------

def plot_cv_performance(df):

    metrics = [
        ("balanced_accuracy", "balanced_accuracy_std",
         "Balanced Accuracy"),
        ("average_precision", "average_precision_std",
         "Average Precision"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (metric, std_metric, title) in zip(axes, metrics):

        ax.bar(
            df["model"],
            df[metric],
            yerr=df[std_metric],
            capsize=5
        )

        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "Repeated Stratified Cross-Validation: Model Performance",
        fontsize=14
    )

    plt.tight_layout()

    output_path = RESULTS_DIR / "final_cv_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# Chart 2: Failure-class detection metrics
# --------------------------------------------------

def plot_fail_detection(df):

    metrics = [
        "fail_precision",
        "fail_recall",
        "fail_f1"
    ]

    labels = [
        "Fail Precision",
        "Fail Recall",
        "Fail F1"
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, labels)):

        ax.bar(
            [position + i * width for position in x],
            df[metric],
            width,
            label=label
        )

    ax.set_title("Repeated CV: Failure-Class Detection Performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")

    ax.set_xticks([
        position + width for position in x
    ])

    ax.set_xticklabels(df["model"], rotation=35)

    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = RESULTS_DIR / "final_cv_fail_detection.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Chart 3: Tuned model test-set comparison
# --------------------------------------------------

def plot_tuned_model_performance():

    file_path = RESULTS_DIR / "tuned_model_evaluation.csv"
    df = pd.read_csv(file_path)

    metrics = [
        "Balanced Accuracy",
        "Fail Precision",
        "Fail Recall",
        "Fail F1",
        "Average Precision"
    ]

    labels = [
        "Balanced Accuracy",
        "Fail Precision",
        "Fail Recall",
        "Fail F1",
        "Average Precision"
    ]

    fig, ax = plt.subplots(figsize=(13, 7))

    x = range(len(df))
    width = 0.15

    for i, (metric, label) in enumerate(zip(metrics, labels)):

        ax.bar(
            [position + i * width for position in x],
            df[metric],
            width,
            label=label
        )

    ax.set_title("Tuned Models: Test-Set Performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Machine Learning Model")

    ax.set_xticks([
        position + (width * 2) for position in x
    ])

    ax.set_xticklabels(df["Model"], rotation=20)

    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = RESULTS_DIR / "final_tuned_model_performance.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {output_path}")


# --------------------------------------------------
# Chart 4: Default vs optimized threshold comparison
# --------------------------------------------------

def plot_threshold_optimization():

    file_path = RESULTS_DIR / "threshold_optimization_test_results.csv"
    df = pd.read_csv(file_path)

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        [position - width / 2 for position in x],
        df["Default Fail F1"],
        width,
        label="Default Threshold"
    )

    ax.bar(
        [position + width / 2 for position in x],
        df["Optimized Fail F1"],
        width,
        label="Optimized Threshold"
    )

    ax.set_title("Default vs Optimized Threshold: Fail F1 Comparison")
    ax.set_ylabel("Fail F1 Score")
    ax.set_xlabel("Machine Learning Model")

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=20)

    ax.set_ylim(0, 0.4)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    output_path = RESULTS_DIR / "final_threshold_optimization.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")



# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():

    df = load_cv_results()

    plot_cv_performance(df)
    plot_fail_detection(df)
    plot_threshold_optimization()
    plot_tuned_model_performance()
    print("\nFinal cross-validation visualizations completed!")


if __name__ == "__main__":
    main()
