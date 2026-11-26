from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_csv(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def main():

    # 1. Repeated cross-validation
    cv = load_csv("cross_validation_summary.csv")
    rows = []

    for _, r in cv.iterrows():
        rows.append({
            "Experiment": "Repeated Cross-Validation",
            "Evaluation_Method": "5-fold CV x 3 repeats on training set",
            "Model": r["Model"],
            "Accuracy": r["Accuracy_mean"],
            "Accuracy_Std": r["Accuracy_std"],
            "Balanced_Accuracy": r["Balanced_Accuracy_mean"],
            "Balanced_Accuracy_Std": r["Balanced_Accuracy_std"],
            "Fail_Precision": r["Fail_Precision_mean"],
            "Fail_Precision_Std": r["Fail_Precision_std"],
            "Fail_Recall": r["Fail_Recall_mean"],
            "Fail_Recall_Std": r["Fail_Recall_std"],
            "Fail_F1": r["Fail_F1_mean"],
            "Fail_F1_Std": r["Fail_F1_std"],
            "Average_Precision": r["Average_Precision_mean"],
            "Average_Precision_Std": r["Average_Precision_std"],
        })

    # 2. Nested cross-validation
    nested = load_csv("nested_cv_summary.csv")
    nested_metrics = {
        r["metric"]: r for _, r in nested.iterrows()
    }

    nested_row = {
        "Experiment": "Nested Cross-Validation",
        "Evaluation_Method": "5 outer folds x 3 inner folds",
        "Model": "Logistic Regression",
    }

    mapping = {
        "accuracy": "Accuracy",
        "precision": "Fail_Precision",
        "recall": "Fail_Recall",
        "f1": "Fail_F1",
        "balanced_accuracy": "Balanced_Accuracy",
        "average_precision": "Average_Precision",
    }

    for source, target in mapping.items():
        nested_row[target] = nested_metrics[source]["mean"]
        nested_row[f"{target}_Std"] = nested_metrics[source]["std"]

    rows.append(nested_row)

    # 3. Hyperparameter tuning
    tuning = load_csv("hyperparameter_tuning_summary.csv")
    tuning_lookup = {
        r["Model"]: r for _, r in tuning.iterrows()
    }

    # 4. Tuned model test evaluation
    tuned = load_csv("tuned_model_evaluation.csv")

    for _, r in tuned.iterrows():
        info = tuning_lookup.get(r["Model"])

        rows.append({
            "Experiment": "Tuned Model Evaluation",
            "Evaluation_Method": "80:20 stratified test split",
            "Model": r["Model"],
            "Accuracy": r["Accuracy"],
            "Balanced_Accuracy": r["Balanced Accuracy"],
            "Fail_Precision": r["Fail Precision"],
            "Fail_Recall": r["Fail Recall"],
            "Fail_F1": r["Fail F1"],
            "Average_Precision": r["Average Precision"],
            "Tuning_CV_Average_Precision": (
                info["Best_Average_Precision"]
                if info is not None else None
            ),
            "Best_Parameters": (
                info["Best_Parameters"]
                if info is not None else None
            ),
        })

    # 5. Threshold optimization
    threshold = load_csv(
        "threshold_optimization_test_results.csv"
    )

    for _, r in threshold.iterrows():
        rows.append({
            "Experiment": "Threshold Optimization",
            "Evaluation_Method": "OOF threshold selection + test evaluation",
            "Model": r["Model"],
            "Accuracy": r["Optimized Accuracy"],
            "Balanced_Accuracy": r["Optimized Balanced Accuracy"],
            "Fail_Precision": r["Optimized Fail Precision"],
            "Fail_Recall": r["Optimized Fail Recall"],
            "Fail_F1": r["Optimized Fail F1"],
            "Average_Precision": r["Optimized Average Precision"],
            "Default_Threshold": r["Default Threshold"],
            "Optimized_Threshold": r["Optimized Threshold"],
        })

    # 6. Save consolidated summary
    final_summary = pd.DataFrame(rows)

    output_path = RESULTS_DIR / "final_experiment_summary.csv"
    final_summary.to_csv(output_path, index=False)

    print("\nFinal experiment summary created!")
    print("Saved to:", output_path)

    print("\nExperiment counts:")
    print(final_summary["Experiment"].value_counts())

    print("\nFinal summary:")
    print(final_summary[
        [
            "Experiment",
            "Model",
            "Accuracy",
            "Balanced_Accuracy",
            "Fail_Precision",
            "Fail_Recall",
            "Fail_F1",
            "Average_Precision",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
