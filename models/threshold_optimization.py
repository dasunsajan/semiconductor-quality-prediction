from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from models.model_pipeline import load_secom_data
from models.hyperparameter_tuning import BalancedGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "tuned_models"


def get_positive_scores(model, X, method):

    if method == "decision_function":
        return model.decision_function(X)

    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)

    return probabilities[:, positive_index]


def find_best_threshold(y_true, scores):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
        pos_label=1
    )

    # The last precision and recall values do not have
    # a corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    best_index = np.argmax(f1_scores)

    return (
        thresholds[best_index],
        precision[best_index],
        recall[best_index],
        f1_scores[best_index]
    )


def evaluate_at_threshold(y_true, scores, threshold):

    y_pred = np.where(scores >= threshold, 1, -1)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "Fail Precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Fail Recall": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Fail F1": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "Average Precision": average_precision_score(
            y_true, scores
        ),
        "Confusion Matrix": confusion_matrix(
            y_true, y_pred, labels=[-1, 1]
        ).tolist()
    }


def run_threshold_optimization():

    # 1. Load dataset and recreate original split
    X, y = load_secom_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    # 2. Define models and scoring methods
    model_files = {
        "Logistic Regression": (
            "logistic_regression.joblib",
            "predict_proba"
        ),
        "SVM": (
            "svm.joblib",
            "decision_function"
        ),
        "Gradient Boosting": (
            "gradient_boosting.joblib",
            "predict_proba"
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    threshold_results = []
    test_results = []

    # 3. Process each tuned model
    for model_name, (filename, method) in model_files.items():

        print("\n" + "=" * 65)
        print("Threshold optimization:", model_name)
        print("=" * 65)

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        best_model = joblib.load(model_path)

        # 4. Generate out-of-fold scores on training data
        oof_scores = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=cv,
            method=method,
            n_jobs=-1
        )

        # Extract positive-class probabilities when necessary
        if method == "predict_proba":
            positive_index = list(best_model.classes_).index(1)
            oof_scores = oof_scores[:, positive_index]

        # 5. Find optimal threshold using training OOF scores
        best_threshold, oof_precision, oof_recall, oof_f1 = (
            find_best_threshold(y_train, oof_scores)
        )

        print("\nOptimal threshold:", round(best_threshold, 6))
        print("OOF Fail Precision:", round(oof_precision, 4))
        print("OOF Fail Recall:", round(oof_recall, 4))
        print("OOF Fail F1:", round(oof_f1, 4))

        # 6. Get test scores from the saved tuned model
        test_scores = get_positive_scores(
            best_model,
            X_test,
            method
        )

        # 7. Evaluate the test set using the optimized threshold
        optimized_metrics = evaluate_at_threshold(
            y_test,
            test_scores,
            best_threshold
        )

        # Also calculate metrics using the model's default threshold
        default_threshold = 0.5 if method == "predict_proba" else 0.0

        default_metrics = evaluate_at_threshold(
            y_test,
            test_scores,
            default_threshold
        )

        print("\nDefault threshold:", default_threshold)
        print("Default Fail F1:", round(default_metrics["Fail F1"], 4))

        print("\nOptimized threshold:", round(best_threshold, 6))
        print("Optimized test metrics:")

        for metric, value in optimized_metrics.items():
            if metric != "Confusion Matrix":
                print(f"{metric}: {value:.4f}")

        print("\nOptimized Confusion Matrix:")
        print(optimized_metrics["Confusion Matrix"])

        threshold_results.append({
            "Model": model_name,
            "Default Threshold": default_threshold,
            "Optimized Threshold": best_threshold,
            "OOF Fail Precision": oof_precision,
            "OOF Fail Recall": oof_recall,
            "OOF Fail F1": oof_f1
        })

        test_results.append({
            "Model": model_name,
            "Default Threshold": default_threshold,
            "Default Fail F1": default_metrics["Fail F1"],
            "Optimized Threshold": best_threshold,
            "Optimized Accuracy": optimized_metrics["Accuracy"],
            "Optimized Balanced Accuracy": optimized_metrics["Balanced Accuracy"],
            "Optimized Fail Precision": optimized_metrics["Fail Precision"],
            "Optimized Fail Recall": optimized_metrics["Fail Recall"],
            "Optimized Fail F1": optimized_metrics["Fail F1"],
            "Optimized Average Precision": optimized_metrics["Average Precision"],
            "Optimized Confusion Matrix": str(
                optimized_metrics["Confusion Matrix"]
            )
        })

    # 8. Save results
    threshold_df = pd.DataFrame(threshold_results)
    test_df = pd.DataFrame(test_results)

    threshold_path = RESULTS_DIR / "optimal_thresholds.csv"
    test_path = RESULTS_DIR / "threshold_optimization_test_results.csv"

    threshold_df.to_csv(threshold_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 75)
    print("THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 75)

    print(threshold_df.to_string(index=False))

    print("\nTEST SET COMPARISON")
    print("=" * 75)

    print(
        test_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nResults saved:")
    print(threshold_path)
    print(test_path)


if __name__ == "__main__":
    run_threshold_optimization()
