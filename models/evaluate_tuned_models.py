from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from models.model_pipeline import load_secom_data

# Import the custom classifier so joblib can load saved models.
from models.hyperparameter_tuning import BalancedGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = RESULTS_DIR / "tuned_models"


def evaluate_tuned_models():

    # 1. Load the dataset
    X, y = load_secom_data()

    # 2. Recreate the original train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nTest class distribution:")
    print(y_test.value_counts())

    # 3. Define saved models
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "SVM": "svm.joblib",
        "Gradient Boosting": "gradient_boosting.joblib"
    }

    evaluation_results = []
    confusion_matrices = []

    # 4. Evaluate each tuned model
    for model_name, filename in model_files.items():

        print("\n" + "=" * 65)
        print("Evaluating:", model_name)
        print("=" * 65)

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        model = joblib.load(model_path)

        # Generate predictions
        y_pred = model.predict(X_test)

        # Obtain continuous scores for Average Precision
        classifier = model.named_steps["classifier"]

        if hasattr(classifier, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            y_pred
        )

        fail_precision = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        fail_recall = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        fail_f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        average_precision = average_precision_score(
            y_test,
            y_scores
        )

        # Confusion matrix: rows = actual, columns = predicted
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[-1, 1]
        )

        # Store metrics
        evaluation_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "Fail Precision": fail_precision,
            "Fail Recall": fail_recall,
            "Fail F1": fail_f1,
            "Average Precision": average_precision
        })

        confusion_matrices.append({
            "Model": model_name,
            "True Pass": cm[0, 0],
            "False Fail": cm[0, 1],
            "Missed Fail": cm[1, 0],
            "Correct Fail": cm[1, 1]
        })

        # Print results
        print("\nEvaluation Metrics:")
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
        print(f"Fail Precision:     {fail_precision:.4f}")
        print(f"Fail Recall:        {fail_recall:.4f}")
        print(f"Fail F1-score:      {fail_f1:.4f}")
        print(f"Average Precision:  {average_precision:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[-1, 1],
                target_names=["Pass", "Fail"],
                zero_division=0
            )
        )

    # 5. Save evaluation results
    results_df = pd.DataFrame(evaluation_results)
    confusion_df = pd.DataFrame(confusion_matrices)

    results_path = RESULTS_DIR / "tuned_model_evaluation.csv"
    confusion_path = RESULTS_DIR / "tuned_model_confusion_matrices.csv"

    results_df.to_csv(results_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    # 6. Display comparison
    print("\n" + "=" * 75)
    print("TUNED MODEL EVALUATION SUMMARY")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nConfusion Matrix Summary:")
    print(confusion_df.to_string(index=False))

    print("\nResults saved:")
    print(results_path)
    print(confusion_path)


if __name__ == "__main__":
    evaluate_tuned_models()
