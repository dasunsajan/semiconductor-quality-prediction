from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from models.model_pipeline import load_secom_data, create_model_pipeline


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Model configurations
# --------------------------------------------------

def create_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }


# --------------------------------------------------
# Repeated Stratified Cross-Validation
# --------------------------------------------------

def run_cross_validation():

    # Load dataset
    X, y = load_secom_data()

    # Keep the original 20% test set separate
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training data shape:", X_train.shape)
    print("Training target distribution:")
    print(y_train.value_counts())

    # Define cross-validation strategy
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    models = create_models()

    all_results = []

    # Evaluate each model
    for model_name, classifier in models.items():

        print("\n" + "=" * 60)
        print("Evaluating:", model_name)
        print("=" * 60)

        # Create complete preprocessing + model pipeline
        pipeline = create_model_pipeline(
            classifier=classifier,
            k=15
        )

        # Each model receives 15 fold evaluations
        for fold_number, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train),
            start=1
        ):

            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Create a fresh pipeline for this fold
            fold_model = clone(pipeline)

            # Fit the model using training fold only
            if model_name == "Gradient Boosting":

                sample_weights = compute_sample_weight(
                    class_weight="balanced",
                    y=y_fold_train
                )

                fold_model.fit(
                    X_fold_train,
                    y_fold_train,
                    classifier__sample_weight=sample_weights
                )

            else:

                fold_model.fit(
                    X_fold_train,
                    y_fold_train
                )

            # Predictions
            y_pred = fold_model.predict(X_fold_val)

            # Obtain scores for Average Precision
            if hasattr(fold_model, "predict_proba"):

                positive_class_index = list(
                    fold_model.classes_
                ).index(1)

                y_scores = fold_model.predict_proba(
                    X_fold_val
                )[:, positive_class_index]

            else:

                y_scores = fold_model.decision_function(
                    X_fold_val
                )

            # Calculate evaluation metrics
            metrics = {
                "Model": model_name,
                "Fold": fold_number,

                "Accuracy": accuracy_score(
                    y_fold_val, y_pred
                ),

                "Balanced_Accuracy": balanced_accuracy_score(
                    y_fold_val, y_pred
                ),

                "Fail_Precision": precision_score(
                    y_fold_val,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                ),

                "Fail_Recall": recall_score(
                    y_fold_val,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                ),

                "Fail_F1": f1_score(
                    y_fold_val,
                    y_pred,
                    pos_label=1,
                    zero_division=0
                ),

                "Average_Precision": average_precision_score(
                    y_fold_val,
                    y_scores,
                    pos_label=1
                )
            }

            all_results.append(metrics)

            print(
                f"Fold {fold_number:02d}/15 | "
                f"Balanced Accuracy: "
                f"{metrics['Balanced_Accuracy']:.4f} | "
                f"Fail Recall: "
                f"{metrics['Fail_Recall']:.4f} | "
                f"Average Precision: "
                f"{metrics['Average_Precision']:.4f}"
            )

    # --------------------------------------------------
    # Save detailed fold results
    # --------------------------------------------------

    results_df = pd.DataFrame(all_results)

    detailed_path = RESULTS_DIR / "cross_validation_results.csv"

    results_df.to_csv(
        detailed_path,
        index=False
    )

    # --------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Fail_Precision",
        "Fail_Recall",
        "Fail_F1",
        "Average_Precision"
    ]

    summary_df = results_df.groupby("Model")[
        metric_columns
    ].agg(["mean", "std"])

    summary_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary_df.columns
    ]

    summary_df = summary_df.reset_index()

    summary_path = RESULTS_DIR / "cross_validation_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------
    # Display final comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 70)

    print(
        summary_df.round(4).to_string(index=False)
    )

    print("\nDetailed results saved to:", detailed_path)
    print("Summary results saved to:", summary_path)

    print("\nTotal model evaluations:", len(results_df))


if __name__ == "__main__":
    run_cross_validation()
