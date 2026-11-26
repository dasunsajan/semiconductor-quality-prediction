# Nested Cross-Validation for Logistic Regression

from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, precision_score

from models.model_pipeline import load_secom_data
from preprocessing.preprocessing import create_preprocessing_pipeline


# --------------------------------------------------
# STEP 1: Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

N_FEATURES = 15


# --------------------------------------------------
# STEP 2: Load dataset
# --------------------------------------------------

X, y = load_secom_data()

print("Dataset shape:", X.shape)
print("Pass records:", (y == -1).sum())
print("Fail records:", (y == 1).sum())


# --------------------------------------------------
# STEP 3: Create complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline([
    (
        "preprocessing",
        create_preprocessing_pipeline(k=N_FEATURES)
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# --------------------------------------------------
# STEP 4: Define hyperparameter search space
# --------------------------------------------------

param_grid = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0],
    "classifier__class_weight": [None, "balanced"]
}


# --------------------------------------------------
# STEP 5: Define evaluation metrics
# --------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),
    "recall": "recall",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision"
}


# --------------------------------------------------
# STEP 6: Inner cross-validation
# Hyperparameter tuning
# --------------------------------------------------

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="average_precision",
    cv=inner_cv,
    n_jobs=1,
    error_score="raise"
)


# --------------------------------------------------
# STEP 7: Outer cross-validation
# Model performance evaluation
# --------------------------------------------------

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

print("\nStarting nested cross-validation...")
print("Outer folds: 5")
print("Inner folds: 3")
print("Hyperparameter combinations:", 8)

results = cross_validate(
    estimator=grid_search,
    X=X,
    y=y,
    cv=outer_cv,
    scoring=scoring,
    return_estimator=True,
    n_jobs=1,
    error_score="raise"
)


# --------------------------------------------------
# STEP 8: Display overall performance
# --------------------------------------------------

print("\n========== NESTED CV RESULTS ==========")

summary_rows = []

for metric in scoring:
    values = results[f"test_{metric}"]

    mean_value = values.mean()
    std_value = values.std()

    print(
        f"{metric}: "
        f"{mean_value:.4f} (+/- {std_value:.4f})"
    )

    summary_rows.append({
        "metric": metric,
        "mean": mean_value,
        "std": std_value
    })


# --------------------------------------------------
# STEP 9: Extract fold results and parameters
# --------------------------------------------------

print("\n========== BEST PARAMETERS ==========")

rows = []

for fold, search in enumerate(
    results["estimator"],
    start=1
):
    best_params = search.best_params_

    print(f"\nOuter Fold {fold}")
    print("Best parameters:", best_params)
    print(
        "Inner CV Average Precision:",
        round(search.best_score_, 4)
    )

    row = {"fold": fold}

    for metric in scoring:
        row[metric] = results[
            f"test_{metric}"
        ][fold - 1]

    row["best_C"] = best_params[
        "classifier__C"
    ]

    row["best_class_weight"] = str(
        best_params["classifier__class_weight"]
    )

    row["inner_cv_average_precision"] = (
        search.best_score_
    )

    rows.append(row)


# --------------------------------------------------
# STEP 10: Save results
# --------------------------------------------------

fold_results = pd.DataFrame(rows)

fold_results.to_csv(
    RESULTS_DIR / "nested_cv_results.csv",
    index=False
)

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    RESULTS_DIR / "nested_cv_summary.csv",
    index=False
)

print("\nResults saved successfully:")
print(" - results/nested_cv_results.csv")
print(" - results/nested_cv_summary.csv")

print("\nNested cross-validation completed!")
