
# Group 29 - Nested Cross Validation
# Logistic Regression Hyperparameter Tuning

import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# STEP 1: Load the SECOM dataset
X = pd.read_csv(
    "secom.data",
    sep=r"\s+",
    header=None
)

labels = pd.read_csv(
    "secom_labels.data",
    sep=r"\s+",
    header=None
)

y = labels.iloc[:, 0]

print("Dataset Shape:", X.shape)
print("Fail Records:", (y == 1).sum())

# STEP 2: Create the ML pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        keep_empty_features=True
    )),
    ("feature_filter", VarianceThreshold()),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# STEP 3: Define parameters to test
param_grid = {
    "classifier__C": [0.01, 0.1, 1.0],
    "classifier__class_weight": [None, "balanced"]
}

# STEP 4: Define evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision"
}

# STEP 5: Inner CV - Hyperparameter tuning
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

# STEP 6: Outer CV - Model evaluation
outer_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=123
)

results = cross_validate(
    grid_search,
    X,
    y,
    cv=outer_cv,
    scoring=scoring,
    return_estimator=True,
    n_jobs=1,
    error_score="raise"
)

# STEP 7: Display average performance
print("\n--- Nested CV Results ---")

for metric in scoring:
    values = results["test_" + metric]

    print(
        f"{metric}: "
        f"{values.mean():.4f} "
        f"(+/- {values.std():.4f})"
    )

# STEP 8: Show best parameters for each outer fold
print("\n--- Selected Hyperparameters ---")

rows = []

for fold, search in enumerate(results["estimator"], start=1):
    params = search.best_params_

    print(f"\nOuter Fold {fold}")
    print("Best Parameters:", params)
    print("Inner CV AP:", round(search.best_score_, 4))

    row = {"fold": fold}
    for metric in scoring:
        row[metric] = results["test_" + metric][fold - 1]

    row["best_C"] = params["classifier__C"]
    row["best_class_weight"] = str(
        params["classifier__class_weight"]
    )
    rows.append(row)

# STEP 9: Save evaluation results
Path("results").mkdir(exist_ok=True)

df = pd.DataFrame(rows)

df.to_csv(
    "results/nested_cv_results.csv",
    index=False
)

print("\nResults saved: results/nested_cv_results.csv")
