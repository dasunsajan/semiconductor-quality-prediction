
# Group 29 - Stratified Cross Validation
# Logistic Regression Model Comparison

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# STEP 1: Load SECOM dataset
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

# STEP 2: Create Stratified 5-Fold CV
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# STEP 3: Define evaluation metrics
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision"
}

# STEP 4: Compare both models
for model_name, weight in [
    ("Baseline", None),
    ("Class Weighted", "balanced")
]:

    # Build a complete preprocessing pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("feature_filter", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight=weight
        ))
    ])

    # STEP 5: Run cross-validation
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise"
    )

    print("\n---------------------------")
    print("Model:", model_name)
    print("---------------------------")

    # STEP 6: Display mean and std
    for metric in scoring:
        values = scores["test_" + metric]

        print(
            f"{metric}: "
            f"{values.mean():.4f} "
            f"(+/- {values.std():.4f})"
        )
