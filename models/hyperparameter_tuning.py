from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import loguniform, randint, uniform

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from models.model_pipeline import (
    load_secom_data,
    create_model_pipeline
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Balanced Gradient Boosting wrapper
# --------------------------------------------------

class BalancedGradientBoostingClassifier(
    ClassifierMixin,
    BaseEstimator
):

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=1.0,
        min_samples_leaf=1,
        random_state=42
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y
        )

        self.estimator_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

        self.estimator_.fit(
            X,
            y,
            sample_weight=sample_weights
        )

        self.classes_ = self.estimator_.classes_

        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# --------------------------------------------------
# Hyperparameter search spaces
# --------------------------------------------------

def create_search_spaces():

    return {

        "Logistic Regression": (
            LogisticRegression(
                max_iter=5000,
                random_state=42
            ),

            {
                "classifier__C": loguniform(0.001, 100),
                "classifier__class_weight": [
                    None,
                    "balanced"
                ]
            },

            20
        ),

        "SVM": (
            SVC(
                random_state=42
            ),

            {
                "classifier__C": loguniform(0.01, 100),
                "classifier__kernel": [
                    "linear",
                    "rbf"
                ],
                "classifier__gamma": [
                    "scale",
                    "auto"
                ],
                "classifier__class_weight": [
                    None,
                    "balanced"
                ]
            },

            25
        ),

        "Gradient Boosting": (
            BalancedGradientBoostingClassifier(
                random_state=42
            ),

            {
                "classifier__n_estimators": randint(50, 301),
                "classifier__learning_rate": loguniform(
                    0.01, 0.2
                ),
                "classifier__max_depth": randint(1, 5),
                "classifier__subsample": uniform(
                    0.6, 0.4
                ),
                "classifier__min_samples_leaf": randint(
                    1, 11
                )
            },

            25
        )
    }


# --------------------------------------------------
# Hyperparameter optimization
# --------------------------------------------------

def run_hyperparameter_tuning():

    # Load dataset
    X, y = load_secom_data()

    # Preserve the existing test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training data:", X_train.shape)
    print("Training class distribution:")
    print(y_train.value_counts())

    # Stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search_spaces = create_search_spaces()

    best_models = {}
    tuning_results = []

    for model_name, (
        classifier,
        parameter_space,
        n_iter
    ) in search_spaces.items():

        print("\n" + "=" * 65)
        print("Hyperparameter tuning:", model_name)
        print("=" * 65)

        pipeline = create_model_pipeline(
            classifier=classifier,
            k=15
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_space,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=1,
            refit=True,
            error_score="raise"
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        best_models[model_name] = best_model

        print("\nBest parameters:")
        print(search.best_params_)

        print(
            "\nBest cross-validation Average Precision:",
            round(search.best_score_, 4)
        )

        tuning_results.append({
            "Model": model_name,
            "Best_Average_Precision": search.best_score_,
            "Best_Parameters": str(search.best_params_)
        })

    # Save tuning summary
    results_df = pd.DataFrame(tuning_results)

    output_path = (
        RESULTS_DIR / "hyperparameter_tuning_summary.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 65)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nResults saved to:", output_path)

    # Save the best estimators for the next stage
    # They are fitted on the complete training split by refit=True.
    import joblib

    model_directory = RESULTS_DIR / "tuned_models"
    model_directory.mkdir(exist_ok=True)

    for model_name, model in best_models.items():

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            model_directory / filename
        )

        print(
            "Saved tuned model:",
            model_directory / filename
        )


if __name__ == "__main__":
    run_hyperparameter_tuning()
