import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.50):
        self.threshold = threshold

    def fit(self, X, y=None):

        missing_ratio = X.isnull().mean()

        self.selected_columns_ = missing_ratio[
            missing_ratio <= self.threshold
        ].index.tolist()

        if not self.selected_columns_:
            raise ValueError("No features remain after missing-value filtering.")

        return self

    def transform(self, X):

        return X[self.selected_columns_]


def create_preprocessing_pipeline(k=20):

    pipeline = Pipeline([

        (
            "missing_filter",
            MissingValueFilter(threshold=0.50)
        ),

        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),

        (
            "variance_filter",
            VarianceThreshold(threshold=0)
        ),

        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=k
            )
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    return pipeline
