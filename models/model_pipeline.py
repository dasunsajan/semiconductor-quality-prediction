from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline

from preprocessing.preprocessing import (
    create_preprocessing_pipeline
)


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
DATA_PATH = PROJECT_ROOT / "data" / "secom.data"
LABEL_PATH = PROJECT_ROOT / "data" / "secom_labels.data"


def load_secom_data():
    """
    Load the SECOM dataset and corresponding labels.

    Returns:
        X: Feature DataFrame
        y: Target Series
    """

    X = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None
    )

    labels = pd.read_csv(
        LABEL_PATH,
        sep=r"\s+",
        header=None
    )

    y = labels.iloc[:, 0]

    if len(X) != len(y):
        raise ValueError(
            "Feature and target datasets have different lengths."
        )

    return X, y


def create_model_pipeline(classifier, k=15):
    """
    Create a complete machine learning pipeline.

    Args:
        classifier: Scikit-learn classification algorithm.
        k: Number of features to select.

    Returns:
        Complete preprocessing and classification pipeline.
    """

    pipeline = Pipeline([
        (
            "preprocessing",
            create_preprocessing_pipeline(k=k)
        ),
        (
            "classifier",
            classifier
        )
    ])

    return pipeline
