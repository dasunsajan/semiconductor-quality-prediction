
# Group 29 - Logistic Regression Baseline Model

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Load the dataset
X = pd.read_csv("secom.data", sep=r"\s+", header=None)
labels = pd.read_csv(
    "secom_labels.data",
    sep=r"\s+",
    header=None
)

y = labels.iloc[:, 0]

print("Dataset Shape:", X.shape)
print("Class Distribution:")
print(y.value_counts())

# STEP 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# STEP 3: Create the ML pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("feature_filter", VarianceThreshold()),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
))
])

# STEP 4: Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# STEP 5: Predict using unseen testing data
y_pred = model.predict(X_test)

# Probability of the Fail class (+1)
y_prob = model.predict_proba(X_test)[:, 1]

# STEP 6: Evaluate the model
print("\n--- Logistic Regression Results ---")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision (Fail):",
      precision_score(y_test, y_pred,
                      pos_label=1, zero_division=0))

print("Recall (Fail):",
      recall_score(y_test, y_pred,
                   pos_label=1, zero_division=0))

print("F1 Score (Fail):",
      f1_score(y_test, y_pred,
               pos_label=1, zero_division=0))

print("Balanced Accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print("Average Precision (PR-AUC summary):",
      average_precision_score(y_test, y_prob,
                              pos_label=1))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[-1, 1],
    target_names=["Pass", "Fail"],
    zero_division=0
))
