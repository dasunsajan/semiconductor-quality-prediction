from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    average_precision_score
)

from models.model_pipeline import (
    load_secom_data,
    create_model_pipeline
)


def main():

    # 1. Load dataset
    X, y = load_secom_data()

    print("Dataset shape:", X.shape)

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3. Create Gradient Boosting classifier
    classifier = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )

    model = create_model_pipeline(
        classifier=classifier,
        k=15
    )

    # 4. Calculate balanced sample weights
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    # 5. Train model
    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights
    )

    print("\nGradient Boosting training completed!")

    # 6. Make predictions
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 7. Evaluate model
    print("\n========== GRADIENT BOOSTING EVALUATION ==========")

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(y_test, y_pred)
    )

    print(
        "Average Precision:",
        average_precision_score(y_test, y_prob)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[-1, 1]))

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


if __name__ == "__main__":
    main()
