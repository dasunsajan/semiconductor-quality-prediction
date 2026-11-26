from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

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
    print("\nClass distribution:")
    print(y.value_counts())

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3. Create Logistic Regression model
    classifier = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    model = create_model_pipeline(
        classifier=classifier,
        k=15
    )

    # 4. Train model
    model.fit(X_train, y_train)

    print("\nLogistic Regression training completed!")

    # 5. Make predictions
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Evaluate model
    print("\n========== MODEL EVALUATION ==========")

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
