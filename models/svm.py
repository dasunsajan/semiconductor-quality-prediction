from sklearn.svm import SVC

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

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 3. Create SVM classifier
    classifier = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )

    model = create_model_pipeline(
        classifier=classifier,
        k=15
    )

    # 4. Train model
    model.fit(X_train, y_train)

    print("\nSVM training completed!")

    # 5. Make predictions
    y_pred = model.predict(X_test)

    y_scores = model.decision_function(X_test)

    # 6. Evaluate model
    print("\n========== SVM EVALUATION ==========")

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(y_test, y_pred)
    )

    print(
        "Average Precision:",
        average_precision_score(y_test, y_scores, pos_label=1)
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
