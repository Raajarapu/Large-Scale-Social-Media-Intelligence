import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/misinformation")
REPORT_DIR = Path("reports/misinformation")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(filename):

    df = pd.read_csv(DATA_DIR / filename)

    df = df.dropna(subset=["clean_text", "label"])

    return df


def main():

    print("Loading LIAR datasets...")

    train_df = load_dataset("liar_train.csv")
    valid_df = load_dataset("liar_valid.csv")
    test_df = load_dataset("liar_test.csv")

    print("Training records:", len(train_df))
    print("Validation records:", len(valid_df))
    print("Testing records:", len(test_df))

    X_train = train_df["clean_text"]
    y_train = train_df["label"]

    X_valid = valid_df["clean_text"]
    y_valid = valid_df["label"]

    X_test = test_df["clean_text"]
    y_test = test_df["label"]

    print("\nTraining TF-IDF + Logistic Regression...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                stop_words="english",
                min_df=2
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    model.fit(X_train, y_train)

    print("\nValidation evaluation...")

    valid_predictions = model.predict(X_valid)

    print(
        "Validation accuracy:",
        accuracy_score(y_valid, valid_predictions)
    )

    print(
        "Validation weighted F1:",
        f1_score(
            y_valid,
            valid_predictions,
            average="weighted"
        )
    )

    print("\nFinal test evaluation...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\nTest Accuracy:", accuracy)
    print("Test Weighted F1:", weighted_f1)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(
        model,
        MODEL_DIR / "liar_classifier.pkl"
    )

    with open(
        REPORT_DIR / "classification_report.txt",
        "w"
    ) as file:
        file.write(report)

    metrics = pd.DataFrame([{
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "training_records": len(train_df),
        "validation_records": len(valid_df),
        "testing_records": len(test_df)
    }])

    metrics.to_csv(
        REPORT_DIR / "model_metrics.csv",
        index=False
    )

    print("\nModel saved successfully.")
    print("Metrics saved successfully.")


if __name__ == "__main__":
    main()
