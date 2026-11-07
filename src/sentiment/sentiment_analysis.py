import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# Configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

INPUT_FILE = Path("data/processed/tweeteval_test.csv")

OUTPUT_DIR = Path("reports/sentiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "sentiment_predictions.csv"


# Label mapping
LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def main():

    print("Loading TweetEval test dataset...")

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["clean_text", "label"])

    # Start with a representative sample to limit CPU usage
    sample_size = min(500, len(df))
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print("Records selected:", len(df))

    print("\nLoading transformer model...")

    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )

    texts = df["clean_text"].astype(str).tolist()

    predictions = []

    print("\nRunning sentiment predictions...")

    for start in tqdm(range(0, len(texts), 16)):

        batch = texts[start:start + 16]

        results = classifier(
            batch,
            truncation=True,
            max_length=128
        )

        predictions.extend(results)

    df["predicted_sentiment"] = [
        result["label"].lower()
        for result in predictions
    ]

    df["confidence"] = [
        result["score"]
        for result in predictions
    ]

    df["actual_sentiment"] = df["label"].map(LABEL_MAP)

    # Save predictions
    df.to_csv(OUTPUT_FILE, index=False)

    # Evaluation
    print("\nSENTIMENT MODEL EVALUATION")

    y_true = df["actual_sentiment"]
    y_pred = df["predicted_sentiment"]

    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nWeighted F1:",
          f1_score(y_true, y_pred, average="weighted"))

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["negative", "neutral", "positive"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["negative", "neutral", "positive"]
        )
    )

    print("\nPredicted sentiment distribution:")

    print(df["predicted_sentiment"].value_counts())

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
