import pandas as pd
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm

MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

INPUT_FILE = Path("data/processed/tweeteval_test.csv")
OUTPUT_FILE = Path("reports/sentiment/multilingual_predictions.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def main():

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["clean_text"])

    df = df.sample(
        n=min(500, len(df)),
        random_state=42
    ).reset_index(drop=True)

    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=-1
    )

    predictions = []

    texts = df["clean_text"].astype(str).tolist()

    for start in tqdm(range(0, len(texts), 16)):

        batch = texts[start:start + 16]

        results = classifier(
            batch,
            truncation=True,
            max_length=128
        )

        predictions.extend(results)

    df["predicted_sentiment"] = [
        item["label"] for item in predictions
    ]

    df["confidence"] = [
        item["score"] for item in predictions
    ]

    df.to_csv(OUTPUT_FILE, index=False)

    print("Multilingual sentiment analysis completed.")
    print("Records processed:", len(df))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
