import pandas as pd
from pathlib import Path

INPUT_FILE = Path("reports/sentiment/sentiment_predictions.csv")

OUTPUT_FILE = Path("reports/sentiment/sentiment_distribution.csv")


def main():

    df = pd.read_csv(INPUT_FILE)

    distribution = (
        df["predicted_sentiment"]
        .value_counts()
        .reset_index()
    )

    distribution.columns = [
        "sentiment",
        "record_count"
    ]

    distribution["percentage"] = (
        distribution["record_count"]
        / distribution["record_count"].sum()
    ) * 100

    distribution.to_csv(OUTPUT_FILE, index=False)

    print(distribution)

    print("\nSentiment distribution saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
