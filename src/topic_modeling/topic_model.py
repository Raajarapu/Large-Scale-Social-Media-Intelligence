import pandas as pd

from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


INPUT_FILE = Path("data/processed/tweeteval_train.csv")

OUTPUT_DIR = Path("reports/topics")
MODEL_DIR = Path("models/topics")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["clean_text"])

    documents = df["clean_text"].astype(str).tolist()

    # Use a manageable sample for Codespaces
    documents = documents[:3000]

    print("Documents selected:", len(documents))

    print("Loading sentence transformer...")

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Initializing BERTopic...")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        language="english",
        min_topic_size=20,
        calculate_probabilities=False,
        verbose=True
    )

    print("Training topic model...")

    topics, probabilities = topic_model.fit_transform(documents)

    # Save topic assignments
    df = df.iloc[:len(documents)].copy()
    df["topic_id"] = topics

    df.to_csv(
        OUTPUT_DIR / "topic_assignments.csv",
        index=False
    )

    # Save topic information
    topic_info = topic_model.get_topic_info()

    topic_info.to_csv(
        OUTPUT_DIR / "topic_summary.csv",
        index=False
    )

    # Save model
    topic_model.save(
        str(MODEL_DIR / "bertopic_model"),
        serialization="safetensors",
        save_ctfidf=True
    )

    print("\nTopic modelling completed.")

    print("\nDiscovered topics:")

    print(topic_info.head(15).to_string(index=False))

    print("\nOutputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
