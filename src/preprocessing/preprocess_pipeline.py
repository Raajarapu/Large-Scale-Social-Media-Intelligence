import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.language_detector import detect_language


BASE_DIR = Path("data/external")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. TWEETEVAL PREPROCESSING
# --------------------------------------------------

def process_tweeteval():

    print("\nProcessing TweetEval...")

    sentiment_dir = (
        BASE_DIR / "tweeteval/tweeteval-main/datasets/sentiment"
    )

    for split in ["train", "val", "test"]:

        texts = pd.read_csv(
            sentiment_dir / f"{split}_text.txt",
            header=None,
            names=["text"],
            sep="\t"
        )

        labels = pd.read_csv(
            sentiment_dir / f"{split}_labels.txt",
            header=None,
            names=["label"]
        )

        df = pd.concat([texts, labels], axis=1)

        df["clean_text"] = df["text"].apply(clean_text)

        df["language"] = df["clean_text"].apply(detect_language)

        df = df.dropna(subset=["clean_text", "label"])

        df = df.drop_duplicates(subset=["clean_text"])

        output = OUTPUT_DIR / f"tweeteval_{split}.csv"

        df.to_csv(output, index=False)

        print(f"{split}: {len(df)} records saved")


# --------------------------------------------------
# 2. LIAR PREPROCESSING
# --------------------------------------------------

def process_liar():

    print("\nProcessing LIAR...")

    columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "barely_true",
        "false",
        "half_true",
        "mostly_true",
        "pants_fire",
        "context"
    ]

    for split in ["train", "valid", "test"]:

        file_path = BASE_DIR / f"liar/{split}.tsv"

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=columns
        )

        df["clean_text"] = df["statement"].apply(clean_text)

        df["language"] = df["clean_text"].apply(detect_language)

        df = df.dropna(subset=["clean_text", "label"])

        df = df.drop_duplicates(subset=["id"])

        output = OUTPUT_DIR / f"liar_{split}.csv"

        df.to_csv(output, index=False)

        print(f"{split}: {len(df)} records saved")


# --------------------------------------------------
# 3. REDDIT PREPROCESSING
# --------------------------------------------------

def process_reddit():

    print("\nProcessing Reddit Hyperlink Network...")

    file_path = (
        BASE_DIR / "reddit/soc-redditHyperlinks-body.tsv"
    )

    output = OUTPUT_DIR / "reddit_clean.csv"

    chunk_size = 50000

    first_chunk = True

    total_records = 0

    for chunk in tqdm(
        pd.read_csv(
            file_path,
            sep="\t",
            chunksize=chunk_size,
            on_bad_lines="skip"
        )
    ):

        text_column = "PROPERTIES"

        if text_column not in chunk.columns:
            raise ValueError(
                f"Expected column {text_column} not found. "
                f"Available columns: {list(chunk.columns)}"
            )

        chunk["clean_text"] = (
            chunk[text_column].fillna("").apply(clean_text)
        )

        chunk = chunk[
            chunk["clean_text"].str.len() > 0
        ]

        chunk.to_csv(
            output,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_records += len(chunk)

    print(f"Reddit: {total_records} records processed")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    process_tweeteval()

    process_liar()

    process_reddit()

    print("\nPreprocessing completed successfully.")
