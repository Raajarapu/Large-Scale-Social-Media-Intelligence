import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

INDEX_NAME = "social-media-intelligence"

es = Elasticsearch(
    os.getenv("ELASTICSEARCH_URL"),
    api_key=os.getenv("ELASTICSEARCH_API_KEY"),
    request_timeout=60
)

def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

def actions():
    reports_dir = Path("reports")

    for file_path in reports_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".csv":
            print(f"Reading: {file_path}")

            for chunk in pd.read_csv(file_path, chunksize=1000):
                for row in chunk.to_dict(orient="records"):
                    document = {
                        "source_file": str(file_path),
                        "dataset": file_path.stem,
                        "record_type": "csv",
                        "content": {
                            key: clean_value(value)
                            for key, value in row.items()
                        }
                    }

                    yield {
                        "_index": INDEX_NAME,
                        "_source": document
                    }

        elif file_path.suffix.lower() == ".txt":
            print(f"Reading: {file_path}")

            document = {
                "source_file": str(file_path),
                "dataset": file_path.stem,
                "record_type": "text",
                "content": {
                    "text": file_path.read_text(
                        encoding="utf-8",
                        errors="replace"
                    )
                }
            }

            yield {
                "_index": INDEX_NAME,
                "_source": document
            }

def main():
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "source_file": {"type": "keyword"},
                    "dataset": {"type": "keyword"},
                    "record_type": {"type": "keyword"},
                    "content": {"type": "object", "dynamic": True}
                }
            }
        )
        print("Created Elasticsearch index.")

    success, errors = helpers.bulk(
        es,
        actions(),
        chunk_size=500,
        raise_on_error=False
    )

    es.indices.refresh(index=INDEX_NAME)

    print("\nINDEXING COMPLETED")
    print("Successfully indexed:", success)
    print("Failed documents:", len(errors))

    stats = es.count(index=INDEX_NAME)
    print("Total documents in Elasticsearch:", stats["count"])

    if errors:
        print("\nFirst indexing error:")
        print(json.dumps(errors[0], indent=2, default=str)[:2000])

    es.close()

if __name__ == "__main__":
    main()
