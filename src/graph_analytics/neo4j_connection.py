import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def test_connection():
    if not URI or not USERNAME or not PASSWORD:
        raise ValueError("Neo4j credentials are missing from .env")

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(
                "RETURN 'Neo4j Connected Successfully' AS message"
            )
            print(result.single()["message"])
    finally:
        driver.close()


if __name__ == "__main__":
    test_connection()
