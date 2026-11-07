import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def get_neo4j_stats():
    if not all([URI, USERNAME, PASSWORD]):
        raise ValueError("Neo4j credentials are missing from .env")

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()

        with driver.session() as session:
            nodes = session.run(
                "MATCH (s:Subreddit) RETURN count(s) AS total"
            ).single()["total"]

            relationships = session.run(
                "MATCH ()-[r:LINKS_TO]->() RETURN count(r) AS total"
            ).single()["total"]

            top_nodes = session.run("""
                MATCH (s:Subreddit)
                OPTIONAL MATCH (s)-[r:LINKS_TO]->()
                RETURN s.name AS subreddit, count(r) AS outgoing_links
                ORDER BY outgoing_links DESC
                LIMIT 10
            """).data()

        return nodes, relationships, top_nodes

    finally:
        driver.close()
