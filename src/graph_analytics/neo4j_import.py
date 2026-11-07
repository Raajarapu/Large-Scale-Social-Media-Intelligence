import os
import networkx as nx
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

GRAPH_FILE = "reports/graph/reddit_network.graphml"
BATCH_SIZE = 1000


def main():
    print("Loading NetworkX graph...")

    graph = nx.read_graphml(GRAPH_FILE)

    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()

        with driver.session() as session:

            print("Creating subreddit nodes...")

            node_batch = [
                {"name": str(node)}
                for node in graph.nodes()
            ]

            for start in range(0, len(node_batch), BATCH_SIZE):
                batch = node_batch[start:start + BATCH_SIZE]

                session.run("""
                    UNWIND $batch AS item
                    MERGE (s:Subreddit {name: item.name})
                """, batch=batch)

            print("Node import completed.")

            print("Importing relationships...")

            edge_batch = []

            for source, target, attributes in graph.edges(data=True):
                edge_batch.append({
                    "source": str(source),
                    "target": str(target),
                    "link_sentiment": str(
                        attributes.get("LINK_SENTIMENT", "unknown")
                    )
                })

                if len(edge_batch) >= BATCH_SIZE:
                    session.run("""
                        UNWIND $batch AS item
                        MATCH (a:Subreddit {name: item.source})
                        MATCH (b:Subreddit {name: item.target})
                        MERGE (a)-[r:LINKS_TO]->(b)
                        SET r.link_sentiment = item.link_sentiment
                    """, batch=edge_batch)

                    edge_batch.clear()

            if edge_batch:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (a:Subreddit {name: item.source})
                    MATCH (b:Subreddit {name: item.target})
                    MERGE (a)-[r:LINKS_TO]->(b)
                    SET r.link_sentiment = item.link_sentiment
                """, batch=edge_batch)

            print("Relationship import completed.")

            result = session.run("""
                MATCH (s:Subreddit)
                RETURN count(s) AS total
            """)
            print("Verified nodes:", result.single()["total"])

            result = session.run("""
                MATCH ()-[r:LINKS_TO]->()
                RETURN count(r) AS total
            """)
            print("Verified relationships:", result.single()["total"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
