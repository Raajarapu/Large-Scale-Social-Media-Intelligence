import os
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from urllib.parse import urlparse

load_dotenv(override=False)


def get_config(key):
    try:
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass

    return os.getenv(key, "").strip()


URI = get_config("NEO4J_URI")
USERNAME = get_config("NEO4J_USERNAME")
PASSWORD = get_config("NEO4J_PASSWORD")


def get_neo4j_stats():
    if not all([URI, USERNAME, PASSWORD]):
        raise ValueError("Neo4j credentials are missing")

    hostname = urlparse(URI).hostname
    st.write("Neo4j hostname loaded:", hostname)

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
        