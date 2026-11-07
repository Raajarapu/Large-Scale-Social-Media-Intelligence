import os
import ast
from pathlib import Path

import pandas as pd
import networkx as nx
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Social Media Intelligence Platform",
    page_icon="",
    layout="wide"
)

# --------------------------------------------------
# DATA LOADER
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    file = BASE / path
    if file.exists():
        return pd.read_csv(file)
    return pd.DataFrame()


sentiment = load_csv("reports/sentiment/sentiment_distribution.csv")
predictions = load_csv("reports/sentiment/sentiment_predictions.csv")
multilingual = load_csv("reports/sentiment/multilingual_predictions.csv")

topics = load_csv("reports/topics/topic_summary.csv")
topic_assignments = load_csv("reports/topics/topic_assignments.csv")

metrics = load_csv("reports/misinformation/model_metrics.csv")

influencers = load_csv("reports/graph/influencer_ranking.csv")
communities = load_csv("reports/graph/community_assignments.csv")

graph_path = BASE / "reports/graph/reddit_network.graphml"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("SOCIAL INTELLIGENCE")

section = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sentiment Analysis",
        "Topic Discovery",
        "Misinformation Analysis",
        "Network Intelligence",
        "Community Detection",
        "Neo4j Graph Intelligence",
        "Elasticsearch Search"
    ]
)

st.sidebar.divider()
st.sidebar.caption("DS-03 | Social Media Intelligence")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Large-Scale Social Media Intelligence Platform")
st.caption(
    "An integrated platform for sentiment analysis, "
    "topic discovery, misinformation classification "
    "and network intelligence."
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if section == "Overview":

    st.header("Platform Overview")

    c1, c2, c3, c4 = st.columns(4)

    total_sentiment = (
        int(sentiment["record_count"].sum())
        if "record_count" in sentiment else 0
    )

    total_topics = (
        len(topics[topics["Topic"] != -1])
        if "Topic" in topics else 0
    )

    total_subreddits = len(influencers)

    total_communities = (
        communities["community_id"].nunique()
        if "community_id" in communities else 0
    )

    c1.metric("Sentiment Records", f"{total_sentiment:,}")
    c2.metric("Discovered Topics", f"{total_topics:,}")
    c3.metric("Ranked Subreddits", f"{total_subreddits:,}")
    c4.metric("Communities", f"{total_communities:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")

        if not sentiment.empty:
            fig = px.pie(
                sentiment,
                names="sentiment",
                values="record_count",
                hole=0.45
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top Influential Subreddits")

        if not influencers.empty:
            st.dataframe(
                influencers.head(10),
                use_container_width=True
            )

    st.subheader("Misinformation Model Performance")

    if not metrics.empty:
        m = metrics.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Weighted F1", f"{m['weighted_f1']:.2%}")
        c3.metric("Test Records", f"{int(m['testing_records']):,}")

# --------------------------------------------------
# SENTIMENT
# --------------------------------------------------

elif section == "Sentiment Analysis":

    st.header("Sentiment Intelligence")

    if not sentiment.empty:

        c1, c2, c3 = st.columns(3)

        for col, label in zip(
            [c1, c2, c3],
            ["positive", "neutral", "negative"]
        ):
            row = sentiment[
                sentiment["sentiment"].str.lower() == label
            ]

            count = int(row["record_count"].sum()) if not row.empty else 0
            col.metric(label.title(), f"{count:,}")

        st.subheader("Sentiment Distribution")

        fig = px.bar(
            sentiment,
            x="sentiment",
            y="record_count",
            color="sentiment",
            text="percentage"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sentiment Statistics")
        st.dataframe(sentiment, use_container_width=True)

    if not predictions.empty:
        st.subheader("Sentiment Predictions")

        st.dataframe(
            predictions.head(100),
            use_container_width=True
        )

    if not multilingual.empty:
        st.subheader("Multilingual Model Predictions")

        st.dataframe(
            multilingual.head(100),
            use_container_width=True
        )

    st.info(
        "The multilingual model output should be interpreted "
        "according to the languages represented in its input dataset."
    )

# --------------------------------------------------
# TOPIC DISCOVERY
# --------------------------------------------------

elif section == "Topic Discovery":

    st.header("BERTopic Topic Discovery")

    if not topics.empty:

        actual_topics = topics[topics["Topic"] != -1]

        outlier_count = int(
            topics.loc[topics["Topic"] == -1, "Count"].sum()
        )

        c1, c2 = st.columns(2)

        c1.metric("Discovered Topics", len(actual_topics))
        c2.metric("Outlier Records", f"{outlier_count:,}")

        st.subheader("Topic Frequency")

        top_topics = actual_topics.sort_values(
            "Count", ascending=False
        ).head(15)

        fig = px.bar(
            top_topics,
            x="Count",
            y="Name",
            orientation="h",
            title="Top 15 Discovered Topics"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Explore Topics")

        for _, row in actual_topics.iterrows():

            with st.expander(
                f"Topic {row['Topic']} | {row['Name']}"
            ):

                st.write("Document count:", int(row["Count"]))

                try:
                    words = ast.literal_eval(row["Representation"])
                    st.write("Keywords:", ", ".join(words))
                except (ValueError, SyntaxError):
                    st.write("Keywords:", row["Representation"])

                st.write("Representative documents:")
                st.write(row["Representative_Docs"])

    else:
        st.warning("Topic summary is unavailable.")

# --------------------------------------------------
# MISINFORMATION
# --------------------------------------------------

elif section == "Misinformation Analysis":

    st.header("Misinformation Classification")

    if not metrics.empty:

        m = metrics.iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Weighted F1", f"{m['weighted_f1']:.2%}")
        c3.metric("Training", f"{int(m['training_records']):,}")
        c4.metric("Validation", f"{int(m['validation_records']):,}")
        c5.metric("Testing", f"{int(m['testing_records']):,}")

        st.warning(
            "The current classifier has limited predictive performance. "
            "Its predictions represent learned LIAR dataset labels, "
            "not independently verified facts."
        )

    report = BASE / "reports/misinformation/classification_report.txt"

    if report.exists():
        st.subheader("Classification Report")
        st.text(report.read_text())

# --------------------------------------------------
# NETWORKX
# --------------------------------------------------

elif section == "Network Intelligence":

    st.header("Network Intelligence")

    if graph_path.exists():

        graph = nx.read_graphml(graph_path)

        c1, c2, c3 = st.columns(3)

        c1.metric("Nodes", f"{graph.number_of_nodes():,}")
        c2.metric("Edges", f"{graph.number_of_edges():,}")
        c3.metric("Network", "Directed")

    if not influencers.empty:

        st.subheader("Influencer Ranking")

        search = st.text_input("Search subreddit")

        filtered = influencers.copy()

        if search:
            filtered = filtered[
                filtered["subreddit"].str.contains(
                    search, case=False, na=False
                )
            ]

        st.dataframe(
            filtered.head(100),
            use_container_width=True
        )

        st.subheader("Top 15 Influential Subreddits")

        fig = px.bar(
            influencers.head(15),
            x="pagerank_score",
            y="subreddit",
            orientation="h"
        )

        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# COMMUNITY DETECTION
# --------------------------------------------------

elif section == "Community Detection":

    st.header("Community Detection")

    if not communities.empty:

        distribution = (
            communities.groupby("community_id")
            .size()
            .reset_index(name="subreddit_count")
            .sort_values("subreddit_count", ascending=False)
        )

        st.metric(
            "Detected Communities",
            communities["community_id"].nunique()
        )

        st.subheader("Community Distribution")

        fig = px.bar(
            distribution.head(20),
            x="community_id",
            y="subreddit_count"
        )

        st.plotly_chart(fig, use_container_width=True)

        selected = st.selectbox(
            "Select Community",
            sorted(communities["community_id"].unique())
        )

        members = communities[
            communities["community_id"] == selected
        ]

        st.write("Subreddits in selected community:", len(members))

        st.dataframe(members, use_container_width=True)

# --------------------------------------------------
# NEO4J LIVE INTEGRATION
# --------------------------------------------------

elif section == "Neo4j Graph Intelligence":

    st.header("Neo4j Graph Intelligence")
    st.caption("Live statistics from Neo4j AuraDB")

    try:

        from utils.neo4j_client import get_neo4j_stats

        nodes, relationships, top_nodes = get_neo4j_stats()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Nodes", f"{nodes:,}")
        c2.metric("Relationships", f"{relationships:,}")
        c3.metric("Database Status", "Connected")

        st.success("Successfully connected to Neo4j AuraDB.")

        st.subheader("Top Subreddits by Outgoing Links")

        ranking = pd.DataFrame(top_nodes)

        st.dataframe(ranking, use_container_width=True)

        if not ranking.empty:
            fig = px.bar(
                ranking,
                x="outgoing_links",
                y="subreddit",
                orientation="h"
            )

            st.plotly_chart(fig, use_container_width=True)

        if relationships < 137821:
            st.warning(
                f"Partial graph import: {relationships:,} relationships "
                "currently exist in Neo4j. The source graph contains "
                "137,821 edges."
            )

    except Exception as e:
        st.error(f"Neo4j connection failed: {e}")

# --------------------------------------------------
# ELASTICSEARCH
# --------------------------------------------------

elif section == "Elasticsearch Search":

    st.header("Elasticsearch Social Media Search")

    elastic_url = os.getenv("ELASTICSEARCH_URL")
    elastic_key = os.getenv("ELASTICSEARCH_API_KEY")

    if not elastic_url or not elastic_key:

        st.warning(
            "Elasticsearch credentials are not configured. "
            "Set ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY "
            "in your environment secrets."
        )

    else:

        try:

            from elasticsearch import Elasticsearch

            es = Elasticsearch(
                elastic_url,
                api_key=elastic_key,
                request_timeout=30
            )

            if es.ping():

                st.success("Elasticsearch connected successfully.")

                index_name = "social-media-intelligence"

                query = st.text_input(
                    "Search indexed social media records"
                )

                if query:

                    response = es.search(
                        index=index_name,
                        query={
                            "multi_match": {
                                "query": query,
                                "fields": ["text", "clean_text", "topic", "sentiment"]
                            }
                        },
                        size=50
                    )

                    hits = [
                        hit["_source"]
                        for hit in response["hits"]["hits"]
                    ]

                    st.write(
                        "Results:",
                        response["hits"]["total"]["value"]
                    )

                    if hits:
                        st.dataframe(
                            pd.DataFrame(hits),
                            use_container_width=True
                        )
                    else:
                        st.info("No matching records found.")

            else:
                st.error("Unable to verify Elasticsearch connection.")

        except Exception as e:
            st.error(f"Elasticsearch error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DS-03 | Large-Scale Social Media Intelligence Platform"
)

st.caption(
    "Research dataset-based analytics | "
    "NetworkX | Neo4j | Elasticsearch | Kibana"
)

kibana_url = os.getenv("KIBANA_URL")

if kibana_url:
    st.markdown(f"[Open Kibana Dashboard]({kibana_url})")
