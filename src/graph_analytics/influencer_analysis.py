import networkx as nx
import pandas as pd

from pathlib import Path

INPUT_FILE = Path("reports/graph/reddit_network.graphml")

OUTPUT_FILE = Path("reports/graph/influencer_ranking.csv")


def main():

    print("Loading graph...")

    graph = nx.read_graphml(INPUT_FILE)

    print("Calculating PageRank...")

    rankings = nx.pagerank(
        graph,
        alpha=0.85,
        max_iter=100
    )

    results = pd.DataFrame(
        rankings.items(),
        columns=["subreddit", "pagerank_score"]
    )

    results = results.sort_values(
        "pagerank_score",
        ascending=False
    )

    results.to_csv(OUTPUT_FILE, index=False)

    print("\nTop 20 influential subreddits:")

    print(results.head(20).to_string(index=False))

    print("\nRanking saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
