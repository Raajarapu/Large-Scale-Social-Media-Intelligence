import networkx as nx
import pandas as pd

from pathlib import Path
from networkx.algorithms.community import louvain_communities

INPUT_FILE = Path("reports/graph/reddit_network.graphml")

OUTPUT_FILE = Path("reports/graph/community_assignments.csv")


def main():

    print("Loading graph...")

    directed_graph = nx.read_graphml(INPUT_FILE)

    graph = directed_graph.to_undirected()

    print("Detecting communities...")

    communities = louvain_communities(
        graph,
        seed=42
    )

    records = []

    for community_id, nodes in enumerate(communities):

        for node in nodes:

            records.append({
                "subreddit": node,
                "community_id": community_id
            })

    results = pd.DataFrame(records)

    results.to_csv(OUTPUT_FILE, index=False)

    print("Total communities:", len(communities))

    print("Community assignments saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
