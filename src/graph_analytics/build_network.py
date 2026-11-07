import pandas as pd
import networkx as nx

from pathlib import Path

INPUT_FILE = Path(
    "data/external/reddit/soc-redditHyperlinks-body.tsv"
)

OUTPUT_DIR = Path("reports/graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading Reddit interaction dataset...")

    df = pd.read_csv(INPUT_FILE, sep="\t")

    print("Dataset shape:", df.shape)

    df = df.dropna(
        subset=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]
    )

    print("Building directed graph...")

    graph = nx.from_pandas_edgelist(
        df,
        source="SOURCE_SUBREDDIT",
        target="TARGET_SUBREDDIT",
        edge_attr="LINK_SENTIMENT",
        create_using=nx.DiGraph()
    )

    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())

    print("Saving graph...")

    nx.write_graphml(
        graph,
        OUTPUT_DIR / "reddit_network.graphml"
    )

    print("Graph construction completed.")


if __name__ == "__main__":
    main()
