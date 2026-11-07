# Large-Scale Social Media Intelligence Platform

An end-to-end Data Science platform designed to transform large-scale social media datasets into actionable intelligence through Natural Language Processing, Machine Learning, Topic Modelling, Graph Analytics and Distributed Search.

## 1. Business Problem

Social media platforms generate enormous volumes of unstructured information every second. Organisations, researchers, marketing teams and decision-makers increasingly depend on these platforms to understand public opinion, identify emerging discussions, monitor brand perception and recognise potentially misleading information.

However, extracting meaningful intelligence from this information presents several challenges:

- **Information Overload:** Analysing millions of unstructured social media posts manually is inefficient and impractical.
- **Sentiment Identification:** Organisations struggle to understand whether public discussions reflect positive, negative or neutral opinions.
- **Emerging Topic Detection:** Identifying new discussions and monitoring changes in public interest requires automated topic discovery.
- **Misinformation Detection:** Potentially misleading claims can spread rapidly, making automated classification and investigation increasingly important.
- **Influence Identification:** Understanding relationships between online communities and identifying structurally influential groups is challenging.
- **Fragmented Intelligence:** Analytical results are often distributed across separate tools, making unified exploration difficult.

### Proposed Solution

The proposed platform integrates multiple Data Science technologies into a unified analytical pipeline.

It processes publicly available social media datasets, performs sentiment classification, discovers discussion topics, builds a misinformation classification model, analyses subreddit relationships, and indexes analytical outputs for efficient retrieval.

The resulting intelligence is presented through an interactive Streamlit dashboard.

The objective is to convert raw social media information into structured, searchable and interpretable insights.

---

## 2. Project Objectives

1. Develop a scalable pipeline for processing large-scale social media datasets.
2. Perform text cleaning, normalisation and language identification.
3. Implement transformer-based sentiment classification.
4. Discover meaningful discussion patterns using BERTopic.
5. Develop a supervised misinformation classification model.
6. Construct directed subreddit relationship networks.
7. Identify influential network nodes using PageRank.
8. Detect interconnected communities using NetworkX.
9. Integrate graph information into Neo4j.
10. Index analytical outputs in Elasticsearch.
11. Develop an interactive dashboard for exploring social media intelligence.

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming | Python | Core implementation |
| Data Processing | Pandas, NumPy | Data manipulation and preparation |
| Text Processing | NLTK, Unicode Normalisation | Text cleaning and standardisation |
| Language Detection | LangDetect | Language identification |
| NLP Models | Hugging Face Transformers | Transformer-based text classification |
| Sentiment Analysis | Twitter-RoBERTa, Multilingual DistilBERT | Sentiment prediction |
| Semantic Embeddings | Sentence Transformers | Text representation |
| Topic Modelling | BERTopic | Semantic topic discovery |
| Machine Learning | Scikit-learn | Misinformation classification |
| Feature Engineering | TF-IDF | Text vectorisation |
| Graph Analytics | NetworkX | PageRank and community detection |
| Graph Database | Neo4j AuraDB | Graph storage and querying |
| Search Engine | Elasticsearch | Indexing and information retrieval |
| Visualisation | Streamlit | Interactive analytical dashboard |
| Version Control | Git, GitHub | Source-code management |

---

## 4. Data Sources

The implementation uses publicly available datasets selected according to the analytical requirements.

| Dataset | Application | Purpose |
|---|---|---|
| TweetEval | Sentiment Analysis | Social media sentiment classification |
| TweetEval | Topic Modelling | Identifying discussion patterns |
| LIAR | Misinformation Detection | Training and evaluating a supervised classifier |
| SNAP Reddit Hyperlink Network | Graph Analytics | Analysing subreddit-to-subreddit relationships |

### Dataset Organisation

**TweetEval:** Provides labelled social media text for sentiment analysis and topic discovery.

**LIAR:** Provides political statements categorised into six truthfulness labels for misinformation classification.

**SNAP Reddit Hyperlink Network:** Provides directed relationships between subreddits based on hyperlinks shared in Reddit posts.

**Implementation scope:** This version uses public datasets rather than live Reddit or YouTube API ingestion. The graph represents subreddit relationships, not individual user interactions.

---

## 5. End-to-End Data Science Lifecycle

The project follows a structured Data Science lifecycle, progressing from business understanding and data acquisition to modelling, evaluation, integration and presentation.

### Stage 1: Business Understanding

Identify the challenges associated with large-scale social media analysis.

Define the analytical requirements:

- Sentiment identification.
- Topic discovery.
- Misinformation classification.
- Network intelligence.
- Search and information retrieval.

**Deliverable:** Defined business problem, objectives and analytical scope.

### Stage 2: Data Acquisition

Identify and obtain the datasets required for each analytical component.

Activities:

- Acquire TweetEval.
- Acquire LIAR.
- Acquire SNAP Reddit Hyperlink Network.
- Organise original datasets.
- Inspect dataset structure and attributes.

**Deliverable:** Structured external datasets ready for processing.

### Stage 3: Data Understanding and Exploration

Examine the datasets to understand their structure, quality and suitability.

Activities:

- Inspect records and columns.
- Analyse dataset distributions.
- Identify missing values.
- Examine text characteristics.
- Understand classification labels.
- Inspect network relationship attributes.

**Deliverable:** Understanding of the input datasets and their analytical requirements.

### Stage 4: Data Cleaning and Preprocessing

Prepare unstructured text and structured records for downstream analysis.

Implemented operations:

- Unicode normalisation.
- URL removal.
- HTML removal.
- Emoji demojisation.
- User-mention normalisation.
- Hashtag marker normalisation.
- Whitespace standardisation.
- Language detection.
- Dataset-specific preparation.

**Deliverable:** Cleaned and structured datasets.

### Stage 5: Feature Engineering and Representation

Convert textual information into representations suitable for machine learning and semantic analysis.

Techniques:

- TF-IDF vectorisation.
- Unigram and bigram extraction.
- Sentence Transformer embeddings.
- Text normalisation.
- Label preparation.

**Deliverable:** Model-ready numerical and semantic representations.

### Stage 6: Model Development

Develop specialised analytical models for different intelligence requirements.

**A. Sentiment Analysis**

Models:
- CardiffNLP Twitter-RoBERTa.
- Multilingual DistilBERT.

Classification categories:
- Positive.
- Negative.
- Neutral.

**B. Topic Modelling**

Technology:
- BERTopic.
- Sentence Transformers.
- MiniLM embeddings.

Purpose:
- Identify semantically related discussions.
- Extract representative terms.
- Discover recurring themes.

**C. Misinformation Classification**

Model:
- TF-IDF.
- Logistic Regression.

Purpose:
- Predict the labelled truthfulness category of a statement using the LIAR dataset.

**Deliverable:** Developed sentiment, topic discovery and misinformation analysis components.

### Stage 7: Graph Construction and Network Intelligence

Construct a directed graph using the SNAP Reddit Hyperlink Network.

Graph representation:

- Source subreddit: Source node.
- Target subreddit: Destination node.
- Hyperlink relationship: Directed edge.

Analytical techniques:

- Network construction.
- PageRank centrality.
- Community detection.
- Graph export.

Technologies:
- NetworkX.
- Neo4j AuraDB.

**Deliverable:** Subreddit relationship graph, influence rankings and community assignments.

### Stage 8: Search and Indexing

Integrate Elasticsearch to make generated analytical reports searchable.

Activities:

- Establish Elasticsearch connectivity.
- Read analytical CSV outputs.
- Convert records into structured documents.
- Index documents.
- Execute text-based search.
- Retrieve matching records.

**Deliverable:** Searchable analytical index containing 75,587 documents.

### Stage 9: Application Development

Integrate the analytical components into a unified Streamlit application.

The dashboard contains:

1. Overview.
2. Sentiment Analysis.
3. Topic Discovery.
4. Misinformation Analysis.
5. Network Intelligence.
6. Community Detection.
7. Neo4j Graph Intelligence.
8. Elasticsearch Search.

**Deliverable:** Interactive social media intelligence dashboard.

### Stage 10: Evaluation and Validation

Evaluate the analytical components using appropriate metrics, generated outputs and integration tests.

Validation includes:

- Classification performance.
- Sentiment distribution.
- Topic extraction results.
- Network statistics.
- Graph database connectivity.
- Elasticsearch indexing and retrieval.
- Application syntax and functionality.

**Deliverable:** Recorded experimental results and an integrated working application.

---

## 6. Model Development and Evaluation

### 6.1 Sentiment Analysis Evaluation

Two pretrained transformer architectures were integrated.

| Model | Purpose |
|---|---|
| CardiffNLP Twitter-RoBERTa | Social media sentiment classification |
| Multilingual DistilBERT | Multilingual sentiment model integration |

The recorded sentiment evaluation used 500 TweetEval records.

#### Results

| Sentiment | Records | Distribution |
|---|---:|---:|
| Negative | 207 | 41.4% |
| Neutral | 201 | 40.2% |
| Positive | 92 | 18.4% |
| **Total** | **500** | **100%** |

**Interpretation:**

Negative sentiment represents the largest category in the evaluated sample, followed closely by neutral sentiment. Positive sentiment accounts for the remaining portion.

These values describe the model's predicted distribution, not independently verified public opinion.

### 6.2 Topic Modelling Evaluation

BERTopic was applied to 3,000 TweetEval training records.

| Evaluation Item | Result |
|---|---:|
| Input Records | 3,000 |
| Identified Topics | 29 |
| Outlier Topic | -1 |

**Interpretation:**

The model identified 29 discussion topics, while records that could not be confidently assigned to a topic were grouped into the outlier category.

Evaluation focused on topic extraction, representative terms and representative documents.

### 6.3 Misinformation Classification Evaluation

A Logistic Regression classifier was trained using TF-IDF features.

#### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Feature Extraction | TF-IDF |
| Maximum Features | 50,000 |
| N-gram Range | (1, 2) |
| Class Weight | Balanced |

#### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 10,240 |
| Validation | 1,284 |
| Testing | 1,267 |
| **Total** | **12,791** |

#### Evaluation Metrics

| Metric | Result |
|---|---:|
| Accuracy | 22.97% |
| Weighted F1-Score | 23.13% |

**Interpretation:**

The baseline classifier demonstrates that a traditional machine learning pipeline can be developed for six-category statement classification. However, the recorded performance is limited and indicates scope for improvement through stronger text representations, model optimisation and additional experiments.

The predictions represent LIAR dataset label classification, not independent fact-checking.

### 6.4 Network Analytics Evaluation

The SNAP Reddit Hyperlink Network was used to construct a directed subreddit graph.

#### Graph Statistics

| Metric | Result |
|---|---:|
| Nodes | 35,776 |
| Directed Edges | 137,821 |
| PageRank | Generated |
| Community Assignments | Generated |

**Evaluation techniques:**

- Graph size and connectivity inspection.
- PageRank-based node ranking.
- Community assignment generation.
- GraphML export.

The resulting graph provides structural insights into relationships between subreddit communities.

### 6.5 Neo4j Integration Evaluation

| Metric | Result |
|---|---:|
| Nodes Imported | 35,776 |
| Relationships Imported | Approximately 77,000 |
| Database | Neo4j AuraDB |

Neo4j connectivity and graph retrieval were tested successfully.

The relationship import is partial compared with the complete NetworkX graph containing 137,821 directed edges.

### 6.6 Elasticsearch Evaluation

| Metric | Result |
|---|---:|
| Index Name | social-media-intelligence |
| Indexed Documents | 75,587 |
| Failed Documents | 0 |
| Search Integration | Successful |

Search functionality was tested using:

- Positive.
- Negative.
- Neutral.

All three sentiment categories returned matching indexed records.

This validates the indexing and retrieval workflow for the tested queries.

---

## 7. System Architecture

```text
                    PUBLIC DATASETS
                           |
          +----------------+----------------+
          |                |                |
       TweetEval          LIAR          SNAP Reddit
          |                |             Network
          +----------------+----------------+
                           |
                           v
                  DATA PREPROCESSING
                           |
                  TEXT CLEANING AND
                  FEATURE ENGINEERING
                           |
                           v
                   ANALYTICAL MODELS
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
     SENTIMENT          TOPIC          MISINFORMATION
     ANALYSIS           MODELLING       CLASSIFICATION
          |                |                |
          +----------------+----------------+
                           |
                           v
                    GRAPH ANALYTICS
                           |
                    +------+------+
                    |             |
                 NetworkX        Neo4j
                    |             |
                 PageRank     Graph Storage
                 Communities  and Queries
                    |             |
                    +------+------+
                           |
                           v
                     ELASTICSEARCH
                           |
                    INDEXING AND SEARCH
                           |
                           v
                    STREAMLIT DASHBOARD
                           |
                           v
                  SOCIAL MEDIA INTELLIGENCE
```

---

## 8. Project Directory Structure

```text
Large-Scale-Social-Media-Intelligence/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
│
├── data/
│   ├── external/
│   │   ├── tweeteval/
│   │   ├── liar/
│   │   │   ├── train.tsv
│   │   │   ├── valid.tsv
│   │   │   └── test.tsv
│   │   └── reddit/
│   │       ├── soc-redditHyperlinks-body.tsv
│   │       └── soc-redditHyperlinks-title.tsv
│   │
│   └── processed/
│       ├── tweeteval_train.csv
│       ├── tweeteval_val.csv
│       ├── tweeteval_test.csv
│       ├── liar_train.csv
│       ├── liar_valid.csv
│       ├── liar_test.csv
│       └── reddit_clean.csv
│
├── src/
│   └── search/
│       └── index_reports.py
│
├── models/
│
├── utils/
│   └── neo4j_client.py
│
└── reports/
    ├── sentiment/
    ├── topics/
    ├── misinformation/
    └── graph/
        ├── reddit_network.graphml
        ├── influencer_ranking.csv
        └── community_assignments.csv
```

---

## 9. Limitations and Future Scope

The current implementation establishes a functional foundation, but several capabilities can be extended.

| Area | Current Limitation | Future Enhancement |
|---|---|---|
| Data Acquisition | Public datasets used instead of live API ingestion | Integrate compliant Reddit and YouTube APIs |
| Sentiment Analysis | Limited evaluation sample | Evaluate across multiple languages and larger datasets |
| Sentiment Trends | No complete timestamp-based analysis | Implement temporal sentiment tracking |
| Topic Modelling | Topic extraction completed | Monitor topic growth and decline over time |
| Misinformation | Baseline model has limited performance | Experiment with transformer-based classifiers |
| Graph Intelligence | Subreddit-level relationships | Extend to user interactions where permitted |
| Neo4j | Partial relationship import | Complete and validate graph synchronisation |
| Search Analytics | Elasticsearch search implemented | Develop Kibana dashboards and alerts |
| Scalability | Current pipeline uses dataset-based processing | Explore distributed data-processing architectures |

---

## 10. Conclusion

The Large-Scale Social Media Intelligence Platform demonstrates the practical integration of Data Science, Natural Language Processing, Machine Learning, Graph Analytics and Search Technologies into a unified analytical solution.

The implementation establishes an end-to-end workflow that transforms raw social media datasets into structured analytical outputs.

Transformer-based sentiment models provide opinion classification, BERTopic identifies semantic discussion patterns, and Logistic Regression establishes a baseline for misinformation classification. NetworkX and Neo4j extend the platform into graph-based intelligence by analysing subreddit relationships, ranking influential nodes and identifying communities. Elasticsearch provides searchable access to generated analytical records, while Streamlit brings the individual components together through an interactive dashboard.

The project also demonstrates the importance of evaluating models objectively, distinguishing experimental results from business conclusions, and recognising the limitations of the underlying datasets and analytical methods.

**Ultimately, the platform provides a modular foundation for converting large-scale social media data into actionable intelligence, supporting deeper understanding of public discussions, online communities and information patterns.**
