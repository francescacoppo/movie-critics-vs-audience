from pathlib import Path
 
import matplotlib.pyplot as plt
import networkx as nx # corresponding library in python (=~ igraph)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
 
BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
 
# SETTINGS
N_NODES = 100                # most frequent words of each group used as nodes
# edge thresholds: share of the possible edges kept (None = cut at the mean weight, as in class)
THRESHOLDS = {"5%": 0.05, "10%": 0.10, "15%": 0.15, "mean weight": None}
MAIN_THRESHOLD = "10%"       # used for the main analysis and figures; the others are a sensitivity check
N_PERM = 1000                # random relabelling of the movies 
SEED = 42                    
 
GROUPS = ["audience_higher", "audience_lower"]
 
# keep the extreme groups 
movies_df = pd.read_csv(PROCESSED_DIR / "movies_with_sentiment.csv")
reviews_df = pd.read_csv(PROCESSED_DIR / "reviews_sentiment.csv")
 
extreme_movies = movies_df.loc[
    movies_df["group_resid"].isin(GROUPS), ["rotten_tomatoes_link", "group_resid"]
]
reviews = reviews_df.merge(extreme_movies, on="rotten_tomatoes_link", how="inner").reset_index(drop=True)
print(f"Extreme movies: {len(extreme_movies)} | reviews: {len(reviews)}")
 
# Text preparation (same choices as 06_topics.py)
texts = (
    reviews["review_content"]
    .fillna("")
    .astype(str)
    .str.replace(r"sci[- ]?fi", "scifi", case=False, regex=True)
    .str.replace(r"\[full review in \w+\]", " ", case=False, regex=True)
)
extra_stop_words = [
    "movie", "movies", "film", "films", "review", "reviews", "just", "like", "really",
    "isn", "don", "doesn", "didn", "wasn", "aren", "haven", "hasn", "hadn",
    "wouldn", "shouldn", "couldn", "ll", "ve", "re",
    "comedy", "horror", "thriller", "action", "drama", "romantic", "romance",
    "scifi", "documentary", "animated", "animation",
]
stop_words = list(ENGLISH_STOP_WORDS.union(extra_stop_words))
 
# binary = True: a word counts once per review, so co-occurrence = reviews containing both words.
# X is the incidence matrix of the bipartite network reviews x words; the word network built below
# is its one-mode projection (R: graph_from_incidence_matrix + bipartite.projection).
vectorizer = CountVectorizer(binary=True, min_df=10, max_df=0.4, stop_words=stop_words)
X = vectorizer.fit_transform(texts).tocsr()
vocab = vectorizer.get_feature_names_out()
print(f"Document-term matrix: {X.shape[0]} reviews x {X.shape[1]} words")
 
# rows of X belonging to each movie, and movies of each group
rows_by_movie = reviews.groupby("rotten_tomatoes_link").indices
movies_by_group = {g: extreme_movies.loc[extreme_movies["group_resid"] == g, "rotten_tomatoes_link"].tolist()
                   for g in GROUPS}
 
 
# Co-occurrence network of a set of movies 
def rows_for(movies):
    return np.concatenate([rows_by_movie[m] for m in movies])
 
 
def cooccurrence(rows):
    """Top N_NODES words of the reviews in `rows`, their document frequency and the Jaccard
    similarity of every pair (reviews with both words / reviews with at least one of them)."""
    Xg = X[rows]
    doc_freq_all = np.asarray(Xg.sum(axis=0)).ravel()
    top = np.argsort(doc_freq_all)[::-1][:N_NODES]
    Xt = Xg[:, top]
    co = (Xt.T @ Xt).toarray().astype(float)       # reviews containing both words
    doc_freq = np.diag(co).copy()
    jaccard = co / (doc_freq[:, None] + doc_freq[None, :] - co)
    np.fill_diagonal(jaccard, 0)
    return vocab[top], doc_freq, jaccard
 
 
def strongest_edges(jaccard, keep_fraction):
    """Keep the strongest `keep_fraction` of all possible pairs (same density for every network).
    keep_fraction=None: cut at the mean weight (delete_edges(weight < mean) seen in class);
    here the two networks can end up with different densities."""
    i, j = np.triu_indices_from(jaccard, k=1)
    w = jaccard[i, j]
    if keep_fraction is None:
        keep = w >= w.mean()
    else:
        keep = w >= np.quantile(w, 1 - keep_fraction)
    return i[keep], j[keep], w[keep]
 
 
def build_graph(words, doc_freq, i, j, w):
    G = nx.Graph()
    for word, f in zip(words, doc_freq):
        G.add_node(word, doc_freq=f)
    for a, b, weight in zip(i, j, w):
        # "weight" = strength of the link; "distance" is needed by betweenness (short path = strong link)
        G.add_edge(words[a], words[b], weight=weight, distance=1 / weight)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G
 
 
def network(cooc_g, keep_fraction):
    """Word network of one group for a given edge threshold."""
    words, doc_freq, jaccard = cooc_g
    i, j, w = strongest_edges(jaccard, keep_fraction)
    return build_graph(words, doc_freq, i, j, w)


def eigen_centrality(G):
    """Eigenvector centrality (largest connected component) of one co-occurrence network.
    Light version of analyse(), used inside the permutation loop."""
    largest = max(nx.connected_components(G), key=len)
    return pd.Series(nx.eigenvector_centrality_numpy(G.subgraph(largest), weight="weight"))
 
 
def spearman_common(ev_a, ev_b):
    """Spearman correlation of eigenvector centrality over the words present in both networks."""
    common = sorted(set(ev_a.index) & set(ev_b.index))
    return spearmanr(ev_a[common], ev_b[common])[0]
 
 
# Network measures (the same used in class: degree, closeness, betweenness, eigenvector, communities)
def analyse(G_full):
    """Measures are computed on the largest connected component (eigenvector centrality
    is not well defined on disconnected graphs)."""
    components = list(nx.connected_components(G_full))
    G = G_full.subgraph(max(components, key=len)).copy()
    n = G.number_of_nodes()
 
    communities = nx.community.greedy_modularity_communities(G, weight="weight")   # R: cluster_fast_greedy
    community_of = {w: k + 1 for k, c in enumerate(communities) for w in c}   # 1 = largest community
    eigen = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)               # R: evcent
    betw = nx.betweenness_centrality(G, weight="distance")                    # R: betweenness (weighted)
    closeness = nx.closeness_centrality(G)                                    # R: closeness(weights = NA), hops
 
    transitivity = nx.transitivity(G)                  # global clustering
    diameter = nx.diameter(G)                          # in number of steps (unweighted)
    avg_path_length = nx.average_shortest_path_length(G)
    degree_values = [d for _, d in G.degree()]
 
    nodes = pd.DataFrame({
        "word": list(G.nodes),
        "doc_freq": [G.nodes[w]["doc_freq"] for w in G.nodes],
        "degree": [G.degree(w) for w in G.nodes],
        "closeness": [closeness[w] for w in G.nodes],
        "eigenvector": [eigen[w] for w in G.nodes],
        "betweenness": [betw[w] for w in G.nodes],
        "community": [community_of[w] for w in G.nodes],
    }).sort_values("eigenvector", ascending=False).reset_index(drop=True)
 
    summary = {
        "n_nodes_with_edges": G_full.number_of_nodes(),
        "n_components": len(components),
        "n_nodes_largest_component": n,
        "n_edges_largest_component": G.number_of_edges(),
        "density": nx.density(G),
        "transitivity": transitivity,
        "diameter": diameter,                         # <-- Diameter Shortest Path
        "avg_shortest_path_length": avg_path_length,  # <-- Mean Distance Paths
        "mean_degree": np.mean(degree_values),
        "max_degree": max(degree_values),
        "n_communities": len(communities),
        "modularity": nx.community.modularity(G, communities, weight="weight"),
    }
    return G, nodes, summary
 
 
cooc = {g: cooccurrence(rows_for(movies_by_group[g])) for g in GROUPS}
graphs, node_tables, summaries = {}, {}, {}
for g in GROUPS:
    graphs[g], node_tables[g], summaries[g] = analyse(network(cooc[g], THRESHOLDS[MAIN_THRESHOLD]))
 
summary_df = pd.DataFrame(summaries)
print("\n--- NETWORK SUMMARY ---")
print(summary_df.round(3))
summary_df.to_csv(OUTPUT_DIR / "network_summary.csv")
 
print("\n--- TOP 15 WORDS BY EIGENVECTOR CENTRALITY ---")
top_words = pd.DataFrame({g: node_tables[g]["word"].head(15).to_numpy() for g in GROUPS})
print(top_words.to_string(index=False))
top_words.to_csv(OUTPUT_DIR / "network_top_words.csv", index=False)
 
print("\n--- TOP 10 WORDS BY BETWEENNESS CENTRALITY ---")
top_betw = pd.DataFrame({
    g: node_tables[g].sort_values("betweenness", ascending=False)["word"].head(10).to_numpy() for g in GROUPS
})
print(top_betw.to_string(index=False))
 
# Words of each community (top 8 by eigenvector centrality)
community_rows = []
for g in GROUPS:
    t = node_tables[g]
    for k, sub in t.groupby("community"):
        community_rows.append({
            "group": g, "community": k, "size": len(sub),
            "top_words": ", ".join(sub["word"].head(8)),
        })
communities_df = pd.DataFrame(community_rows)
print("\n--- COMMUNITIES ---")
print(communities_df.to_string(index=False))
communities_df.to_csv(OUTPUT_DIR / "network_communities.csv", index=False)
for g in GROUPS:
    node_tables[g].to_csv(OUTPUT_DIR / f"network_nodes_{g}.csv", index=False)
 
 
# Are the two networks different? Comparison with random relabelling of the movies.
# Repeated for every edge threshold: does the conclusion depend on the 10% choice?
def jaccard_sets(a, b):
    return len(a & b) / len(a | b)


def p_low(perm_values, observed):
    """Share of random relabellings at least as low as the observed value."""
    return (1 + (np.array(perm_values) <= observed).sum()) / (1 + N_PERM)


# The nodes are the top N_NODES words of each group, so node overlap does not depend on the threshold
obs_node_j = jaccard_sets(set(cooc[GROUPS[0]][0]), set(cooc[GROUPS[1]][0]))

# Observed values for every threshold
obs_rho, density = {}, {}
for name, keep_fraction in THRESHOLDS.items():
    G = {g: network(cooc[g], keep_fraction) for g in GROUPS}
    density[name] = [nx.density(G[g]) for g in GROUPS]
    # Spearman correlation of eigenvector centrality over the words present in both networks
    obs_rho[name] = spearman_common(eigen_centrality(G[GROUPS[0]]), eigen_centrality(G[GROUPS[1]]))

# Relabelling the MOVIES at random (reviews of the same movie stay together)
rng = np.random.default_rng(SEED)
all_movies = np.array(extreme_movies["rotten_tomatoes_link"])
n_first = len(movies_by_group[GROUPS[0]])
perm_node_j = []
perm_rho = {name: [] for name in THRESHOLDS}
for _ in range(N_PERM):
    shuffled = rng.permutation(all_movies)
    cooc_perm = [cooccurrence(rows_for(movies)) for movies in (shuffled[:n_first], shuffled[n_first:])]
    perm_node_j.append(jaccard_sets(set(cooc_perm[0][0]), set(cooc_perm[1][0])))
    for name, keep_fraction in THRESHOLDS.items():
        G_a, G_b = [network(c, keep_fraction) for c in cooc_perm]
        perm_rho[name].append(spearman_common(eigen_centrality(G_a), eigen_centrality(G_b)))

# Main comparison (main threshold)
m = MAIN_THRESHOLD
perm_node_j, perm_rho_m = np.array(perm_node_j), np.array(perm_rho[m])
comparison = pd.DataFrame({
    "observed": [obs_node_j, obs_rho[m]],
    "random_mean": [perm_node_j.mean(), perm_rho_m.mean()],
    "random_5th_percentile": [np.percentile(perm_node_j, 5), np.percentile(perm_rho_m, 5)],
    # share of random relabellings at least as different (low overlap / low correlation) as the real groups
    "p_value_one_sided": [p_low(perm_node_j, obs_node_j), p_low(perm_rho_m, obs_rho[m])],
}, index=["node_overlap_jaccard", "spearman_eigenvector_common_words"])
print(f"\n--- COMPARISON OF THE TWO NETWORKS vs RANDOM RELABELLING (threshold: {m}) ---")
print(comparison.round(4))
comparison.to_csv(OUTPUT_DIR / "network_comparison.csv")

# Sensitivity to the edge threshold
sensitivity_df = pd.DataFrame({
    name: {
        f"density_{GROUPS[0]}": density[name][0],
        f"density_{GROUPS[1]}": density[name][1],
        "spearman_eigenvector": obs_rho[name],
        "spearman_random_mean": np.mean(perm_rho[name]),
        "spearman_p_value": p_low(perm_rho[name], obs_rho[name]),
    } for name in THRESHOLDS
})
print("\n--- SENSITIVITY TO THE EDGE THRESHOLD ---")
print(sensitivity_df.round(3))
sensitivity_df.to_csv(OUTPUT_DIR / "network_threshold_sensitivity.csv")
 
 
# Same colours as the other figures: blue = audience higher than expected, red = audience lower than expected
COLORS = {"audience_higher": "#3b6fb6", "audience_lower": "#c0392b"}

# Figure 1: the 30 most central words.
# Node size = eigenvector centrality, colour = group.
N_SHOWN = 30   # only the most central words 
 
 
def draw(G, nodes, ax, title, color):
    H = G.subgraph(nodes["word"].head(N_SHOWN))
    pos = nx.spring_layout(H, weight="weight", seed=SEED, k=1.5 / np.sqrt(len(H)))  # Fruchterman-Reingold
    ev_s = nodes.set_index("word")["eigenvector"]
    sizes = [60 + 900 * ev_s[w] / ev_s.max() for w in H.nodes]   # area, relative to this network
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.8, alpha=0.3, edge_color="grey")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=color, node_size=sizes, alpha=0.85)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=8)
    ax.set_title(title)
    ax.axis("off")
 
 
titles = {"audience_higher": "Audience more generous than expected",
          "audience_lower": "Audience less generous than expected"}
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, g in zip(axes, GROUPS):
    draw(graphs[g], node_tables[g], ax, titles[g], COLORS[g])
plt.tight_layout()
fig.savefig(FIG_DIR / "network_groups.pdf", bbox_inches="tight")
plt.show()