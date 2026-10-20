from pathlib import Path
 
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score
 
BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
 
# SETTINGS
N_NODES = 100                # most frequent words of each group used as nodes
EDGE_KEEP_FRACTION = 0.10    # keep only the strongest 10% of the possible edges
N_PERM = 1000                # random relabelling of the movies 
N_LABELS = 20                # words labelled in the figure
RUN_GIRVAN_NEWMAN = True     # robustness check of the communities (slower: a few minutes)
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
 
 
def strongest_edges(jaccard):
    """Keep the strongest EDGE_KEEP_FRACTION of all possible pairs (same density for every network).
    Alternative seen in class: cut at the mean weight (delete_edges(weight < mean)); not used here
    because it would give the two networks different densities."""
    i, j = np.triu_indices_from(jaccard, k=1)
    w = jaccard[i, j]
    keep = w >= np.quantile(w, 1 - EDGE_KEEP_FRACTION)
    return i[keep], j[keep], w[keep]
 
 
def edge_set(words, i, j):
    return {tuple(sorted((words[a], words[b]))) for a, b in zip(i, j)}
 
 
def build_graph(words, doc_freq, i, j, w):
    G = nx.Graph()
    for word, f in zip(words, doc_freq):
        G.add_node(word, doc_freq=f)
    for a, b, weight in zip(i, j, w):
        # "weight" = strength of the link; "distance" is needed by betweenness (short path = strong link)
        G.add_edge(words[a], words[b], weight=weight, distance=1 / weight)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G
 
 
def eigen_centrality(words, doc_freq, i, j, w):
    """Eigenvector centrality (largest connected component) of one co-occurrence network.
    Light version of analyse(), used inside the permutation loop."""
    G = build_graph(words, doc_freq, i, j, w)
    largest = max(nx.connected_components(G), key=len)
    return pd.Series(nx.eigenvector_centrality_numpy(G.subgraph(largest), weight="weight"))
 
 
def spearman_common(ev_a, ev_b):
    """Spearman correlation of eigenvector centrality over the words present in both networks."""
    common = sorted(set(ev_a.index) & set(ev_b.index))
    return spearmanr(ev_a[common], ev_b[common])[0]
 
 
# Network measures (the same used in class: degree, closeness, betweenness, eigenvector, communities)
def centralization(values, theoretical_max):
    """Freeman centralization: sum of (max - value) over the nodes, divided by the largest value it
    can take (a star graph with the same number of nodes). 0 = all nodes alike, 1 = star.
    R: centr_degree / centr_clo / centr_betw (normalized = TRUE)."""
    v = np.asarray(list(values), dtype=float)
    return (v.max() - v).sum() / theoretical_max
 
 
def girvan_newman_best(G):
    """Girvan-Newman communities (R: cluster_edge_betweenness, as seen in class). Edges are removed in
    order of weighted edge betweenness (distance = 1/weight); the partition with the highest
    modularity is kept."""
    def most_valuable_edge(H):
        eb = nx.edge_betweenness_centrality(H, weight="distance")
        return max(eb, key=eb.get)
 
    best, best_q = None, -np.inf
    for partition in nx.community.girvan_newman(G, most_valuable_edge=most_valuable_edge):
        q = nx.community.modularity(G, partition, weight="weight")
        if q > best_q:
            best, best_q = partition, q
    return best, best_q
 
 
def analyse(G_full):
    """Measures are computed on the largest connected component (eigenvector centrality
    is not well defined on disconnected graphs)."""
    components = list(nx.connected_components(G_full))
    G = G_full.subgraph(max(components, key=len)).copy()
    n = G.number_of_nodes()
 
    communities = nx.community.greedy_modularity_communities(G, weight="weight")   # R: cluster_fast_greedy
    community_of = {w: k + 1 for k, c in enumerate(communities) for w in c}   # 1 = largest community
    eigen = nx.eigenvector_centrality_numpy(G, weight="weight")               # R: evcent
    betw = nx.betweenness_centrality(G, weight="distance")                    # R: betweenness (weighted)
    closeness = nx.closeness_centrality(G)                                    # R: closeness(weights = NA), hops
    degree_c = nx.degree_centrality(G)                                        # R: degree(normalized = TRUE)
    betw_hops = nx.betweenness_centrality(G)                                  # unweighted, for centralization only
 
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
        "avg_clustering": nx.average_clustering(G, weight="weight"),
        "transitivity": transitivity,
        "diameter": diameter,                         # <-- Diameter Shortest Path
        "avg_shortest_path_length": avg_path_length,  # <-- Mean Distance Paths
        "mean_degree": np.mean(degree_values),
        "max_degree": max(degree_values),
        # Freeman centralization (theoretical maxima for an undirected star graph with n nodes,
        # applied to the normalized centralities, as in igraph with normalized = TRUE)
        "centralization_degree": centralization(degree_c.values(), n - 2),
        "centralization_closeness": centralization(closeness.values(), (n - 1) * (n - 2) / (2 * n - 3)),
        "centralization_betweenness": centralization(betw_hops.values(), n - 1),
        # do the "central" words (eigenvector) and the "bridge" words (betweenness) coincide?
        "corr_betw_eigen_pearson": nodes["betweenness"].corr(nodes["eigenvector"]),
        "corr_betw_eigen_spearman": nodes["betweenness"].corr(nodes["eigenvector"], method="spearman"),
        "n_communities": len(communities),
        "modularity": nx.community.modularity(G, communities, weight="weight"),
    }
 
    if RUN_GIRVAN_NEWMAN:
        gn_partition, gn_q = girvan_newman_best(G)
        gn_of = {w: k for k, c in enumerate(gn_partition) for w in c}
        summary["n_communities_girvan_newman"] = len(gn_partition)
        summary["modularity_girvan_newman"] = gn_q
        summary["adjusted_rand_greedy_vs_girvan_newman"] = adjusted_rand_score(
            [community_of[w] for w in G.nodes], [gn_of[w] for w in G.nodes])
    return G, nodes, summary
 
 
graphs, node_tables, summaries, edge_sets = {}, {}, {}, {}
for g in GROUPS:
    words, doc_freq, jac = cooccurrence(rows_for(movies_by_group[g]))
    i, j, w = strongest_edges(jac)
    graphs[g], node_tables[g], summaries[g] = analyse(build_graph(words, doc_freq, i, j, w))
    edge_sets[g] = (set(words), edge_set(words, i, j))
 
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
 
 
# Are the two networks different? Comparison with random relabelling
def jaccard_sets(a, b):
    return len(a & b) / len(a | b)
 
 
def overlap(nodes_edges_a, nodes_edges_b):
    return jaccard_sets(nodes_edges_a[0], nodes_edges_b[0]), jaccard_sets(nodes_edges_a[1], nodes_edges_b[1])
 
 
obs_node_j, obs_edge_j = overlap(edge_sets[GROUPS[0]], edge_sets[GROUPS[1]])
 
# Spearman correlation of eigenvector centrality over the words present in both networks
ev = {g: node_tables[g].set_index("word")["eigenvector"] for g in GROUPS}
rho = spearman_common(ev[GROUPS[0]], ev[GROUPS[1]])
 
# Relabelling the MOVIES at random (reviews of the same movie stay together)
rng = np.random.default_rng(SEED)
all_movies = np.array(extreme_movies["rotten_tomatoes_link"])
n_first = len(movies_by_group[GROUPS[0]])
perm_node_j, perm_edge_j, perm_rho = [], [], []
for _ in range(N_PERM):
    shuffled = rng.permutation(all_movies)
    parts, evs = [], []
    for movies in (shuffled[:n_first], shuffled[n_first:]):
        words, doc_freq, jac = cooccurrence(rows_for(movies))
        i, j, w = strongest_edges(jac)
        parts.append((set(words), edge_set(words, i, j)))
        evs.append(eigen_centrality(words, doc_freq, i, j, w))
    nj, ej = overlap(parts[0], parts[1])
    perm_node_j.append(nj)
    perm_edge_j.append(ej)
    perm_rho.append(spearman_common(evs[0], evs[1]))
perm_node_j, perm_edge_j, perm_rho = np.array(perm_node_j), np.array(perm_edge_j), np.array(perm_rho)
 
comparison = pd.DataFrame({
    "observed": [obs_node_j, obs_edge_j, rho],
    "random_mean": [perm_node_j.mean(), perm_edge_j.mean(), perm_rho.mean()],
    "random_5th_percentile": [np.percentile(perm_node_j, 5), np.percentile(perm_edge_j, 5),
                              np.percentile(perm_rho, 5)],
    # share of random relabellings at least as different (low overlap / low correlation) as the real groups
    "p_value_one_sided": [(1 + (perm_node_j <= obs_node_j).sum()) / (1 + N_PERM),
                          (1 + (perm_edge_j <= obs_edge_j).sum()) / (1 + N_PERM),
                          (1 + (perm_rho <= rho).sum()) / (1 + N_PERM)],
}, index=["node_overlap_jaccard", "edge_overlap_jaccard", "spearman_eigenvector_common_words"])
print("\n--- COMPARISON OF THE TWO NETWORKS vs RANDOM RELABELLING ---")
print(comparison.round(4))
comparison.to_csv(OUTPUT_DIR / "network_comparison.csv")
 
 
# Figure 1: the 30 most central words, coloured by community (greedy modularity).
# Node size = eigenvector centrality. Community numbers (1 = largest) belong to each network separately,
# so the same colour in the two panels does NOT mean the same community.
N_SHOWN = 30   # only the most central words are drawn
 
 
def draw(G, nodes, ax, title):
    H = G.subgraph(nodes["word"].head(N_SHOWN))
    pos = nx.spring_layout(H, weight="weight", seed=SEED, k=1.5 / np.sqrt(len(H)))  # Fruchterman-Reingold
    ev_s = nodes.set_index("word")["eigenvector"]
    comm = nodes.set_index("word")["community"]
    palette = plt.get_cmap("tab10")
    colors = [palette((comm[w] - 1) % 10) for w in H.nodes]
    sizes = [60 + 900 * ev_s[w] / ev_s.max() for w in H.nodes]   # area, relative to this network
    nx.draw_networkx_edges(H, pos, ax=ax, width=0.8, alpha=0.3, edge_color="grey")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.85)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=8)
    ax.set_title(title)
    ax.axis("off")
 
 
titles = {"audience_higher": "Audience more generous than expected",
          "audience_lower": "Audience less generous than expected"}
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, g in zip(axes, GROUPS):
    draw(graphs[g], node_tables[g], ax, titles[g])
plt.tight_layout()
fig.savefig(FIG_DIR / "network_groups.pdf", bbox_inches="tight")
 
# Figure 2: degree distribution of the two networks (R: hist(degree) / degree_distribution)
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, g in zip(axes2, GROUPS):
    deg = node_tables[g]["degree"]
    ax.hist(deg, bins=np.arange(deg.min(), deg.max() + 2) - 0.5, color="grey", edgecolor="white")
    ax.set_title(titles[g])
    ax.set_xlabel("Degree")
axes2[0].set_ylabel("Number of words")
plt.tight_layout()
fig2.savefig(FIG_DIR / "network_degree_distribution.pdf", bbox_inches="tight")
plt.show()
 