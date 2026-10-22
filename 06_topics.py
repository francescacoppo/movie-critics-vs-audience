from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#  settings 
N_TOPICS = 5
SEED = 42
N_TOP_WORDS = 12
CHECK_STABILITY = True   

GENRE_COLS = [
    "action_adventure", "animation", "comedy", "documentary", "drama", "horror",
    "kids_family", "musical_performing_arts", "mystery_suspense", "romance",
    "science_fiction_fantasy", "art_house_international",
]


movies_df = pd.read_csv(PROCESSED_DIR / "movies_with_sentiment.csv")
reviews_df = pd.read_csv(PROCESSED_DIR / "reviews_sentiment.csv")
print(f"Movies: {len(movies_df)} | Reviews: {len(reviews_df)}")

# text preparation
texts = (
    reviews_df["review_content"]
    .fillna("")
    .astype(str)
    .str.replace(r"sci[- ]?fi", "scifi", case=False, regex=True) # merge "sci" and "fi" in "sci-fi"
    .str.replace(r"\[full review in \w+\]", " ", case=False, regex=True)  # to correct “[Full review in Spanish]”
)

# Stop words: standard English list + empty words (movie, film...) + pieces of contractions
extra_stop_words = [
    "movie", "movies", "film", "films", "review", "reviews", "just", "like", "really",
    "isn", "don", "doesn", "didn", "wasn", "aren", "haven", "hasn", "hadn",
    "wouldn", "shouldn", "couldn", "ll", "ve", "re",
    # Genre names removed, so that topics describe aspects of the review and not the genre of the film.
    "comedy", "horror", "thriller", "action", "drama", "romantic", "romance", "scifi", "documentary", "animated", "animation"
]
stop_words = list(ENGLISH_STOP_WORDS.union(extra_stop_words))

# ----------  Document-term matrix (ALL reviews) ----------
vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=0.4, min_df=10, stop_words=stop_words)
dtm = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()
print(f"Document-term matrix: {dtm.shape[0]} reviews x {dtm.shape[1]} terms")


def fit_lda(seed):
    model = LatentDirichletAllocation(n_components=N_TOPICS, random_state=seed, n_jobs=-1)
    doc_topic = model.fit_transform(dtm)   # rows = reviews, columns = topic shares
    return model, doc_topic


print(f"\nFitting LDA (k = {N_TOPICS}, seed = {SEED})...")
lda, doc_topic = fit_lda(SEED)

# ----------  Top words of each topic ----------
print("\n--- TOP WORDS PER TOPIC ---")
keywords = []
for t, weights in enumerate(lda.components_):
    top_words = [feature_names[i] for i in weights.argsort()[::-1][:N_TOP_WORDS]]
    keywords.append({"topic": f"topic_{t + 1}", "top_words": ", ".join(top_words)})
    print(f"topic_{t + 1}: {', '.join(top_words)}")
pd.DataFrame(keywords).to_csv(OUTPUT_DIR / "lda_topic_keywords.csv", index=False)  # topic words are not specific

# Stability check: same model with another seed, topics matched by cosine similarity
lda_b, _ = fit_lda(SEED + 1)
best_match = cosine_similarity(lda.components_, lda_b.components_).max(axis=1)
print("Stability check:", best_match.round(2))

# ----------  Topic shares per movie (ALL movies) ----------
topic_cols = [f"topic_{t + 1}" for t in range(N_TOPICS)]
review_topics = pd.DataFrame(doc_topic, columns=topic_cols)
review_topics["rotten_tomatoes_link"] = reviews_df["rotten_tomatoes_link"].to_numpy()
movie_topics = review_topics.groupby("rotten_tomatoes_link")[topic_cols].mean().reset_index()

movie_topics.to_csv(PROCESSED_DIR / "movies_topic_shares.csv", index=False)
print(f"\nTopic shares saved for {len(movie_topics)} movies")

df = movies_df.merge(movie_topics, on="rotten_tomatoes_link", how="inner")

# ---------- Genre composition of the two extreme groups ----------
extreme = df[df["group_resid"].isin(["audience_higher", "audience_lower"])]
genre_comp = (extreme.groupby("group_resid")[GENRE_COLS].mean() * 100).T
genre_comp["all_movies"] = df[GENRE_COLS].mean() * 100
genre_comp["diff_higher_minus_lower"] = genre_comp["audience_higher"] - genre_comp["audience_lower"]
print("\n--- GENRE COMPOSITION OF THE EXTREME GROUPS (% of movies with the genre) ---")
print(genre_comp.round(1))


# ---------- How much do the topics just reflect genre? ----------
topic_genre_corr = pd.DataFrame(
    {g: df[topic_cols].corrwith(df[g]) for g in GENRE_COLS}
)
print("\n--- CORRELATION BETWEEN TOPIC SHARES AND GENRE (all movies) ---")
print(topic_genre_corr.round(2))
# ----------  Comparison of topic shares between the two groups ----------
higher = extreme[extreme["group_resid"] == "audience_higher"]
lower = extreme[extreme["group_resid"] == "audience_lower"]

rows = []
for col in topic_cols:
    u, p = mannwhitneyu(higher[col], lower[col], alternative="two-sided")   # Mann-Whitney: non parametric test, more accurate for shares
    rows.append({
        "topic": col,
        "mean_diff": higher[col].mean() - lower[col].mean(),
        "prob_higher_gt_lower": u / (len(higher) * len(lower)),   # 0.5 = no difference
        "p_value": p,
    })

stats_df = pd.DataFrame(rows)
#stats_df["p_value_holm"] = multipletests(stats_df["p_value"], method="holm")[1]   no significance (= 1.0)

print(f"\n--- TOPIC SHARES: audience_higher ({len(higher)}) vs audience_lower ({len(lower)}) ---")  # no difference, p-value not significant
print(stats_df.round(4).to_string(index=False))
stats_df.to_csv(OUTPUT_DIR / "lda_group_comparison_stats.csv", index=False)


