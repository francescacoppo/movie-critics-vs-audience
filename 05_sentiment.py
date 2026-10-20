from pathlib import Path

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_ind
from sklearn.metrics import roc_auc_score

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df_rev = pd.read_csv(PROCESSED_DIR / "reviews_sample.csv")
df_movies = pd.read_csv(PROCESSED_DIR / "movies_to_model.csv")
print(f"Loaded reviews: {len(df_rev)} | movies: {len(df_movies)}")

#  Sentiment of each review (VADER compound score, from -1 to +1)
sia = SentimentIntensityAnalyzer()
df_rev["sentiment_score"] = df_rev["review_content"].astype(str).apply(
    lambda x: sia.polarity_scores(x)["compound"]
)
df_rev["is_fresh"] = (df_rev["review_type"].str.lower() == "fresh").astype(int)
df_rev["predicted_fresh"] = (df_rev["sentiment_score"] > 0).astype(int)

# Review-level file (for topic and network scripts)
df_rev.to_csv(PROCESSED_DIR / "reviews_sentiment.csv", index=False)

# Validation against the fresh/rotten label
n_reviews = len(df_rev)
share_fresh = df_rev["is_fresh"].mean()
accuracy = (df_rev["predicted_fresh"] == df_rev["is_fresh"]).mean()
auc = roc_auc_score(df_rev["is_fresh"], df_rev["sentiment_score"])
share_zero = (df_rev["sentiment_score"] == 0).mean()

by_label = df_rev.groupby("review_type")["sentiment_score"].agg(
    count="count",
    mean="mean",
    median="median",
    std="std",
    share_zero=lambda x: (x == 0).mean(),
)

# conf.matrix
confusion = pd.crosstab(
    df_rev["is_fresh"].map({1: "fresh", 0: "rotten"}).rename("true_label"),
    df_rev["predicted_fresh"].map({1: "predicted_fresh", 0: "predicted_rotten"}).rename("prediction"),
    normalize="index",
)

print("\n--- VALIDATION: VADER vs fresh/rotten label ---")
print(f"Baseline (always 'fresh'): {share_fresh:.3f}")
print(f"Accuracy (threshold > 0): {accuracy:.3f}")
print(f"ROC AUC: {auc:.3f}")
print(f"Reviews with compound == 0: {share_zero:.2%}")

print("\nSentiment by label:")
print(by_label.round(3))

print("\nConfusion matrix (normalized):")
print(confusion.round(3))

# MOVIE LEVEL
movie_sentiment = (
    df_rev.groupby("rotten_tomatoes_link")
    .agg(mean_review_sentiment=("sentiment_score", "mean"),
         share_fresh_in_sample=("is_fresh", "mean"))
    .reset_index()
)
movie_corr_sentiment_vs_share_fresh = movie_sentiment["mean_review_sentiment"].corr(
    movie_sentiment["share_fresh_in_sample"]
)
print(f"\nCorrelation between mean sentiment and share of fresh reviews (movie level): "
      f"{movie_corr_sentiment_vs_share_fresh:.3f}")

df_merged = pd.merge(df_movies, movie_sentiment, on="rotten_tomatoes_link", how="inner")
print(f"Movies after the merge: {len(df_merged)}")


# Correlations with the gap measures 
corr_targets = ["tomatometer_rating", "audience_rating", "D", "audience_resid"]
corr_rows = []
for var in corr_targets:
    sub = df_merged[["mean_review_sentiment", var]].dropna()
    p_r, p_p = pearsonr(sub["mean_review_sentiment"], sub[var])
    s_r, s_p = spearmanr(sub["mean_review_sentiment"], sub[var])
    corr_rows.append({
        "variable": var,
        "pearson_r": p_r, "pearson_p_value": p_p,
        "spearman_rho": s_r, "spearman_p_value": s_p,
    })
corr_df = pd.DataFrame(corr_rows)
print("\n--- CORRELATIONS of mean sentiment with ---")
print(corr_df.to_string(index=False))

# Extreme groups (Tomatometer held fixed) 
group_stats = df_merged.groupby("group_resid")["mean_review_sentiment"].agg(
    ["count", "mean", "median", "std"]
)
print("\n--- MEAN SENTIMENT BY GROUP (group_resid) ---")
print(group_stats.round(4))

higher = df_merged.loc[df_merged["group_resid"] == "audience_higher", "mean_review_sentiment"].dropna()
lower = df_merged.loc[df_merged["group_resid"] == "audience_lower", "mean_review_sentiment"].dropna()

t_stat, p_val = ttest_ind(higher, lower, equal_var=False) # t-test

group_test = pd.Series({
    "n_audience_higher": len(higher),
    "n_audience_lower": len(lower),
    "mean_audience_higher": higher.mean(),
    "mean_audience_lower": lower.mean(),
    "mean_diff": higher.mean() - lower.mean(),
    "t_statistic": t_stat,
    "p_value": p_val,
})
print("\n--- TEST: audience_higher vs audience_lower (T-test) ---")
print(group_test.round(4).to_string())

# Save 
by_label.to_csv(OUTPUT_DIR / "sentiment_by_label.csv")
confusion.to_csv(OUTPUT_DIR / "sentiment_confusion_matrix.csv")
corr_df.to_csv(OUTPUT_DIR / "sentiment_correlations_detailed.csv", index=False)
group_stats.to_csv(OUTPUT_DIR / "sentiment_group_stats.csv")
group_test.to_csv(OUTPUT_DIR / "sentiment_group_test.csv", header=["value"])
df_merged.to_csv(PROCESSED_DIR / "movies_with_sentiment.csv", index=False)