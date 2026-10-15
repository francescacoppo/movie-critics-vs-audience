from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. Load data ----------
df_rev = pd.read_csv(PROCESSED_DIR / "reviews_sample.csv")
df_movies = pd.read_csv(PROCESSED_DIR / "movies_to_model.csv")
print(f"Loaded reviews: {len(df_rev)} | movies: {len(df_movies)}")

# ---------- 2. Sentiment of each review (VADER compound score, from -1 to +1) ----------
sia = SentimentIntensityAnalyzer()
df_rev["sentiment_score"] = df_rev["review_content"].astype(str).apply(
    lambda x: sia.polarity_scores(x)["compound"]
)
df_rev["is_fresh"] = (df_rev["review_type"].str.lower() == "fresh").astype(int)
df_rev["predicted_fresh"] = (df_rev["sentiment_score"] > 0).astype(int)

# Review-level file, reused by the topic and network scripts
df_rev.to_csv(PROCESSED_DIR / "reviews_sentiment.csv", index=False)

# ---------- 3. Validation against the fresh/rotten label ----------
metrics = {}
metrics["n_reviews"] = len(df_rev)
metrics["share_fresh"] = df_rev["is_fresh"].mean()   # accuracy of always answering "fresh"
metrics["accuracy_threshold_0"] = (df_rev["predicted_fresh"] == df_rev["is_fresh"]).mean()
metrics["roc_auc"] = roc_auc_score(df_rev["is_fresh"], df_rev["sentiment_score"])
metrics["share_compound_zero_total"] = (df_rev["sentiment_score"] == 0).mean()

by_label = df_rev.groupby("review_type")["sentiment_score"].agg(
    count="count",
    mean="mean",
    median="median",
    std="std",
    share_zero=lambda x: (x == 0).mean(),
)
for label, row in by_label.iterrows():
    metrics[f"mean_sentiment_{label.lower()}"] = row["mean"]
    metrics[f"share_compound_zero_{label.lower()}"] = row["share_zero"]

confusion = pd.crosstab(
    df_rev["is_fresh"].map({1: "fresh", 0: "rotten"}).rename("true_label"),
    df_rev["predicted_fresh"].map({1: "predicted_fresh", 0: "predicted_rotten"}).rename("prediction"),
    normalize="index",
)

print("\n--- VALIDATION: VADER vs fresh/rotten label ---")
print(f"Baseline (always 'fresh'): {metrics['share_fresh']:.3f}")
print(f"Accuracy (compound > 0 = fresh): {metrics['accuracy_threshold_0']:.3f}")
print(f"ROC AUC: {metrics['roc_auc']:.3f}")
print(f"Reviews with compound == 0: {metrics['share_compound_zero_total']:.3%}")
print("\nSentiment by label:")
print(by_label.round(3))
print("\nConfusion matrix (rows sum to 1):")
print(confusion.round(3))

# ---------- 4. Movie level ----------
movie_sentiment = (
    df_rev.groupby("rotten_tomatoes_link")
    .agg(mean_review_sentiment=("sentiment_score", "mean"),
         share_fresh_in_sample=("is_fresh", "mean"))
    .reset_index()
)
metrics["movie_corr_sentiment_vs_share_fresh"] = movie_sentiment["mean_review_sentiment"].corr(
    movie_sentiment["share_fresh_in_sample"]
)
print(f"\nCorrelation between mean sentiment and share of fresh reviews (movie level): "
      f"{metrics['movie_corr_sentiment_vs_share_fresh']:.3f}")

df_merged = pd.merge(df_movies, movie_sentiment, on="rotten_tomatoes_link", how="inner")
print(f"Movies after the merge: {len(df_merged)}")
assert len(df_merged) == len(df_movies), "Some movies lost in the merge"

# ---------- 5. Correlations with the gap measures ----------
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

# ---------- 6. Extreme groups (Tomatometer held fixed) ----------
group_stats = df_merged.groupby("group_resid")["mean_review_sentiment"].agg(
    ["count", "mean", "median", "std"]
)
print("\n--- MEAN SENTIMENT BY GROUP (group_resid) ---")
print(group_stats.round(4))

higher = df_merged.loc[df_merged["group_resid"] == "audience_higher", "mean_review_sentiment"].dropna()
lower = df_merged.loc[df_merged["group_resid"] == "audience_lower", "mean_review_sentiment"].dropna()

u_stat, p_val = mannwhitneyu(higher, lower, alternative="two-sided")
diff = higher.mean() - lower.mean()
se = np.sqrt(higher.var(ddof=1) / len(higher) + lower.var(ddof=1) / len(lower))
pooled_sd = np.sqrt(((len(higher) - 1) * higher.var(ddof=1) + (len(lower) - 1) * lower.var(ddof=1))
                    / (len(higher) + len(lower) - 2))

group_test = pd.Series({
    "n_audience_higher": len(higher),
    "n_audience_lower": len(lower),
    "mean_audience_higher": higher.mean(),
    "mean_audience_lower": lower.mean(),
    "mean_diff_higher_minus_lower": diff,
    "mean_diff_ci95_low": diff - 1.96 * se,    # approximate 95% confidence interval
    "mean_diff_ci95_high": diff + 1.96 * se,
    "cohen_d": diff / pooled_sd,
    "prob_higher_gt_lower": u_stat / (len(higher) * len(lower)),   # 0.5 = no difference
    "mannwhitney_U": u_stat,
    "mannwhitney_p_value": p_val,
})
print("\n--- TEST: audience_higher vs audience_lower ---")
print(group_test.round(4).to_string())

# ---------- 7. Save everything needed for the report ----------
pd.Series(metrics).to_csv(OUTPUT_DIR / "sentiment_validation.csv", header=["value"])
by_label.to_csv(OUTPUT_DIR / "sentiment_by_label.csv")
confusion.to_csv(OUTPUT_DIR / "sentiment_confusion_matrix.csv")
corr_df.to_csv(OUTPUT_DIR / "sentiment_correlations_detailed.csv", index=False)
group_stats.to_csv(OUTPUT_DIR / "sentiment_group_stats.csv")
group_test.to_csv(OUTPUT_DIR / "sentiment_group_test.csv", header=["value"])
df_merged.to_csv(PROCESSED_DIR / "movies_with_sentiment.csv", index=False)
