from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
reviews_sample_path = PROCESSED_DIR / "reviews_sample.csv"
movies_model_path = PROCESSED_DIR / "movies_to_model.csv"

df_rev = pd.read_csv(reviews_sample_path)
df_movies = pd.read_csv(movies_model_path)

print(f"Loaded reviews: {len(df_rev)}") #84925

# SENTIMENT ANALYSIS
sia = SentimentIntensityAnalyzer()
df_rev["sentiment_score"] = df_rev["review_content"].astype(str).apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

df_rev["is_fresh"] = (df_rev["review_type"].str.lower() == "fresh").astype(int)

# Saving the file at the review level (for reuse in LDA and Network Analysis)
reviews_sentiment_path = PROCESSED_DIR / "reviews_sentiment.csv"
df_rev.to_csv(reviews_sentiment_path, index=False)

# Agreement with label and AUC
df_rev["predicted_fresh"] = (df_rev["sentiment_score"] > 0).astype(int)
exact_match_pct = (df_rev["predicted_fresh"] == df_rev["is_fresh"]).mean() * 100
print(f"\n--- EXACT MATCH (compound > 0 vs fresh) ---")
print(f"Accuracy based on threshold 0: {exact_match_pct:.2f}%")

auc_score = roc_auc_score(df_rev["is_fresh"], df_rev["sentiment_score"])
print(f"ROC AUC Score (compound vs is_fresh): {auc_score:.4f}")

# Proportion of reviews with compound == 0 (total and by label)
zero_compound_total = (df_rev["sentiment_score"] == 0).mean() * 100
zero_compound_by_label = df_rev.groupby("review_type")["sentiment_score"].apply(lambda x: (x == 0).mean() * 100)
print(f"\n--- PROPORTION REVIEWS WITH COMPOUND == 0 ---")
print(f"Total neutral stake (0.0): {zero_compound_total:.2f}%")
print("Share by review type:\n", zero_compound_by_label)

# Aggregation of average sentiment at the film level
movie_sentiment = df_rev.groupby("rotten_tomatoes_link")["sentiment_score"].mean().reset_index()
movie_sentiment.rename(columns={"sentiment_score": "mean_review_sentiment"}, inplace=True)

# Rigorous control of the merge (verification of the cardinality at 3,397 films)
df_merged = pd.merge(df_movies, movie_sentiment, on="rotten_tomatoes_link", how="inner")
print(f"\n--- CHECK THE MERGE ---")
print(f"Total number of film after the merge: {len(df_merged)}")

# Correlations (Pearson and Spearman) with D, audience_resid, Tomatometer, and Audience Score
corr_targets = ["tomatometer_rating", "audience_rating", "D", "audience_resid"]
corr_results = []
for var in corr_targets:
    valid_sub = df_merged[["mean_review_sentiment", var]].dropna()
    p_corr, p_val = pearsonr(valid_sub["mean_review_sentiment"], valid_sub[var])
    s_corr, s_val = spearmanr(valid_sub["mean_review_sentiment"], valid_sub[var])
    corr_results.append({
        "variable": var,
        "pearson_r": p_corr,
        "pearson_p_value": p_val,
        "spearman_rho": s_corr,
        "spearman_p_value": s_val
    })

corr_df = pd.DataFrame(corr_results)
print(f"\n--- CORRELATIONS ANALYSIS ---")
print(corr_df.to_string(index=False))

# audience_higher vs audience_lower based on group_resid
print(f"\n--- COMPARISON BETWEEN EXTREME GROUPS (group_resid) ---")
group_stats = df_merged.groupby("group_resid")["mean_review_sentiment"].agg(["count", "mean", "median", "std"])
print(group_stats)

higher_vals = df_merged.loc[df_merged["group_resid"] == "audience_higher", "mean_review_sentiment"].dropna()
lower_vals = df_merged.loc[df_merged["group_resid"] == "audience_lower", "mean_review_sentiment"].dropna()

if len(higher_vals) > 0 and len(lower_vals) > 0:
    mwu_stat, mwu_p = mannwhitneyu(higher_vals, lower_vals, alternative="two-sided")
    print(f"Mann-Whitney U test (audience_higher vs audience_lower): statistic = {mwu_stat:.2f}, p-value = {mwu_p:.4e}")


df_merged.to_csv(PROCESSED_DIR / "movies_with_sentiment.csv", index=False)
corr_df.to_csv(OUTPUT_DIR / "sentiment_correlations_detailed.csv", index=False)