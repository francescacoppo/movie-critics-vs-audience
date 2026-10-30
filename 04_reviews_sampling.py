from pathlib import Path
import pandas as pd

BASE_DIR = Path.cwd()
print("Base Directory:", BASE_DIR)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


reviews_path = RAW_DIR / "rotten_tomatoes_critic_reviews.csv"
df_reviews = pd.read_csv(reviews_path)
model_path = PROCESSED_DIR / "movies_to_model.csv"
movies_model = pd.read_csv(model_path)

print("\nDATASET SHAPE AND INFO") # (1130017, 8)
print(df_reviews.shape)
print(df_reviews.info())
    
print("\nNaN")
print(df_reviews.isna().sum()) 
    
print("\nDATASET HEAD")
print(df_reviews.head())

# Keep only films in movies_to_model.csv (rotten_tomatoes_link)
valid_links = set(movies_model["rotten_tomatoes_link"])
df_reviews = df_reviews[df_reviews["rotten_tomatoes_link"].isin(valid_links)].copy()
print(f"After the join with the model's films: {len(df_reviews)}")
    
# Delete empty reviews
df_reviews = df_reviews.dropna(subset=["review_content"])
df_reviews["review_content"] = df_reviews["review_content"].astype(str)
    
# Substitute characters Unicode U+2028 (Line Separator) and U+2029 (Paragraph Separator) with a space
df_reviews["review_content"] = (
    df_reviews["review_content"]
    .str.replace("\u2028", " ", regex=False)
    .str.replace("\u2029", " ", regex=False)
)
    
# Delete reviews if too short (ex. less than 5 words)
df_reviews["word_count"] = df_reviews["review_content"].str.split().str.len()
df_reviews = df_reviews[df_reviews["word_count"] >= 5] # 26000 more or less
print(f"After length filter: {len(df_reviews)}") 
    
# Delete duplicates (same review for the same film)
df_reviews = df_reviews.drop_duplicates(subset=["rotten_tomatoes_link", "review_content"]) # 68000 more or less
print(f"After removing duplicates: {len(df_reviews)}")
    

output_file = PROCESSED_DIR / "critic_reviews_clean.csv"
df_reviews.to_csv(output_file, index=False)


print("\nSAMPLING")
SAMPLE_SIZE = 25
RANDOM_STATE = 42
    
sampled_reviews = (
    df_reviews
    .sample(frac=1, random_state=RANDOM_STATE)             # shuffle all reviews (reproducible)
    .groupby("rotten_tomatoes_link")
    .head(SAMPLE_SIZE)                                     # keep the first 25 of each film
    .copy()
)

print(f"Total number of reviews in the sample: {len(sampled_reviews)}") #84925

counts = sampled_reviews.groupby("rotten_tomatoes_link").size()
all_movies = movies_model["rotten_tomatoes_link"]
counts = counts.reindex(all_movies, fill_value=0)
    
zero_reviews = (counts == 0).sum()
less_than_max = (counts < SAMPLE_SIZE).sum()
print(f"Films with 0 reviews in the sample: {zero_reviews}") #0
print(f"Film with less than {SAMPLE_SIZE} reviews in the sample: {less_than_max}") #0
    

# Select necessary columns: link, content and label fresh/rotten
cols_to_keep = ["rotten_tomatoes_link", "review_content", "review_type"]
reviews_sample = sampled_reviews[cols_to_keep].copy()
    
output_sample_path = PROCESSED_DIR / "reviews_sample.csv"
reviews_sample.to_csv(output_sample_path, index=False)
    

print("\nCONSISTENCY CHECK (Fresh Quota vs Tomatometer)")
# Binary column (1 if 'fresh', 0 otherwise)
sampled_reviews["is_fresh"] = (sampled_reviews["review_type"].str.lower() == "fresh").astype(int)
    
# % fresh reviews/film
sample_fresh_pct = sampled_reviews.groupby("rotten_tomatoes_link")["is_fresh"].mean() * 100
    

validation_df = movies_model.set_index("rotten_tomatoes_link")[["movie_title", "tomatometer_rating"]].copy()
validation_df["sample_fresh_pct"] = sample_fresh_pct
validation_df = validation_df.dropna(subset=["sample_fresh_pct"])
    

corr = validation_df["tomatometer_rating"].corr(validation_df["sample_fresh_pct"])
print(f"Pearson correlation between Tomatometer and % Fresh in the sample: {corr:.4f}")
# 96%
# The share of positive reviews tracks the Tomatometer almost perfectly - sample is not biased

print("\nGlimpse of comparison (5 lines):")
print(validation_df[["movie_title", "tomatometer_rating", "sample_fresh_pct"]].head())
    
    