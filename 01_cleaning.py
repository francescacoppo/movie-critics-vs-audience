import pandas as pd
from pathlib import Path  # for reproducibility

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

df = pd.read_csv(RAW_DIR / "rotten_tomatoes_movies.csv")

print(df.shape)  #17712 observations, 22 variables
df.info()
print(df.isna().sum())
print(df.duplicated().sum()) # no duplicated

# NA CLEANING
df = df.dropna(
    subset=[
        "tomatometer_rating",
        "audience_rating",
        "tomatometer_count",
        "audience_count"
    ]
)

print("After deleting NaN rating/count:", len(df)) #17406

# THRESHOLD 100 RATINGS 
df = df[
    (df["tomatometer_count"] >= 100) &
    (df["audience_count"] >= 100)
]

print("After threshold >=100 / >=100:", len(df))   #3428

# DATE HANDLING
print(df[[
    "runtime",
    "genres",
    "original_release_date"
]].dtypes)

df["original_release_date"] = pd.to_datetime(
    df["original_release_date"],
    errors="coerce"
)

print("\nNaN after date conversion:")
print(df[[
    "runtime",   #14
    "genres",    #0
    "original_release_date" #16
]].isna().sum())

df = df.dropna(
    subset=[
        "runtime",
        "genres",
        "original_release_date"
    ]
)

print("After deleting NaN runtime/genres/data:", len(df)) #3398

# D = Audience rating - Tomatometer rating

df["D"] = df["audience_rating"] - df["tomatometer_rating"]

print("\nD calculated")
print(df[[
    "movie_title",
    "tomatometer_rating",
    "audience_rating",
    "D"
]].head())

# YEAR EXTRACTION
df["year"] = df["original_release_date"].dt.year


cols = ["rotten_tomatoes_link", "movie_title", "genres", "year", "runtime",
        "content_rating", "tomatometer_rating", "audience_rating", "D",
        "tomatometer_count", "audience_count"]
df = df[cols]
print(df.shape)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(
    PROCESSED_DIR / "movies_clean.csv",
    index=False
)

print("Dataset saved in:", PROCESSED_DIR / "movies_clean.csv")

