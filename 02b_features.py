from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
print(BASE_DIR) 

PROCESSED_DIR = BASE_DIR / "data" / "processed"
print(PROCESSED_DIR)

df = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")


# 12 categories - standardize project features' names
GENRE_MAP = {
    "Action & Adventure": "action_adventure",
    "Animation": "animation",
    "Comedy": "comedy",
    "Documentary": "documentary",
    "Drama": "drama",
    "Horror": "horror",
    "Kids & Family": "kids_family",
    "Musical & Performing Arts": "musical_performing_arts",
    "Mystery & Suspense": "mystery_suspense",
    "Romance": "romance",
    "Science Fiction & Fantasy": "science_fiction_fantasy",
    "Art House & International": "art_house_international",
}

genre_dummies = df["genres"].str.get_dummies(sep=", ")
genre_dummies = genre_dummies[list(GENRE_MAP.keys())].rename(columns=GENRE_MAP)
df = pd.concat([df, genre_dummies], axis=1)

# Films without genre
genre_cols = list(GENRE_MAP.values()) #extract columns names
df = df[df[genre_cols].sum(axis=1) > 0].copy()

# Release period (3 categories)
df["period"] = pd.NA
df.loc[df["year"] < 2000, "period"] = "<2000"
df.loc[df["year"].between(2000, 2009), "period"] = "2000s"
df.loc[df["year"].between(2010, 2020), "period"] = "2010-2020"

df["period"] = pd.Categorical(
    df["period"],
    categories=["<2000", "2000s", "2010-2020"],
    ordered=True,
)

# Merge NC17 (8 films) in R 
df["content_rating"] = df["content_rating"].replace({"NC17": "R", "NC-17": "R"})


print(df[["D", "tomatometer_rating"]].corr())


# 1) Audience score expected from the Tomatometer (quadratic curve, because the relation is not exactly linear: ceiling effect at the extremes)
coefs = np.polyfit(df["tomatometer_rating"], df["audience_rating"], deg=2)
df["audience_expected"] = np.polyval(coefs, df["tomatometer_rating"])

# 2) Residual: positive = audience rated the movie higher than expected
df["audience_resid"] = df["audience_rating"] - df["audience_expected"]

# 3) Thresholds computed WITHIN Tomatometer bins, so that both extreme groups have the same Tomatometer distribution
N_BINS = 5
df["tomato_bin"] = pd.qcut(df["tomatometer_rating"], q=N_BINS, duplicates="drop")
by_bin = df.groupby("tomato_bin", observed=True)["audience_resid"]
low = by_bin.transform(lambda s: s.quantile(0.10))
high = by_bin.transform(lambda s: s.quantile(0.90))

df["group_resid"] = "middle"
df.loc[df["audience_resid"] <= low, "group_resid"] = "audience_lower"
df.loc[df["audience_resid"] >= high, "group_resid"] = "audience_higher"
df["tomato_bin"] = df["tomato_bin"].astype(str)   

# Checks 
print(df.groupby("group_resid")["tomatometer_rating"].agg(["mean", "median", "std", "count"]).round(1))
print(df.groupby("group_resid")["audience_resid"].agg(["min", "max"]).round(1))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(
    PROCESSED_DIR / "movies_to_model.csv",
    index=False
)

print("Dataset saved in:", PROCESSED_DIR / "movies_to_model.csv")

