from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

movies_path = PROCESSED_DIR / "movies_with_sentiment.csv"
topics_path = PROCESSED_DIR / "movies_topic_shares.csv"

df_model = (
    pd.read_csv(movies_path)
    .merge(pd.read_csv(topics_path), on="rotten_tomatoes_link", how="inner")
)

print(f"Movies after merge: {len(df_model)}")


# Standardization of sentiment and topic shares (mean=0, sd=1) 
topic_cols = [c for c in df_model.columns if c.startswith("topic_")]
cols_to_std = ["mean_review_sentiment"] + topic_cols
for col in cols_to_std:
    df_model[f"{col}_std"] = (df_model[col] - df_model[col].mean()) / df_model[col].std()

std_sentiment_col = "mean_review_sentiment_std"
std_topic_cols = [f"{c}_std" for c in topic_cols]

# Reference levels to avoid perfect collinearity (-1 topic)
period_ref = "2010-2020"
rating_ref = "R"

GENRE_COLS = [
    "action_adventure", "animation", "comedy", "documentary", "drama", "horror",
    "kids_family", "musical_performing_arts", "mystery_suspense", "romance",
    "science_fiction_fantasy", "art_house_international",
]

genre_terms = " + ".join(GENRE_COLS)

df_model["runtime_10"] = df_model["runtime"] / 10

base_formula = (
    f"audience_resid ~ {genre_terms} "
    f"+ C(period, Treatment(reference='{period_ref}')) "
    f"+ runtime_10 "
    f"+ C(content_rating, Treatment(reference='{rating_ref}'))"
)

# Nested Models Estimation
model_base = smf.ols(base_formula, data=df_model).fit(cov_type="HC3") # only structural variable

formula_sent = base_formula + f" + {std_sentiment_col}"  
model_sent = smf.ols(formula_sent, data=df_model).fit(cov_type="HC3") # base + sentiment

active_topics = std_topic_cols[1:]  

topics_terms = " + ".join(active_topics)
formula_full = formula_sent + f" + {topics_terms}"
model_full = smf.ols(formula_full, data=df_model).fit(cov_type="HC3") # base + sentiment + topics

# Nested comparison table
summary_df = pd.DataFrame({
    "Model": ["Base (Structural)", "Base + Sentiment", "Base + Sentiment + Topics"],
    "R2": [model_base.rsquared, model_sent.rsquared, model_full.rsquared],
    "Adj_R2": [model_base.rsquared_adj, model_sent.rsquared_adj, model_full.rsquared_adj],
    "AIC": [model_base.aic, model_sent.aic, model_full.aic],
    "N": [int(model_base.nobs), int(model_sent.nobs), int(model_full.nobs)]
})
print("\n--- NESTED MODELS COMPARISON ---")
print(summary_df.to_string(index=False))

# Wald tests for text blocks (significance sentiment/topic)
def wald_block(model, terms):
    res = model.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
    return {"chi2": float(res.statistic), "df": int(res.df_denom), "p_value": float(res.pvalue)}  # p < 0,001

text_vars = [std_sentiment_col] + active_topics
wald_tests_df = pd.DataFrame({
    "sentiment (Base+Sentiment model)": wald_block(model_sent, [std_sentiment_col]),
    "topics only (full model)": wald_block(model_full, active_topics),
    "all text variables (full model)": wald_block(model_full, text_vars),
}).T
print("\n--- WALD TESTS ---")
print(wald_tests_df.round(4))
wald_tests_df.to_csv(OUTPUT_DIR / "model2_wald_tests.csv")
# Under the null hypothesis, the Wald statistic follows a chi-squared (χ²) distribution with df degrees of freedom 
# (where df is the number of restrictions or parameters being tested).

# Robustness check (Tomatometer & Tomatometer^2) 
formula_rob = formula_full + " + tomatometer_rating + I(tomatometer_rating ** 2)" # non-linearity
model_rob = smf.ols(formula_rob, data=df_model).fit(cov_type="HC3")

# Update nested comparison table to include robust model AIC and N
robust_row = pd.DataFrame({
    "Model": ["Robust (Tomatometer + Sq)"],
    "R2": [model_rob.rsquared],
    "Adj_R2": [model_rob.rsquared_adj],
    "AIC": [model_rob.aic],
    "N": [int(model_rob.nobs)]
})
summary_full_df = pd.concat([summary_df, robust_row], ignore_index=True)
summary_full_df.to_csv(OUTPUT_DIR / "model2_nested_comparison.csv", index=False)

print("\n--- ROBUSTNESS CHECK ---")
print(f"Robust Model R2: {model_rob.rsquared:.4f} | Adj R2: {model_rob.rsquared_adj:.4f} | AIC: {model_rob.aic:.2f}")

# Coefficient tables for text variables
def coef_table(model):
    ci = model.conf_int()
    return pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "se_HC3": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values
    })

text_terms = [std_sentiment_col] + active_topics
tab_full = coef_table(model_full)
tab_full.to_csv(OUTPUT_DIR / "model2_full_coefficients.csv", index=False)
print("\n--- Text variables, full model ---")
print(tab_full[tab_full["term"].isin(text_terms)].round(3).to_string(index=False))

tab_rob = coef_table(model_rob)    # robustness
print("\n--- Text variables, robust model (with Tomatometer and its square) ---")
print(tab_rob[tab_rob["term"].isin(text_terms)].round(3).to_string(index=False))

# VIF - multicollinearity check
X_matrix = pd.DataFrame(model_full.model.exog, columns=model_full.model.exog_names)
vif = {c: variance_inflation_factor(X_matrix.to_numpy(dtype=float), i)
       for i, c in enumerate(X_matrix.columns) if c != "const"}
print("\n--- Largest VIF (full model) ---")
print(pd.Series(vif).sort_values(ascending=False).head(5).round(2))  # max vif 2,47 --> OK (limit 5/10)