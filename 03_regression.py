# D ~ 12 dummys genre + period + runtime/10 + content_rating

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
#from statsmodels.tools.tools import add_constant

BASE_DIR = Path.cwd()
print(BASE_DIR) 

PROCESSED_DIR = BASE_DIR / "data" / "processed"
print(PROCESSED_DIR)

df = pd.read_csv(PROCESSED_DIR / "movies_to_model.csv")

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENRE_COLS = [
    "action_adventure",
    "animation",
    "comedy",
    "documentary",
    "drama",
    "horror",
    "kids_family",
    "musical_performing_arts",
    "mystery_suspense",
    "romance",
    "science_fiction_fantasy",
    "art_house_international",
]

PERIOD_REFERENCE = "2010-2020"
RATING_REFERENCE = "R"
# regression benchmarks

def make_formula(genre_cols):
    genre_terms = " + ".join(genre_cols)
    return (
        f"D ~ {genre_terms} "
        f"+ C(period, Treatment(reference='{PERIOD_REFERENCE}')) "
        f"+ runtime_10 "
        f"+ C(content_rating, Treatment(reference='{RATING_REFERENCE}'))"
    )


df["D"] = pd.to_numeric(df["D"], errors="coerce")
df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
df["runtime_10"] = df["runtime"] / 10
for col in GENRE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["period"] = df["period"].astype("string")
df["content_rating"] = df["content_rating"].astype("string").str.strip()

model_cols = ["D", "runtime_10", "period", "content_rating", *GENRE_COLS]
model_df = df[model_cols].dropna().copy()
model_df = model_df[
    model_df["period"].isin(["<2000", "2000s", "2010-2020"])
        & model_df["content_rating"].isin(["R", "PG-13", "PG", "G", "NR"])
    ].copy()

formula = make_formula(GENRE_COLS)

model1 = smf.ols(formula, data=model_df).fit(cov_type="HC3")


# Summary table and coefficients
with (OUTPUT_DIR / "model1_summary.txt").open("w", encoding="utf-8") as f:
    f.write("MODEL 1 — STRUCTURAL\n")
    f.write(f"Formula: {formula}\n")
    f.write(f"Used observations: {int(model1.nobs)} of {len(df)} lines\n")
    f.write("Robust standard errors: HC3\n\n") # handle heteroschedasticity
    f.write(model1.summary().as_text())
    f.write("\n\nNote: runtime_10 is runtime/10; the coefficient is the effect associated to ten minutes more.\n")


ci = model1.conf_int()
coef = pd.DataFrame({
    "term": model1.params.index, #estiamted variable
    "coefficient": model1.params.values,
    "std_error_HC3": model1.bse.values,
    "p_value": model1.pvalues.values,
    "ci_95_low": ci[0].values,
    "ci_95_high": ci[1].values,
})
coef.to_csv(OUTPUT_DIR / "model1_coefficients.csv", index=False)

# Residual diagnostics 
fig = plt.figure(figsize=(8, 6))
qqplot(model1.resid, line="45", fit=True, ax=plt.gca())
plt.title("Model 1 — Q-Q plot residuals")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "model1_qqplot.pdf", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(model1.fittedvalues, model1.resid, alpha=0.45, s=18)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Estimated values")
ax.set_ylabel("Residuals")
ax.set_title("Model 1 — Residuals vs estimated values")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "model1_residuals.pdf")
plt.close(fig)

# VIF: computed on the numerical feature matrix
X = pd.DataFrame(model1.model.exog, columns=model1.model.exog_names)
vif_rows = []
for i, col in enumerate(X.columns):
    if col == "const":
        continue
    try:
        value = variance_inflation_factor(X.to_numpy(dtype=float), i)
    except Exception:
        value = float("nan")
    vif_rows.append({"variable": col, "VIF": value})
pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).to_csv(
        OUTPUT_DIR / "model1_vif.csv", index=False
    )

# ROBUSTNESS: leave out film with runtime over 99 percentile of the sample
runtime_cutoff = model_df["runtime_10"].quantile(0.99)
shorter_df = model_df.loc[model_df["runtime_10"] <= runtime_cutoff].copy()

sensitivity = smf.ols(formula, data=shorter_df).fit(cov_type="HC3")
with (OUTPUT_DIR / "model1_robustness_no_long_runtime.txt").open("w", encoding="utf-8") as f:
    f.write("ROBUSTNESS\n")
    f.write(f"Runtime threshold: {runtime_cutoff * 10:.1f} minutes\n")
    f.write(f"Observations: {int(sensitivity.nobs)} (main model: {int(model1.nobs)})\n")
    f.write(f"R²: {sensitivity.rsquared:.4f}\n")
    f.write(f"Adjusted R²: {sensitivity.rsquared_adj:.4f}\n\n")
    f.write(sensitivity.summary().as_text())

print("MODEL 1 completed")
print(f"Used observations: {int(model1.nobs)}")
print(f"R² = {model1.rsquared:.4f}")
print(f"Adjusted R² = {model1.rsquared_adj:.4f}")
print(f"Formula: {formula}")

