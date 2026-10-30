# sentiment figure and summary table of the models for the final report.
# (Mean D by genre comes from the EDA script, the network figure from 08_network.py)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same colours as the other figures: blue = audience higher than expected, red = audience lower than expected
GROUPS = ["audience_higher", "audience_lower"]
COLORS = {"audience_higher": "#3b6fb6", "audience_lower": "#c0392b"}

# Sentiment of the critic reviews
s = pd.read_csv(PROCESSED_DIR / "movies_with_sentiment.csv")
test = pd.read_csv(OUTPUT_DIR / "sentiment_group_test.csv", index_col=0)["value"]
data = [s.loc[s["group_resid"] == g, "mean_review_sentiment"].dropna() for g in GROUPS]

fig, ax = plt.subplots(figsize=(6, 4.5))
box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black"))
for patch, g in zip(box["boxes"], GROUPS):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.7)

rng = np.random.default_rng(0)
for pos, d in enumerate(data, start=1):
    ax.scatter(pos + rng.uniform(-0.15, 0.15, len(d)), d, s=5, color="black", alpha=0.2, linewidths=0)

p_text = "p < 0.001" if test["p_value"] < 0.001 else f"p = {test['p_value']:.3f}"
ax.set_xticklabels(["Audience higher \nthan expected", "Audience lower \nthan expected"])
ax.set_ylabel("Mean VADER compound score per film")
ax.set_title("Mean sentiment of critic reviews, by group", fontsize=11)
ax.set_xlabel(f"Difference in means = {test['mean_diff']:.3f} (Welch t = {test['t_statistic']:.2f}, {p_text})",
              fontsize=9)
fig.savefig(FIG_DIR / "sentiment_groups.pdf", bbox_inches="tight")


# Nested regression models
nested = pd.read_csv(OUTPUT_DIR / "model2_nested_comparison.csv")
table = pd.DataFrame({
    "Model": ["M1  Structural controls",
              "M2  M1 + review sentiment",
              "M3  M2 + topic shares",
              "M4  M3 + Tomatometer (robustness)"],
    "R2": nested["R2"].round(3),
    "Adj_R2": nested["Adj_R2"].round(3),
    "AIC": nested["AIC"].round(1),
    "N": nested["N"].astype(int),
})
print(table.to_string(index=False))
table.to_csv(OUTPUT_DIR / "table_models.csv", index=False)

fig_t, ax_t = plt.subplots(figsize=(8, 2.2))
ax_t.axis("off")
tbl = ax_t.table(cellText=table.values, colLabels=["Model", "R²", "Adj. R²", "AIC", "N"],
                 loc="center", cellLoc="center", colWidths=[0.44, 0.14, 0.14, 0.16, 0.12])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("lightgrey")
    cell.set_linewidth(0.6)
    if row == 0:
        cell.set_facecolor("#eeeeee")
        cell.set_text_props(fontweight="bold")
fig_t.savefig(FIG_DIR / "table_models.pdf", bbox_inches="tight")

plt.show()