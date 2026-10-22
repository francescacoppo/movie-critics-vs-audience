#sentiment figure and summary table of the models for the final report.
#(Mean D by genre comes from the EDA script, the network figure from 08_network.py)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same colours as the other figures: blue = audience higher than critics, red = critics higher than audience
GROUPS = ["audience_higher", "audience_lower"]
COLORS = {"audience_higher": "#3b6fb6", "audience_lower": "#c0392b"}


# Figure: sentiment of the critic reviews
s = pd.read_csv(PROCESSED_DIR / "movies_with_sentiment.csv")
test = pd.read_csv(OUTPUT_DIR / "sentiment_group_test.csv", index_col=0)["value"]
data = [s.loc[s["group_resid"] == g, "mean_review_sentiment"].dropna() for g in GROUPS]

fig, ax = plt.subplots(figsize=(6, 4.5))
box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black"))
for patch, g in zip(box["boxes"], GROUPS):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.7)

# single films on top of the boxes, so the box does not hide the distribution
rng = np.random.default_rng(0)
for pos, d in enumerate(data, start=1):
    ax.scatter(pos + rng.uniform(-0.15, 0.15, len(d)), d, s=5, color="black", alpha=0.2, linewidths=0)

p_text = "p < 0.001" if test["p_value"] < 0.001 else f"p = {test['p_value']:.3f}"
ax.set_xticklabels(["Audience higher\nthan critics", "Critics higher\nthan audience"])
ax.set_ylabel("Mean VADER compound score per film")
ax.set_title(f"Sentiment of critic reviews\nDifference in means = {test['mean_diff']:.3f} "
             f"(Welch t = {test['t_statistic']:.2f}, {p_text})", fontsize=10)
fig.savefig(FIG_DIR / "sentiment_groups.pdf", bbox_inches="tight")


# Table: nested regression models
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

fig_t, ax_t = plt.subplots(figsize=(8, 1.8))
ax_t.axis("off")
ax_t.table(cellText=table.values, colLabels=table.columns, loc="center")
fig_t.savefig(FIG_DIR / "table_models.pdf", bbox_inches="tight")

plt.show()