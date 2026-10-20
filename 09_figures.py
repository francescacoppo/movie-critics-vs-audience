"""
09_figures.py - figures and table of the final report (the project allows at most 4 in total).

  Figure 1  fig1_groups.pdf / .png        (a) how the two extreme groups are defined
                                          (b) mean sentiment of the critic reviews in the two groups
  Table 1   table1_nested_models.csv/.tex/.pdf   nested regression models of audience_resid
  Figure 2  fig2_coefficients.pdf / .png  coefficients of the full Model 2 (95% CI, HC3 robust SE)
  Figure 3  figures/network_groups.pdf    produced by 08_network.py (not redrawn here)

Colour convention (same as 08_network.py):
  blue = audience_higher (audience rates the film higher than expected from the Tomatometer)
  red  = audience_lower  (audience rates it lower than expected)
  grey = all other films

Inputs (all produced by the previous scripts):
  data/processed/movies_to_model.csv           (02b_features.py)
  data/processed/movies_with_sentiment.csv     (05_sentiment.py)   -> panel (b) of Figure 1
  results/sentiment_group_test.csv             (05_sentiment.py: mean_diff, ci95_low, ci95_high, t_statistic, p_value)
  results/model2_full_coefficients.csv         (07_model2_regression.py)
  results/model2_nested_comparison.csv         (07_model2_regression.py)
  results/model2_wald_tests.csv                (07_model2_regression.py)
  results/lda_topic_keywords.csv               (06_topics.py)      -> topic labels (optional)

Run it AFTER 06 and 07 have been run with the final number of topics (N_TOPICS in 06_topics.py).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path.cwd()
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- settings
BLUE, RED, GREY = "#3b6fb6", "#c0392b", "#bdbdbd"
GROUPS = ["audience_higher", "audience_lower"]          # same order as in 08_network.py
GROUP_COLORS = {"audience_higher": BLUE, "audience_lower": RED}
GROUP_NAMES = {"audience_higher": "Audience higher than expected",
               "audience_lower": "Audience lower than expected"}

GENRE_LABELS = {
    "action_adventure": "Action & Adventure", "animation": "Animation", "comedy": "Comedy",
    "documentary": "Documentary", "drama": "Drama", "horror": "Horror",
    "kids_family": "Kids & Family", "musical_performing_arts": "Musical & Performing Arts",
    "mystery_suspense": "Mystery & Suspense", "romance": "Romance",
    "science_fiction_fantasy": "Science Fiction & Fantasy",
    "art_house_international": "Art House & International",
}

# Interpretive names for the topics, if you want them instead of the top words, e.g.
# {"topic_5": "Humour and warmth"}. Leave empty to label topics with their top words.
TOPIC_LABEL_OVERRIDES = {}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG_DIR / name}.pdf/.png")


# ================================================================ FIGURE 1
df = pd.read_csv(PROCESSED_DIR / "movies_to_model.csv")


def bin_edges(series):
    """Internal Tomatometer bin edges, read from labels such as '(36.0, 57.0]'."""
    nums = sorted({float(x) for s in series.dropna().unique() for x in re.findall(r"-?\d+\.?\d*", s)})
    return nums[1:-1]


sent_path = PROCESSED_DIR / "movies_with_sentiment.csv"
test_path = OUTPUT_DIR / "sentiment_group_test.csv"
has_panel_b = sent_path.exists() and test_path.exists()
if not has_panel_b:
    print("WARNING: movies_with_sentiment.csv or sentiment_group_test.csv missing -> Figure 1 without panel (b)")

if has_panel_b:
    fig1, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
else:
    fig1, axa = plt.subplots(figsize=(7, 4.8))

# (a) group definition
mid = df[df["group_resid"] == "middle"]
axa.scatter(mid["tomatometer_rating"], mid["audience_rating"], s=8, color=GREY, alpha=0.35,
            linewidths=0, label=f"Other films (n = {len(mid)})", rasterized=True)
for g in ["audience_lower", "audience_higher"]:
    sub = df[df["group_resid"] == g]
    axa.scatter(sub["tomatometer_rating"], sub["audience_rating"], s=12, color=GROUP_COLORS[g],
                alpha=0.8, linewidths=0, label=f"{GROUP_NAMES[g]} (n = {len(sub)})", rasterized=True)
curve = df.sort_values("tomatometer_rating")
axa.plot(curve["tomatometer_rating"], curve["audience_expected"], color="black", lw=1.6,
         label="Expected audience score (quadratic fit)")
for edge in bin_edges(df["tomato_bin"]):
    axa.axvline(edge, color="black", lw=0.6, ls=":", alpha=0.6)
axa.set_xlabel("Tomatometer (critics, %)")
axa.set_ylabel("Audience score (%)")
axa.set_title("(a) Extreme groups, defined within Tomatometer bins", loc="left", fontsize=10)
axa.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)

# (b) sentiment of the critic reviews in the two groups
if has_panel_b:
    s = pd.read_csv(sent_path)
    t = pd.read_csv(test_path, index_col=0)["value"]
    data = [s.loc[s["group_resid"] == g, "mean_review_sentiment"].dropna() for g in GROUPS]
    bp = axb.boxplot(data, positions=[1, 2], widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=1.4))
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLORS[g])
        patch.set_alpha(0.7)
    axb.scatter([1, 2], [d.mean() for d in data], marker="D", s=28, color="white",
                edgecolor="black", zorder=3, label="Mean")
    axb.set_xticks([1, 2])
    axb.set_xticklabels(["Audience\nhigher", "Audience\nlower"])
    axb.set_ylabel("Mean VADER compound score per film")
    axb.set_title("(b) Sentiment of critic reviews", loc="left", fontsize=10)
    axb.legend(loc="lower left", fontsize=7.5, frameon=False)
    axb.text(0.97, 0.97,
             f"Difference in means = {t['mean_diff']:.3f}\n"
             f"95% CI [{t['ci95_low']:.3f}, {t['ci95_high']:.3f}]\n"
             f"Welch t = {t['t_statistic']:.2f}, p {fmt_p(t['p_value'])}",
             transform=axb.transAxes, ha="right", va="top", fontsize=7.5,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgrey"))
fig1.tight_layout()
save(fig1, "fig1_groups")


# ================================================================ TABLE 1
nested = pd.read_csv(OUTPUT_DIR / "model2_nested_comparison.csv")
wald = pd.read_csv(OUTPUT_DIR / "model2_wald_tests.csv", index_col=0)
assert len(nested) == 4, "model2_nested_comparison.csv should have 4 rows (3 nested models + robustness)"

labels = ["M1  Structural controls",
          "M2  M1 + review sentiment",
          "M3  M2 + topic shares",
          "M4  M3 + Tomatometer and its square (robustness)"]


def wald_text(row_name):
    r = wald.loc[row_name]
    return f"chi2({int(r['df'])}) = {r['chi2']:.2f}, p {fmt_p(r['p_value'])}"


wald_cells = ["-",
              wald_text("sentiment (Base+Sentiment model)"),
              wald_text("all text variables (full model)"),
              "-"]
table = pd.DataFrame({
    "Model": labels,
    "R2": nested["R2"],
    "Adj_R2": nested["Adj_R2"],
    "dR2_vs_previous": nested["R2"].diff(),
    "AIC": nested["AIC"],
    "dAIC_vs_previous": nested["AIC"].diff(),
    "N": nested["N"].astype(int),
    "Wald_test_added_text_variables": wald_cells,
})
table.to_csv(OUTPUT_DIR / "table1_nested_models.csv", index=False)

n_topics_wald = f"chi2({int(wald.loc['topics only (full model)', 'df'])}) = " \
                f"{wald.loc['topics only (full model)', 'chi2']:.2f}, " \
                f"p {fmt_p(wald.loc['topics only (full model)', 'p_value'])}"
note = (f"Dependent variable: audience_resid. Robust (HC3) standard errors. Wald tests are joint tests "
        f"of the text variables added in that model; topics only (M3): {n_topics_wald}.")


def cell(x, nd):
    return "-" if pd.isna(x) else f"{x:.{nd}f}"


body = [[r.Model, cell(r.R2, 3), cell(r.Adj_R2, 3), cell(r.dR2_vs_previous, 3), cell(r.AIC, 1),
         cell(r.dAIC_vs_previous, 1), f"{r.N:,}", r.Wald_test_added_text_variables]
        for r in table.itertuples()]
header = ["Model", "R$^2$", "Adj. R$^2$", "$\\Delta$R$^2$", "AIC", "$\\Delta$AIC", "N", "Wald test (text variables)"]

# LaTeX version (needs the booktabs package)
tex = ["\\begin{tabular}{lrrrrrrl}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
for row in body:
    row = [c.replace("chi2", "$\\chi^2$").replace("p < 0.001", "$p<0.001$") for c in row]
    tex.append(" & ".join(row) + " \\\\")
tex += ["\\bottomrule", "\\end{tabular}", f"% {note}"]
(OUTPUT_DIR / "table1_nested_models.tex").write_text("\n".join(tex), encoding="utf-8")

# PDF/PNG rendering of the table (to paste it as an image in Word)
fig_t, ax_t = plt.subplots(figsize=(11, 2.0))
ax_t.axis("off")
plain_header = ["Model", "R²", "Adj. R²", "ΔR²", "AIC", "ΔAIC", "N", "Wald test (text variables)"]
tbl = ax_t.table(cellText=[[c.replace("chi2", "χ²") for c in row] for row in body],
                 colLabels=plain_header, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.auto_set_column_width(list(range(len(plain_header))))
for (r, c), cl in tbl.get_celld().items():
    cl.set_edgecolor("lightgrey")
    if r == 0:
        cl.set_facecolor("#eeeeee")
        cl.set_text_props(fontweight="bold")
save(fig_t, "table1_nested_models")
print(note)


# ================================================================ FIGURE 2
coef = pd.read_csv(OUTPUT_DIR / "model2_full_coefficients.csv")
coef = coef[coef["term"] != "Intercept"].set_index("term")

topic_terms = [t for t in coef.index if re.fullmatch(r"topic_\d+_std", t)]
n_topics_total = len(topic_terms) + 1            # the reference topic (topic_1) is not in the model


def topic_label(term):
    tid = term.replace("_std", "")
    num = tid.split("_")[1]
    if tid in TOPIC_LABEL_OVERRIDES:
        return f"Topic {num}: {TOPIC_LABEL_OVERRIDES[tid]}, +1 SD"
    kw_path = OUTPUT_DIR / "lda_topic_keywords.csv"
    if kw_path.exists():
        kw = pd.read_csv(kw_path)
        if len(kw) == n_topics_total:
            words = kw.loc[kw["topic"] == tid, "top_words"]
            if len(words):
                return f"Topic {num} ({', '.join(words.iloc[0].split(', ')[:3])}), +1 SD"
        else:
            print(f"WARNING: lda_topic_keywords.csv has {len(kw)} topics but the regression has "
                  f"{n_topics_total}: topic words not shown (re-run 06 and 07 with the same N_TOPICS).")
    return f"Topic {num} share, +1 SD"


def period_label(term):
    return {"2000s": "Released 2000-2009 (vs 2010-2020)",
            "<2000": "Released before 2000 (vs 2010-2020)"}[re.search(r"\[T\.(.+)\]", term).group(1)]


def rating_label(term):
    level = re.search(r"\[T\.(.+)\]", term).group(1)
    return f"Content rating {level} (vs R)"


sections = []
sections.append(("Text variables", [("mean_review_sentiment_std", "Mean review sentiment, +1 SD")]
                 + [(t, topic_label(t)) for t in topic_terms]))
sections.append(("Release period",
                 [(t, period_label(t)) for t in coef.index if t.startswith("C(period")]))
sections.append(("Runtime", [("runtime_10", "Runtime, +10 minutes")]))
genre_terms = sorted(GENRE_LABELS, key=lambda g: coef.loc[g, "coef"], reverse=True)
sections.append(("Genre (film has the genre vs not)", [(g, GENRE_LABELS[g]) for g in genre_terms]))
sections.append(("Content rating",
                 [(t, rating_label(t)) for t in coef.index if t.startswith("C(content_rating")]))

rows = []                                          # (kind, label, term)
for title, items in sections:
    rows.append(("header", title, None))
    rows += [("coef", label, term) for term, label in items]

n = len(rows)
fig2, ax = plt.subplots(figsize=(8.4, 0.27 * n + 1.3))
ys = np.arange(n)[::-1]
for y, (kind, label, term) in zip(ys, rows):
    if kind == "header":
        ax.axhline(y + 0.5, color="lightgrey", lw=0.6)
        continue
    c = coef.loc[term]
    color = BLUE if c["coef"] > 0 else RED
    ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=color, lw=1.6, solid_capstyle="round")
    ax.plot(c["coef"], y, "o", color=color, mfc=color if c["p_value"] < 0.05 else "white",
            ms=5.5, zorder=3)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(ys)
ax.set_yticklabels([r[1] for r in rows])
for tick, (kind, _, _) in zip(ax.get_yticklabels(), rows):
    if kind == "header":
        tick.set_fontweight("bold")
ax.set_ylim(-0.7, n - 0.3)
ax.set_xlabel("Change in the audience residual (rating points), with 95% CI (HC3 robust errors)")
ax.legend(handles=[Line2D([], [], marker="o", color="grey", mfc="grey", ls="", label="p < 0.05"),
                   Line2D([], [], marker="o", color="grey", mfc="white", ls="", label="p ≥ 0.05"),
                   Line2D([], [], color=BLUE, lw=2, label="audience higher than expected"),
                   Line2D([], [], color=RED, lw=2, label="audience lower than expected")],
          loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=7.5, frameon=False)
fig2.tight_layout()
save(fig2, "fig2_coefficients")

# ================================================================ console summary
print("\nTable 1")
print(table.round(4).to_string(index=False))
print(f"\nTotal change in R2, M3 vs M1: {nested['R2'][2] - nested['R2'][0]:.4f}")
print("Figure 3 = figures/network_groups.pdf (from 08_network.py); exists:",
      (FIG_DIR / "network_groups.pdf").exists())