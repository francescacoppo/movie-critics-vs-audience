# Movie critics vs audience

Project on Rotten Tomatoes: when does the audience rate a film
higher or lower than the critics, and does the language of the critic reviews say something about it?

The project looks at the gap between the Tomatometer (critics) and the audience score, and uses
text mining (sentiment, topics, word networks) on the critic reviews to describe it.

## Research questions

1. Which films are rated higher by the audience than by the critics, and which lower? Does the gap depend on genre, release period, runtime or content rating?
2. Is the gap related to the sentiment and the topics of the critic reviews?
3. Do the critic reviews of the two groups of films use a different vocabulary? (semantic network analysis)

## Main idea

The gap is measured as `D = audience score - Tomatometer`.

The Tomatometer is computed from the same critic reviews that are analysed, so their sentiment can be related to
the gap almost by construction. To avoid this, the two extreme groups are defined **within bins of the Tomatometer**:

- the expected audience score is estimated from the Tomatometer (quadratic fit);
- the residual (`audience_resid`) is the difference between the real and the expected audience score;
- inside each of 5 Tomatometer bins, the top 10% of residuals form the group **audience_higher** and the bottom 10% the group **audience_lower**.

In this way the two groups have a comparable Tomatometer but a different audience score.

## Pipeline

| Script | What it does |
|---|---|
| `01_cleaning.py`| Movie data cleaning |
| `02_EDA.ipynb` | Exploratory analysis: distribution of `D`, mean `D` by genre and by content rating |
| `02b_features.py` | Genre dummies, release period, extreme groups, expected audience score and residual |
| `03_regression.py` | Model 1 OLS regression - genre, period, runtime, classification|
| `04_reviews_sampling.py` | Reviews data cleaning and sampling|
| `05_sentiment.py` | VADER sentiment of each review, validation against the fresh/rotten label, sentiment of the two groups (Welch t-test) |
| `06_topics.py` | LDA topic model of the reviews |
| `07_model2_regression.py` | Nested OLS regressions of `audience_resid` (robust HC3 standard errors) |
| `08_network.py` | Word co-occurrence networks of the two groups and comparison with random relabelling |
| `09_figures.py` | Sentiment figure and summary table of the models for the report |

Run the scripts in this order from the root of the repository.

## Methods in short

- **Sentiment:** VADER compound score (-1 to +1) for each review, averaged by film.
- **Regression models:** four nested models of `audience_resid`: structural controls (genre, period, runtime, content rating), plus review sentiment, plus topic shares, plus Tomatometer (robustness check).
- **Word network:** nodes are the 100 most frequent words of the reviews of each group, edges are the Jaccard co-occurrence between words in the same review, keeping the strongest 10% of the possible edges. Measures: degree, closeness, betweenness, eigenvector centrality and communities (greedy modularity).
- **Network comparison:** node overlap and Spearman correlation of the eigenvector centrality of the shared words, compared with 1000 random relabellings of the films. The result is checked with other edge thresholds (5%, 15% and the mean weight).

## Main results

- On average, the critic reviews of the films where the audience is more generous are slightly more positive (difference in mean VADER score about 0.06, Welch t = 4.25, p < 0.001).
- The two networks are different from what is expected by chance: the shared words have a different importance (Spearman correlation of eigenvector centrality about 0.47, against about 0.70 expected at random), while the overall structure (density, diameter, number of communities) is similar. The result holds with the other edge thresholds.
- The vocabulary is mostly generic, so the networks describe the style of the reviews more than the content of the films.

## Colours

Same convention in all figures: blue = audience higher than critics, red = critics higher than audience.

## Repository structure

```
data/
  raw/            original data (not included, see below)
  processed/      cleaned data and intermediate files
results/          tables (csv) produced by the scripts
figures/          figures (pdf) produced by the scripts
```

## Data

data/raw/rotten_tomatoes_critic_reviews.csv

data/raw/rotten_tomatoes_movies.csv

## Requirements

Python 3 with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `nltk` (VADER lexicon), `networkx`, `matplotlib`, `statsmodels`.

```
pip install pandas numpy scipy scikit-learn nltk networkx matplotlib statsmodels
```

## Author

Francesca Coppo