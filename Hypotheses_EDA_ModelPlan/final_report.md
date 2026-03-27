# Steam Game Analysis - Week 2 Final Report

## Project Context
This Week 2 report builds on the existing Steam analysis materials in the repository workspace and the modeling guidance in [eda_insights_report.md](/Users/laasyavenugopal/Downloads/eda_insights_report.md). The success definition remains fixed:

`success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0`

Under that rule:

- Successful games: 14,617
- Unsuccessful games: 75,001
- Positive class share: 16.31%

## Hypothesis Creation

Four hypotheses were selected for Week 2:

1. Paid games are more likely to be successful than free games.
2. Price tier matters more than raw price when predicting success.
3. Developer track record predicts success.
4. Certain genres are associated with higher success rates than others.

These hypotheses fit the EDA report more closely and avoid centering the final analysis on post-launch leakage features such as `peak_ccu`.

## Hypothesis Validation With EDA

### Hypothesis 1: Paid games succeed more than free games
This hypothesis is **supported**.

Using the existing price-based EDA:

| Group | Games | Success rate |
|---|---:|---:|
| Free | 14,160 | 11.12% |
| Paid | 75,458 | 17.28% |

Interpretation:

- Paid games show a higher success rate than free games.
- This suggests business model is one meaningful factor under the current review-based success definition.

### Hypothesis 2: Price tier matters more than raw price
This hypothesis is **supported**.

| Price tier | Games | Success rate |
|---|---:|---:|
| Free | 14,160 | 11.12% |
| Budget (<$15) | 66,147 | 14.64% |
| Mid ($15-$50) | 8,731 | 36.81% |
| Premium ($50-$85) | 361 | 38.23% |
| AAA ($85+) | 219 | 2.28% |

Supporting detail:

- Pearson correlation between success and raw price: `0.1215`

Interpretation:

- The tier breakdown is more informative than the raw price correlation.
- Mid and premium games perform better than both low-priced games and the AAA-priced outlier group.
- This supports the EDA report’s suggestion that price tier is more useful than raw price alone.

### Hypothesis 3: Developer track record predicts success
This hypothesis is **supported by the EDA insights report**.

| Feature | Mean (fail) | Mean (success) | Cohen's d | Strength |
|---|---:|---:|---:|---|
| Developer Historical Success | 0.15 | 0.34 | 0.901 | Strong |

Interpretation:

- Developer track record has the strongest separation between successful and unsuccessful games in the EDA report.
- This makes it a very strong candidate for predictive modeling.

### Hypothesis 4: Genre affects success probability
This hypothesis is **supported**.

| Genre | Games | Success rate |
|---|---:|---:|
| RPG | 16,342 | 18.69% |
| Adventure | 35,452 | 18.36% |
| Simulation | 18,570 | 16.84% |
| Indie | 63,189 | 16.52% |
| Strategy | 17,366 | 16.05% |
| Action | 36,842 | 15.51% |
| Sports | 3,939 | 13.81% |
| Racing | 3,301 | 13.54% |
| Casual | 38,699 | 13.47% |
| Free To Play | 8,867 | 11.63% |
| Early Access | 9,113 | 11.24% |
| Massively Multiplayer | 2,123 | 7.87% |

Interpretation:

- `RPG` and `Adventure` show the highest success rates among major genres.
- `Massively Multiplayer`, `Early Access`, and `Free To Play` perform worse under the current success definition.
- Because games can have multiple genres, these results should be interpreted as broad associations.

## Findings

### Key finding 1
Paid games outperform free games under the current success definition.

### Key finding 2
Price works better as a tiered variable than as a simple linear feature.

### Key finding 3
Developer track record appears to be one of the strongest predictors available in the EDA report.

### Key finding 4
Genre remains a meaningful explanatory feature, especially for `RPG` and `Adventure` games.

## Issues And Challenges

### Defining success
The project still uses a review-based success proxy rather than a direct revenue metric. This should be treated as a fixed project definition, but it remains an interpretation limit.

### Class imbalance
Only 16.3% of games are labeled successful. This means accuracy alone would be misleading, so evaluation should emphasize ROC-AUC and F1.

### Leakage risk
The EDA report confirms that post-launch features such as `peak_ccu`, owners, and Metacritic can artificially inflate model performance. Those should not be the center of the final predictive story.

### Genre overlap
Many games belong to multiple genres, so genre findings should be treated as overlapping effects rather than mutually exclusive categories.

### Temporal drift
The EDA report shows a meaningful drop in success rates across release years, so a time-based split may be more realistic than a purely random split.

## Tasks Remaining

### Modeling
Train a stronger sequence of models rather than relying on one baseline only.

### Tuning
Use Optuna or a similar tuner to optimize the strongest tree-based models.

### Interpretation
Use SHAP to confirm which features actually drive performance.

### Clustering
Run K-Means or a similar clustering method to identify market archetypes.

### Presentation
Integrate the EDA charts and model comparison visuals into the final submission.

## Suggested Models For Each Hypothesis

### Hypothesis 1 model
Use **Random Forest** with a binary `is_free` feature and supporting predictors. This keeps the hypothesis aligned with the agreed baseline from the strategy report.
It fits this hypothesis well because Random Forest is our approved baseline and still gives a clear test of whether the free-versus-paid difference matters.

### Hypothesis 2 model
No separate standalone model is needed for this hypothesis. The price-tier pattern is already validated in EDA, and `price_tier` should be used as an engineered feature inside the main **Random Forest**, **XGBoost/LightGBM**, and **MLP** pipeline.
It fits this hypothesis well because the key value of price tiers is feature engineering, not benchmarking a weaker standalone tree after Random Forest.

### Hypothesis 3 model
Use **LightGBM** with developer-history variables, since the EDA report shows very strong separation for this feature group.
It fits this hypothesis well because developer track record likely combines with other signals, and LightGBM is strong at learning those interactions.

### Hypothesis 4 model
Use **CatBoost** or **Random Forest** with one-hot encoded genre variables so the model can learn interactions across overlapping genre labels.
It fits this hypothesis well because games often belong to more than one genre, so a tree-based model is better suited to handling that overlap.

## Full Modeling Plan

The most suitable progression from the EDA report is:

1. Random Forest baseline
2. XGBoost or LightGBM
3. Tabular MLP with log-transformed skewed features
4. Stacking ensemble

This gives the project both interpretable baselines and stronger non-linear models.

## Final Integration

The revised Week 2 analysis supports four cleaner hypotheses that align with the EDA report and the actual modeling plan. Paid vs free structure, price tiers, developer track record, and genre all provide clearer and more defensible directions for the final report than leakage-prone post-launch metrics.

Overall, the project is now set up to connect EDA, hypothesis testing, and advanced modeling into one consistent final narrative built around Random Forest, boosting models, MLP, and stacking.
