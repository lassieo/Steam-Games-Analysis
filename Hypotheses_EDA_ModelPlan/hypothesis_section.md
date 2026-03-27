# Hypothesis Section

## Scope
This section uses the existing Steam analysis outputs together with the modeling guidance summarized in [eda_insights_report.md](/Users/laasyavenugopal/Downloads/eda_insights_report.md). The project success label remains:

`success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0`

That definition produces 14,617 successful games and 75,001 unsuccessful games, so the positive class share is 16.31%.

## Hypothesis Creation

### Hypothesis 1
Paid games are more likely to be successful than free games.

### Hypothesis 2
Price tier matters more than raw price when predicting success.

### Hypothesis 3
Developer track record predicts success.

### Hypothesis 4
Certain genres are associated with higher success rates than others.

These hypotheses were selected because they fit the EDA insights report and avoid relying on post-launch leakage features such as `peak_ccu`.

## Hypothesis Validation Using EDA

### Hypothesis 1: Paid games succeed more than free games
Result: **Supported.**

Evidence from the existing price-tier EDA:

| Group | Games | Success rate |
|---|---:|---:|
| Free | 14,160 | 11.12% |
| Paid | 75,458 | 17.28% |

Interpretation:

- Paid games have a higher success rate than free games under the current project definition.
- This supports the idea that monetized titles tend to show stronger review-based success outcomes in the Steam dataset.

### Hypothesis 2: Price tier matters more than raw price
Result: **Supported.**

Evidence from the grouped EDA:

| Price tier | Games | Success rate |
|---|---:|---:|
| Free | 14,160 | 11.12% |
| Budget (<$15) | 66,147 | 14.64% |
| Mid ($15-$50) | 8,731 | 36.81% |
| Premium ($50-$85) | 361 | 38.23% |
| AAA ($85+) | 219 | 2.28% |

Additional supporting detail:

- Pearson correlation between success and raw price: `0.1215`

Interpretation:

- The weak raw correlation and the much clearer tier pattern suggest that price bands are more informative than a simple linear price effect.
- Mid and premium games perform best, while free and AAA-priced games perform worse.

### Hypothesis 3: Developer track record predicts success
Result: **Supported by the EDA insights report.**

Evidence from [eda_insights_report.md](/Users/laasyavenugopal/Downloads/eda_insights_report.md):

| Feature | Mean (fail) | Mean (success) | Cohen's d | Strength |
|---|---:|---:|---:|---|
| Developer Historical Success | 0.15 | 0.34 | 0.901 | Strong |

Interpretation:

- Developer historical success has the strongest class separation in the EDA report.
- This makes it one of the best candidate predictors for downstream modeling.

### Hypothesis 4: Certain genres affect success probability
Result: **Supported.**

Evidence from genre-level EDA for genres with at least 1,000 tagged games:

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

- Genre is associated with different success outcomes in the dataset.
- `RPG` and `Adventure` show some of the highest success rates among major genre groups.
- `Massively Multiplayer`, `Early Access`, and `Free To Play` show lower success rates under the current project definition.

## Models To Build For Each Hypothesis

### Hypothesis 1 model
Use **Random Forest** as the baseline model with `is_free` and supporting metadata included. This keeps the hypothesis aligned with the agreed project pipeline, where Random Forest is the first real benchmark.
This model fits well because it tests the free-versus-paid distinction without dropping below the agreed baseline complexity for the project.

### Hypothesis 2 model
No separate standalone model is needed for this hypothesis. The price-tier result is already validated in EDA, and `price_tier` should be carried into the main **Random Forest**, **XGBoost/LightGBM**, and **MLP** pipeline as an engineered feature.
This fits well because the tier structure is already visible in the tables, so the goal is to preserve it inside the main models rather than benchmark a weaker standalone tree.

### Hypothesis 3 model
Use **LightGBM** with developer-history features included. Since developer track record shows the strongest class separation, boosting is a strong fit for capturing its importance and interaction effects.
This model fits well because developer track record may interact with several other features, and LightGBM is good at learning those more complex patterns.

### Hypothesis 4 model
Use **CatBoost** or a **Random Forest** with one-hot encoded genre variables. Because games can belong to multiple genres, tree-based models are better than a simple regression at learning interaction patterns across overlapping genre tags.
This model fits well because genre is a multi-label feature, and tree-based models handle overlapping categories better than a simple linear model.

### Full comparison model set
For the final project, the strongest modeling sequence from the EDA report is:

1. Random Forest baseline
2. XGBoost or LightGBM
3. Tabular MLP with log-transformed skewed features
4. Stacking ensemble

## Findings And Pattern Interpretation

### Pattern 1
Business model matters. Paid games outperform free games under the current success definition.

### Pattern 2
Price behaves non-linearly. Price tier is more informative than treating price as a straight numeric increase.

### Pattern 3
Developer track record appears to be one of the strongest predictors available in the EDA report.

### Pattern 4
Genre remains an important explanatory factor, but it should be treated as an overlapping multi-label signal rather than a single-category variable.

### Overall conclusion
These four hypotheses align better with the EDA insights report and support a stronger modeling pipeline built around Random Forest, boosting, MLP, and stacking rather than weaker standalone baselines.
