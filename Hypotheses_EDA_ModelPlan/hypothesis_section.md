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
| Paid | 75,458 | 17.06% |

Interpretation:

- Paid games have a higher success rate than free games under the current project definition.
- This supports the idea that monetized titles tend to show stronger review-based success outcomes in the Steam dataset.

### Hypothesis 2: Price tier matters more than raw price
Result: **Supported.**

Evidence from the grouped EDA:

| Price tier | Games | Success rate |
|---|---:|---:|
| Free | 14,160 | 11.12% |
| 0.01-4.99 | 39,388 | 10.52% |
| 5.00-9.99 | 19,090 | 17.87% |
| 10.00-19.99 | 12,731 | 31.11% |
| 20.00-29.99 | 2,558 | 36.75% |
| 30.00-59.99 | 1,383 | 40.27% |
| 60+ | 308 | 9.74% |

Additional supporting detail:

- Pearson correlation between success and raw price: `0.1215`

Interpretation:

- The weak raw correlation and the much clearer tier pattern suggest that price bands are more informative than a simple linear price effect.
- Mid-priced games perform best, while the cheapest and most expensive groups perform worse.

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
Use a **logistic regression** model with `is_free` as the main predictor and a few control variables. This is the simplest and cleanest baseline for testing whether the free-vs-paid difference remains meaningful.
This model fits well because the hypothesis is a straightforward yes/no comparison, and logistic regression is easy to explain and interpret.

### Hypothesis 2 model
Use a **decision tree** with engineered price tiers. This is a good fit because the EDA suggests threshold behavior rather than a smooth linear price effect.
This model fits well because a decision tree can naturally split games into price ranges and show where success rates change the most.

### Hypothesis 3 model
Use **LightGBM** with developer-history features included. Since developer track record shows the strongest class separation, boosting is a strong fit for capturing its importance and interaction effects.
This model fits well because developer track record may interact with several other features, and LightGBM is good at learning those more complex patterns.

### Hypothesis 4 model
Use **CatBoost** or a **Random Forest** with one-hot encoded genre variables. Because games can belong to multiple genres, tree-based models are better than a simple regression at learning interaction patterns across overlapping genre tags.
This model fits well because genre is a multi-label feature, and tree-based models handle overlapping categories better than a simple linear model.

### Full comparison model set
For the final project, the strongest modeling sequence from the EDA report is:

1. Random Forest baseline
2. Logistic Regression baseline
3. XGBoost or LightGBM
4. Tabular MLP with log-transformed skewed features
5. Stacking ensemble

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
These four hypotheses align better with the EDA insights report and support stronger downstream models than the earlier leakage-prone hypotheses.
