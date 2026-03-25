# EDA & Insights Report

## Dataset summary
- Total games: 89,618
- Successful: 14,617 (16.3%) | Unsuccessful: 75,001 (83.7%)
- Model features: 68 | EDA reference: 9
- Class imbalance ratio: 5.1:1

## Charts produced
All saved to `EDA/charts/`:
1. `01_class_balance.png` — Target variable distribution
2. `02_feature_distributions.png` — Feature histograms split by class
3. `03_class_separability.png` — Cohen's d for each feature
4. `04_multicollinearity.png` — Feature-feature correlation heatmap
5. `05_success_by_genre.png` — Success rate by genre
6. `06_outliers.png` — Box plots with outlier counts
7. `07_temporal_trend.png` — Success rate over release years
8. `08_leakage_proof.png` — Post-launch correlation with success
9. `09_success_breakdowns.png` — Success by price tier, platform, free/paid

---

## Insight 1: Class imbalance requires careful handling
Only 16.3% of games are successful. The imbalance ratio is 5.1:1.

**Modeling decision:** Use `class_weight='balanced'` for all tree models. Evaluate SMOTE on the training fold only (never on validation). Primary metric is ROC-AUC, not accuracy.

---

## Insight 2: Feature skewness — log transforms needed for MLP
| Feature | Skewness | Needs log transform? |
|---|---:|---|
| achievements | 28.46 | Yes — use log1p |
| price | 20.72 | Yes — use log1p |
| audio_language_count | 8.70 | Yes — use log1p |
| developer_game_count | 8.03 | Yes — use log1p |
| language_count | 6.26 | Yes — use log1p |
| platform_count | 1.73 | No |
| genre_count | 0.87 | No |
| description_length | -0.38 | No |

**Modeling decision:** Apply `np.log1p()` to features with skewness > 2 before feeding into the Tabular MLP. Tree-based models (RF, XGBoost) are robust to skewness and do not need this transformation.

---

## Insight 3: Feature separability — which features actually help
| Feature | Mean (fail) | Mean (success) | Cohen's d | Strength |
|---|---:|---:|---:|---|
| Developer Historical Success | 0.15 | 0.34 | 0.901 | Strong |
| Tag Count | 10.30 | 16.16 | 0.769 | Strong |
| Has Website | 0.42 | 0.64 | 0.435 | Strong |
| Platform Count | 1.29 | 1.54 | 0.385 | Strong |
| Price | 6.59 | 10.98 | 0.331 | Strong |
| Description Length | 193.75 | 205.55 | 0.152 | Moderate |
| Achievements | 18.06 | 33.33 | 0.093 | Weak |
| Genre Count | 2.90 | 2.77 | 0.092 | Weak |
| Developer Game Count | 3.53 | 2.43 | 0.086 | Weak |
| Language Count | 5.08 | 6.06 | 0.074 | Weak |

**Modeling decision:** Features with Cohen's d > 0.2 are strong candidates. Features with d < 0.05 may be dropped if they add noise. Developer track record and price should be among the most important features.

---

## Insight 4: Multicollinearity — redundant feature pairs
| Feature pair | Correlation | Action |
|---|---:|---|
| Language Count vs Audio Language Count | 0.794 | Consider dropping one |
| Developer Historical Success vs Publisher Historical Success | 0.770 | Consider dropping one |
| Release Year vs Years Since Release | -1.000 | Consider dropping one |

**Modeling decision:** Drop one feature from any pair with correlation > 0.85. For pairs in the 0.7-0.85 range, keep both but monitor feature importance — if both rank low, drop the weaker one. Tree models handle this naturally; MLP is sensitive.

---

## Insight 5: Outlier analysis — how many, do they carry signal, and what to do
| Feature | Outliers | % of data | Skewness | Signal bias | Treatment |
|---|---:|---:|---:|---|---|
| developer_game_count | 16,592 | 18.5% | 8.0 | balanced | log1p |
| language_count | 13,281 | 14.8% | 6.3 | successful | log1p |
| achievements | 5,636 | 6.3% | 28.5 | successful | log1p |
| price | 4,146 | 4.6% | 20.7 | successful | log1p |
| description_length | 0 | 0.0% | -0.4 | balanced | none |

**Signal bias** indicates whether outliers are disproportionately in one class. If biased toward "successful," removing outliers would destroy predictive signal. Capping preserves the direction while limiting magnitude.

**Treatment strategy by model type:**
- **RF / XGBoost / LightGBM:** No outlier treatment needed. Tree models split on rank order, not magnitude. A game priced at $999 vs $60 makes no difference if the split threshold is at $30.
- **MLP (Neural Network):** Treatment required because the network does arithmetic with raw values. Large outliers cause gradient explosion and dominate weight updates.
  - Features with skewness > 2: Apply `np.log1p()` to compress the right tail naturally.
  - Features with skewness 1-2 and >5% outliers: Cap at the 99th percentile.
  - After treatment: Apply `StandardScaler` to normalize all features to mean=0, std=1.
- **Important:** Never remove outlier rows. Capping or transforming values preserves all training examples while limiting their numerical influence.

---

## Insight 6: Temporal concept drift
Early years avg success rate: 36.7%, Late years: 14.5%, Drift: -22.2%

**Modeling decision:** If success rates have shifted significantly over time, consider using a time-based train/test split (train on pre-2022, test on 2022+) instead of random splitting. This simulates real-world deployment where the model predicts future games.

---

## Insight 7: Leakage proof — why post-launch columns are excluded
| Column | Correlation with success | Risk level |
|---|---:|---|
| Log Peak Ccu | 0.428 | HIGH |
| Log Owners | 0.271 | moderate |
| Metacritic Score | 0.249 | moderate |
| Has Metacritic | 0.233 | moderate |
| Owners Midpoint | 0.074 | low |
| Dlc Count | 0.031 | low |
| Peak Ccu | 0.024 | low |
| Avg Playtime | 0.021 | low |
| Median Playtime | 0.016 | low |

**Modeling decision:** These correlations confirm that including post-launch metrics would inflate model performance artificially. The exclusion decision from Week 1 is validated.

---

## Insight 8: Edge cases to handle before modeling
| Condition | Count |
|---|---:|
| Games with 0 genres | 212 |
| Games with 0 categories | 876 |
| Games with 0 tags | 16,872 |
| Games with platform_count=0 | 0 |
| Games with price > $200 | 8 |
| Games with 0 achievements | 41,090 |
| Games with no release date | 0 |
| Games with description_length=0 | 120 |

**Modeling decision:** Games with 0 genres/categories/tags should be imputed as a separate "Unknown" category or dropped if count is small. Games with no release date need imputation or removal.

---

## Hypothesis validation summary

| Hypothesis | Result | Key evidence |
|---|---|---|
| Paid games succeed more than free games | To validate | Compare success rates by is_free |
| Multi-platform games succeed more | To validate | Success rate by platform_count |
| Developer track record predicts success | To validate | Cohen's d and correlation analysis |
| Genre affects success probability | To validate | Success rate varies significantly by genre |
| Price tier matters more than raw price | To validate | Compare model performance with/without tiers |
| Post-launch metrics would cause leakage | Validated | High correlations in leakage proof chart |

---

## Modeling recommendations
1. **Train/test split:** Consider time-based split if concept drift is significant; otherwise 80/20 stratified random split.
2. **Class balancing:** `class_weight='balanced'` for RF/XGBoost. SMOTE only on training folds.
3. **Feature preprocessing:** Log-transform skewed features for MLP only. Standardize all features for MLP.
4. **Feature selection:** After initial RF, use SHAP importance to identify and potentially drop low-signal features.
5. **Evaluation:** ROC-AUC primary, F1 secondary. Report confusion matrix, precision, recall for both classes.

## What is applied in the CSV vs deferred to modeling

| EDA recommendation | Applied in CSV? | Why |
|---|---|---|
| Drop `years_since_release` (redundant with `release_year`, corr=-1.0) | Yes | Redundancy is always wrong, regardless of model type |
| Fill NaN values in numeric features with median | Yes | All models need complete data |
| Log-transform skewed features (achievements, developer_game_count) | No — deferred | Tree models perform worse with log transforms. Applied per-model in training. |
| Cap outliers at 99th percentile | No — deferred | Only needed for MLP. Trees are robust to outliers. |
| StandardScaler normalization | No — deferred | Only needed for MLP. Trees are invariant to scaling. |
| Drop low-separability features (tag_count, description_length) | No — deferred | Let SHAP confirm after training rather than pre-judging. |

## Issues and challenges
1. Class imbalance (16.3% positive) is the dominant challenge.
2. Sentiment-based success metric does not capture commercial revenue directly.
3. Survivorship bias — delisted games are absent.
4. Review bombing affects competitive multiplayer games disproportionately.
5. Feature sparsity in one-hot genre/tag columns for rare categories.

## Tasks remaining
1. Model training: RF → XGBoost/LightGBM → MLP → Stacking Ensemble
2. Hyperparameter tuning with Optuna (Bayesian optimization)
3. SHAP feature importance analysis
4. K-Means clustering for market archetypes
5. Interactive prediction tool
