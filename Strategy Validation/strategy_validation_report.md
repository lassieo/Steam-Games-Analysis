# Strategy & Validation Report
## Steam Games Success Prediction — Dataset Review & Feature Strategy

**Author:** Pujitha Attuluri
**Role:** Strategy & Validation Lead
**Dataset:** `games_march2025_cleaned_binary_success.csv` (89,618 games, 51 columns)

---

## 1. Professor Feedback Alignment

The professor raised two specific concerns that shape every decision in this report:

**Concern 1: Weak baselines inflate perceived improvement.**
Logistic Regression is too simple for data with nonlinear relationships. If we start there, even modest models look good by comparison.

**Our response:** Random Forest replaces Logistic Regression as the baseline. The full model ladder is: Random Forest (baseline) → XGBoost / LightGBM (primary) → Tabular MLP (advanced) → Stacking Ensemble (final). Each step adds genuine complexity. Logistic Regression coefficients may appear in an appendix for interpretability, but it is not treated as a benchmark.

**Concern 2: Not exhausting model options can lead to shallow conclusions.**
Claiming "our model performs well" without trying more complex approaches is insufficient.

**Our response:** We train five distinct model architectures, tune hyperparameters with Bayesian optimization (Optuna), and evaluate with stratified 5-fold cross-validation using ROC-AUC, F1, precision, and recall. The stacking ensemble combines predictions from the best three models.

---

## 2. Success Definition Validation

Here is the validation for success:

| Option | Successful games | % of dataset |
|---|---:|---:|
| Option 1 (Simple): ratio >= 90% | 24,405 | 27.23% |
| Option 2 (Balanced): ratio >= 80% AND reviews >= 50 | 14,617 | 16.31% |
| Option 3 (Relative): top 25% in both | 2,401 | 2.68% |

**Selected: Option 2 (Balanced)** — `success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0`

**Why this works:**
- Captures quality through strong sentiment (80% positive).
- Captures popularity through minimum review evidence (50 reviews).
- Avoids Option 1's failure of labeling 1-review games as successful.
- Avoids Option 3's failure of making success so exclusive the positive class becomes too small.

**Acknowledged limitation:** This metric measures community reception, not commercial revenue. Some commercially successful games with polarized reviews (e.g., PUBG at 59% positive, Apex Legends at 67% positive) are labeled unsuccessful because competitive multiplayer communities generate more divisive feedback. This is a known limitation, not a flaw — the metric captures what players think, which is a valuable signal.

---

## 3. Column Audit & Feature Classification

Every column in the dataset was audited and classified into one of three groups.

**Summary:** 16 model features | 9 EDA reference | 26 excluded

### 3a. Model Features — pre-launch, no leakage (16 columns)

These are the raw columns selected for modeling. During feature engineering, they will be transformed into ~49 model-ready features (one-hot encoding, counts, track records, temporal decomposition).

| Column | Type | Null % | Rationale |
|---|---|---:|---|
| `release_date` | object | 0.0% | Decomposed into release_year, release_month, release_quarter, release_day_of_week, released_in_sale_month flag, and years_since_release. |
| `required_age` | int64 | 0.0% | Age rating signal that affects audience reach. Used directly as a numeric feature. |
| `price` | float64 | 0.0% | Core business decision. Used directly and also converted into is_free flag and price_tier bins. |
| `short_description` | object | 0.1% | Text length measured as description_length, a proxy for marketing effort. |
| `website` | object | 54.1% | Converted to has_website binary flag indicating whether a dedicated game website exists. |
| `windows` | bool | 0.0% | Platform support flag. Combined with mac and linux into a single platform_count feature. |
| `mac` | bool | 0.0% | Platform support flag. Combined with windows and linux into a single platform_count feature. |
| `linux` | bool | 0.0% | Platform support flag. Combined with windows and mac into a single platform_count feature. |
| `achievements` | int64 | 0.0% | Proxy for content depth and polish. Used directly as a numeric feature. |
| `supported_languages` | object | 0.0% | Counted to get language_count, which reflects localization investment level. |
| `full_audio_languages` | object | 0.0% | Counted to get audio_language_count, a stronger signal of localization investment than text-only. |
| `developers` | object | 0.0% | Studio name used to compute developer_historical_success (past success rate) and developer_game_count (experience). Both computed using only prior games to avoid leakage. |
| `publishers` | object | 0.0% | Publisher name used to compute publisher_historical_success using only prior games. |
| `categories` | object | 0.0% | Platform capabilities like multiplayer, co-op, and workshop support. One-hot encoded into the top 12. |
| `genres` | object | 0.0% | Primary discovery signal on Steam. One-hot encoded into the top 15 most common genres. |
| `tags` | object | 0.0% | Community-assigned descriptors more granular than genres. Top 20 tags encoded as binary features. |

**Engineered features that will be derived from these:**
- `is_free`, `price_tier` (free/budget/mid/premium/AAA) — from `price`
- `platform_count` — sum of windows + mac + linux
- `language_count`, `audio_language_count` — from supported_languages, full_audio_languages
- `release_year`, `release_month`, `release_quarter`, `release_day_of_week`, `released_in_sale_month`, `years_since_release` — from release_date
- `developer_historical_success`, `publisher_historical_success`, `developer_game_count` — from developers/publishers (computed with no future leakage)
- `description_length` — from short_description
- `has_website` — from website
- `genre_count`, `category_count`, `tag_count` — diversity counts
- Top 15 genre, 12 category, 20 tag binary features — one-hot encoded

### 3b. EDA Reference — post-launch, validation only (9 columns)

Included in the output dataset with `eda_` prefix for exploratory analysis. Used to validate that model predictions correlate with real-world outcomes. **Never used as model inputs.**

| Column | Why EDA only |
|---|---|
| `dlc_count` | Post-launch content strategy decision |
| `metacritic_score` | Available days/weeks after launch — borderline but excluded for clean boundary |
| `estimated_owners` | Post-launch outcome — would cause temporal leakage |
| `average_playtime_forever` | Post-launch engagement — only observable after purchase |
| `average_playtime_2weeks` | Post-launch recent engagement |
| `median_playtime_forever` | Post-launch engagement (median) |
| `median_playtime_2weeks` | Post-launch recent engagement (median) |
| `discount` | Post-launch pricing action |
| `peak_ccu` | Post-launch popularity metric |

### 3c. Excluded — target leakage or metadata (26 columns)

These columns are dropped entirely from the output.

| Column | Reason |
|---|---|
| `appid` | Row identifier — not a feature |
| `name` | Row identifier — not a feature |
| `detailed_description` | Full HTML text — not used without NLP (deferred) |
| `about_the_game` | Duplicate of detailed_description |
| `reviews` | Press review quotes — 60% null, unstructured |
| `header_image` | URL — not a feature without image analysis |
| `support_url` | 20% null — low signal |
| `support_email` | 70% null — too sparse |
| `metacritic_url` | URL — not a feature |
| `recommendations` | Engagement metric correlated with target — target leakage |
| `notes` | 65% null — content warnings, low signal |
| `packages` | Pricing structure JSON — complex, low marginal value |
| `screenshots` | URL list — not a feature without image analysis |
| `movies` | URL list — not a feature without video analysis |
| `user_score` | All zeros in dataset — no signal |
| `score_rank` | 100% null — no data |
| `positive` | Direct component of success formula — target leakage |
| `negative` | Direct component of success formula — target leakage |
| `pct_pos_total` | Derived from positive/negative — target leakage |
| `num_reviews_total` | Inconsistent with positive+negative — also target leakage |
| `pct_pos_recent` | Derived from review counts — target leakage |
| `num_reviews_recent` | Recent review count — target leakage |
| `review_total_calc` | Engineered in Step 1 to compute target — target leakage |
| `positive_ratio_calc` | Engineered in Step 1 to compute target — target leakage |
| `has_reviews_calc` | Boolean from review count — target leakage |
| `success` | TARGET VARIABLE — not a feature |

---

## 4. Risk Analysis

### 4a. Class imbalance
- **Finding:** 16.3% positive class (14,617 of 89,618). Imbalance ratio: 5.1:1.
- **Risk:** Models will favor majority class (unsuccessful) for raw accuracy.
- **Mitigation:** `class_weight='balanced'` in tree models, SMOTE evaluation, stratified CV. ROC-AUC as primary metric.

### 4b. Price skewness
- **Finding:** 15.8% of games are free. Mean $7.31, median $4.99, max $999.98.
- **Risk:** Heavy right skew with spike at zero.
- **Mitigation:** `price_tier` categorical encoding alongside raw price. `is_free` binary flag.

### 4c. Missing data
| Column | Missing | % |
|---|---:|---:|
| `score_rank` | 89,579 | 100.0% |
| `metacritic_url` | 86,071 | 96.0% |
| `reviews` | 79,217 | 88.4% |
| `notes` | 72,975 | 81.4% |
| `website` | 48,504 | 54.1% |
| `support_url` | 45,508 | 50.8% |
| `positive_ratio_calc` | 17,055 | 19.0% |
| `support_email` | 10,820 | 12.1% |
| `about_the_game` | 220 | 0.2% |
| `detailed_description` | 197 | 0.2% |
| `short_description` | 120 | 0.1% |
- **Mitigation:** Drop `score_rank` (100% null). Use `eda_has_metacritic` binary flag for metacritic. Impute `has_website` as 0 for nulls.

### 4d. Temporal leakage
- **Risk:** Post-launch columns (owners, playtime, CCU) would inflate model performance artificially.
- **Mitigation:** Strict separation. Post-launch columns prefixed `eda_` and excluded from all model feature lists. Developer track records computed using only historically prior games.

### 4e. Target leakage
- **Risk:** Columns like `positive`, `negative`, `positive_ratio_calc` are direct components of the success formula.
- **Mitigation:** Excluded entirely — not even in EDA reference columns.

### 4f. Survivorship bias
- **Risk:** Dataset only contains games currently on Steam. Delisted/removed games (often failures) are absent.
- **Limitation:** Acknowledged. Results may slightly overstate overall success rates.

### 4g. Review bombing
- **Risk:** Competitive games receive coordinated negative campaigns unrelated to quality.
- **Limitation:** Metric captures sentiment as-is. Acknowledged that community dynamics can depress ratios.

### 4h. Multicollinearity
- **Risk:** Correlated pairs: `price` ↔ `is_free`, `genre_action` ↔ `tag_action`, `language_count` ↔ `audio_language_count`.
- **Mitigation:** Tree-based models are robust. Check VIF and correlation heatmap in EDA, drop if needed for MLP.

---

## 5. Assumptions

1. Steam user reviews are a meaningful proxy for game quality and community reception.
2. The 80% positive threshold and 50-review minimum are reasonable boundaries for "successful."
3. Pre-launch features are sufficient to capture meaningful predictive signal.
4. Community-assigned tags stabilize within the first week and are treated as near-launch features.
5. The dataset represents the Steam ecosystem as of March 2025 and may not generalize to future conditions.


