"""
Strategy & Validation — Feature Selection Analysis
====================================================

Reads the success-labeled dataset and performs:
  1. Column audit — dtype, nulls, unique values, sample
  2. Feature classification — model feature / EDA reference / excluded
  3. Temporal leakage detection
  4. Target leakage detection
  5. Risk analysis — class imbalance, skew, missing data, bias
  6. Generates strategy_validation_report.md

Reads:  games_march2025_cleaned_binary_success.csv
Writes: strategy_validation_report.md (to this folder)
        feature_audit.csv (detailed per-column audit)
"""

from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths — update BASE_DIR to match your local layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_OUTPUT = REPO_ROOT / "data" / "output"
SOURCE_FILE = DATA_OUTPUT / "games_march2025_with_success.csv"
OUTPUT_REPORT = Path(__file__).resolve().parent / "strategy_validation_report.md"
OUTPUT_AUDIT = Path(__file__).resolve().parent / "feature_audit.csv"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def safe_parse(value, fallback=None):
    if pd.isna(value):
        return fallback
    try:
        return literal_eval(str(value))
    except (ValueError, SyntaxError):
        return fallback


# ---------------------------------------------------------------------------
# Column classifications
# ---------------------------------------------------------------------------

# Model features: pre-launch, no leakage
MODEL_FEATURES = {
    "price": "Core business decision. Used directly and also converted into is_free flag and price_tier bins.",
    "required_age": "Age rating signal that affects audience reach. Used directly as a numeric feature.",
    "achievements": "Proxy for content depth and polish. Used directly as a numeric feature.",
    "genres": "Primary discovery signal on Steam. One-hot encoded into the top 15 most common genres.",
    "categories": "Platform capabilities like multiplayer, co-op, and workshop support. One-hot encoded into the top 12.",
    "tags": "Community-assigned descriptors more granular than genres. Top 20 tags encoded as binary features.",
    "release_date": "Decomposed into release_year, release_month, release_quarter, release_day_of_week, released_in_sale_month flag, and years_since_release.",
    "supported_languages": "Counted to get language_count, which reflects localization investment level.",
    "full_audio_languages": "Counted to get audio_language_count, a stronger signal of localization investment than text-only.",
    "windows": "Platform support flag. Combined with mac and linux into a single platform_count feature.",
    "mac": "Platform support flag. Combined with windows and linux into a single platform_count feature.",
    "linux": "Platform support flag. Combined with windows and mac into a single platform_count feature.",
    "developers": "Studio name used to compute developer_historical_success (past success rate) and developer_game_count (experience). Both computed using only prior games to avoid leakage.",
    "publishers": "Publisher name used to compute publisher_historical_success using only prior games.",
    "short_description": "Text length measured as description_length, a proxy for marketing effort.",
    "website": "Converted to has_website binary flag indicating whether a dedicated game website exists.",
}

# EDA reference: post-launch, for validation only (prefixed eda_ in output)
EDA_REFERENCE = {
    "estimated_owners": "Post-launch outcome — would cause temporal leakage",
    "peak_ccu": "Post-launch popularity metric",
    "average_playtime_forever": "Post-launch engagement — only observable after purchase",
    "average_playtime_2weeks": "Post-launch recent engagement",
    "median_playtime_forever": "Post-launch engagement (median)",
    "median_playtime_2weeks": "Post-launch recent engagement (median)",
    "metacritic_score": "Available days/weeks after launch — borderline but excluded for clean boundary",
    "dlc_count": "Post-launch content strategy decision",
    "discount": "Post-launch pricing action",
}

# Excluded: target leakage or metadata
EXCLUDED = {
    "appid": "Row identifier — not a feature",
    "name": "Row identifier — not a feature",
    "positive": "Direct component of success formula — target leakage",
    "negative": "Direct component of success formula — target leakage",
    "pct_pos_total": "Derived from positive/negative — target leakage",
    "pct_pos_recent": "Derived from review counts — target leakage",
    "num_reviews_total": "Inconsistent with positive+negative — also target leakage",
    "num_reviews_recent": "Recent review count — target leakage",
    "review_total_calc": "Engineered in Step 1 to compute target — target leakage",
    "positive_ratio_calc": "Engineered in Step 1 to compute target — target leakage",
    "has_reviews_calc": "Boolean from review count — target leakage",
    "recommendations": "Engagement metric correlated with target — target leakage",
    "user_score": "All zeros in dataset — no signal",
    "score_rank": "100% null — no data",
    "header_image": "URL — not a feature without image analysis",
    "screenshots": "URL list — not a feature without image analysis",
    "movies": "URL list — not a feature without video analysis",
    "detailed_description": "Full HTML text — not used without NLP (deferred)",
    "about_the_game": "Duplicate of detailed_description",
    "reviews": "Press review quotes — 60% null, unstructured",
    "support_email": "70% null — too sparse",
    "support_url": "20% null — low signal",
    "notes": "65% null — content warnings, low signal",
    "metacritic_url": "URL — not a feature",
    "packages": "Pricing structure JSON — complex, low marginal value",
    "success": "TARGET VARIABLE — not a feature",
}


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def column_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Audit every column: dtype, nulls, unique values, classification."""
    rows = []
    for col in df.columns:
        vals = df[col].dropna()
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        sample = str(vals.iloc[0])[:80] if len(vals) > 0 else "N/A"

        if col in MODEL_FEATURES:
            classification = "MODEL_FEATURE"
            reason = MODEL_FEATURES[col]
        elif col in EDA_REFERENCE:
            classification = "EDA_REFERENCE"
            reason = EDA_REFERENCE[col]
        elif col in EXCLUDED:
            classification = "EXCLUDED"
            reason = EXCLUDED[col]
        else:
            classification = "UNCLASSIFIED"
            reason = "Not in any classification list — review needed"

        rows.append({
            "column": col,
            "dtype": str(vals.dtype) if len(vals) > 0 else "empty",
            "null_count": null_count,
            "null_pct": round(null_pct, 1),
            "unique_values": vals.nunique(),
            "classification": classification,
            "reason": reason,
            "sample_value": sample,
        })
    return pd.DataFrame(rows)


def class_balance_analysis(df: pd.DataFrame) -> dict:
    """Analyze target variable distribution."""
    total = len(df)
    success = int(df["success"].sum())
    return {
        "total_games": total,
        "successful": success,
        "unsuccessful": total - success,
        "positive_pct": round(success / total * 100, 1),
        "imbalance_ratio": round((total - success) / success, 1) if success > 0 else float("inf"),
    }


def price_analysis(df: pd.DataFrame) -> dict:
    """Analyze price distribution and skewness."""
    prices = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    return {
        "free_count": int((prices == 0).sum()),
        "free_pct": round((prices == 0).sum() / len(df) * 100, 1),
        "mean": round(prices.mean(), 2),
        "median": round(prices.median(), 2),
        "max": round(prices.max(), 2),
        "std": round(prices.std(), 2),
    }


def missing_data_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Find columns with significant missing data."""
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    return pd.DataFrame({
        "column": nulls.index,
        "missing_count": nulls.values,
        "missing_pct": (nulls.values / len(df) * 100).round(1),
    })


def success_definition_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the 3-option comparison for validation."""
    valid = df["positive_ratio_calc"].notna() & (df["review_total_calc"] > 0)
    reviewed = df.loc[valid].copy()
    reviewed["positive_ratio_calc"] = pd.to_numeric(reviewed["positive_ratio_calc"], errors="coerce")
    reviewed["review_total_calc"] = pd.to_numeric(reviewed["review_total_calc"], errors="coerce")

    ratio_p75 = reviewed["positive_ratio_calc"].quantile(0.75)
    reviews_p75 = reviewed["review_total_calc"].quantile(0.75)

    options = {
        "Option 1 (Simple): ratio >= 90%": valid & (
            pd.to_numeric(df["positive_ratio_calc"], errors="coerce") >= 0.90
        ),
        "Option 2 (Balanced): ratio >= 80% AND reviews >= 50": valid & (
            pd.to_numeric(df["positive_ratio_calc"], errors="coerce") >= 0.80
        ) & (
            pd.to_numeric(df["review_total_calc"], errors="coerce") >= 50
        ),
        "Option 3 (Relative): top 25% in both": valid & (
            pd.to_numeric(df["positive_ratio_calc"], errors="coerce") >= ratio_p75
        ) & (
            pd.to_numeric(df["review_total_calc"], errors="coerce") >= reviews_p75
        ),
    }

    rows = []
    for name, mask in options.items():
        n = int(mask.sum())
        rows.append({
            "option": name,
            "successful_games": n,
            "pct_of_dataset": round(n / len(df) * 100, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    df: pd.DataFrame,
    audit: pd.DataFrame,
    balance: dict,
    price: dict,
    missing: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    """Generate the full strategy & validation markdown report."""

    # Count features by classification
    model_count = len(audit[audit["classification"] == "MODEL_FEATURE"])
    eda_count = len(audit[audit["classification"] == "EDA_REFERENCE"])
    excluded_count = len(audit[audit["classification"] == "EXCLUDED"])

    # Build model features table
    model_rows = audit[audit["classification"] == "MODEL_FEATURE"]
    model_table = "\n".join(
        f"| `{row['column']}` | {row['dtype']} | {row['null_pct']}% | {row['reason']} |"
        for _, row in model_rows.iterrows()
    )

    # Build EDA reference table
    eda_rows = audit[audit["classification"] == "EDA_REFERENCE"]
    eda_table = "\n".join(
        f"| `{row['column']}` | {row['reason']} |"
        for _, row in eda_rows.iterrows()
    )

    # Build excluded table
    excl_rows = audit[audit["classification"] == "EXCLUDED"]
    excl_table = "\n".join(
        f"| `{row['column']}` | {row['reason']} |"
        for _, row in excl_rows.iterrows()
    )

    # Build missing data table
    if len(missing) > 0:
        missing_table = "\n".join(
            f"| `{row['column']}` | {row['missing_count']:,} | {row['missing_pct']}% |"
            for _, row in missing.iterrows()
        )
    else:
        missing_table = "| (none) | 0 | 0% |"

    # Build comparison table
    comp_table = "\n".join(
        f"| {row['option']} | {row['successful_games']:,} | {row['pct_of_dataset']}% |"
        for _, row in comparison.iterrows()
    )

    report = f"""# Strategy & Validation Report
## Steam Games Success Prediction — Dataset Review & Feature Strategy

**Author:** Pujitha Attuluri
**Role:** Strategy & Validation Lead
**Dataset:** `games_march2025_cleaned_binary_success.csv` ({balance['total_games']:,} games, {len(audit)} columns)

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
{comp_table}

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

**Summary:** {model_count} model features | {eda_count} EDA reference | {excluded_count} excluded

### 3a. Model Features — pre-launch, no leakage ({model_count} columns)

These are the raw columns selected for modeling. During feature engineering, they will be transformed into ~49 model-ready features (one-hot encoding, counts, track records, temporal decomposition).

| Column | Type | Null % | Rationale |
|---|---|---:|---|
{model_table}

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

### 3b. EDA Reference — post-launch, validation only ({eda_count} columns)

Included in the output dataset with `eda_` prefix for exploratory analysis. Used to validate that model predictions correlate with real-world outcomes. **Never used as model inputs.**

| Column | Why EDA only |
|---|---|
{eda_table}

### 3c. Excluded — target leakage or metadata ({excluded_count} columns)

These columns are dropped entirely from the output.

| Column | Reason |
|---|---|
{excl_table}

---

## 4. Risk Analysis

### 4a. Class imbalance
- **Finding:** {balance['positive_pct']}% positive class ({balance['successful']:,} of {balance['total_games']:,}). Imbalance ratio: {balance['imbalance_ratio']}:1.
- **Risk:** Models will favor majority class (unsuccessful) for raw accuracy.
- **Mitigation:** `class_weight='balanced'` in tree models, SMOTE evaluation, stratified CV. ROC-AUC as primary metric.

### 4b. Price skewness
- **Finding:** {price['free_pct']}% of games are free. Mean ${price['mean']}, median ${price['median']}, max ${price['max']}.
- **Risk:** Heavy right skew with spike at zero.
- **Mitigation:** `price_tier` categorical encoding alongside raw price. `is_free` binary flag.

### 4c. Missing data
| Column | Missing | % |
|---|---:|---:|
{missing_table}
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


"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading dataset …")
    df = pd.read_csv(SOURCE_FILE, low_memory=False)
    print(f"  {len(df):,} games, {len(df.columns)} columns loaded.")

    print("Running column audit …")
    audit = column_audit(df)

    # Flag any unclassified columns
    unclassified = audit[audit["classification"] == "UNCLASSIFIED"]
    if len(unclassified) > 0:
        print(f"  WARNING: {len(unclassified)} unclassified columns:")
        for _, row in unclassified.iterrows():
            print(f"    - {row['column']}")
    else:
        print(f"  All {len(audit)} columns classified.")

    print("Analyzing class balance …")
    balance = class_balance_analysis(df)
    print(f"  Success: {balance['successful']:,} ({balance['positive_pct']}%) | "
          f"Imbalance ratio: {balance['imbalance_ratio']}:1")

    print("Analyzing price distribution …")
    price = price_analysis(df)
    print(f"  Free: {price['free_pct']}% | Mean: ${price['mean']} | Median: ${price['median']}")

    print("Analyzing missing data …")
    missing = missing_data_analysis(df)
    print(f"  {len(missing)} columns with nulls")

    print("Validating success definitions …")
    comparison = success_definition_comparison(df)
    for _, row in comparison.iterrows():
        print(f"  {row['option']}: {row['successful_games']:,} ({row['pct_of_dataset']}%)")

    print("Saving feature audit …")
    audit.to_csv(OUTPUT_AUDIT, index=False)
    print(f"  Saved → {OUTPUT_AUDIT}")

    print("Generating report …")
    report = generate_report(df, audit, balance, price, missing, comparison)
    OUTPUT_REPORT.write_text(report)
    print(f"  Saved → {OUTPUT_REPORT}")

    # Summary
    counts = audit["classification"].value_counts()
    print("\n--- Summary ---")
    print(f"  MODEL_FEATURE:  {counts.get('MODEL_FEATURE', 0)} columns")
    print(f"  EDA_REFERENCE:  {counts.get('EDA_REFERENCE', 0)} columns")
    print(f"  EXCLUDED:       {counts.get('EXCLUDED', 0)} columns")
    print(f"  UNCLASSIFIED:   {counts.get('UNCLASSIFIED', 0)} columns")
    print("\nDone.")


if __name__ == "__main__":
    main()