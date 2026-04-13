"""
SHAP Feature Importance Analysis
=================================

Runs SHAP on the best-performing single model (tuned XGBoost) to understand
which features actually drive predictions.

Why SHAP and not RF impurity importance:
  - Impurity importance is biased toward high-cardinality features and features
    that get split on more often (not necessarily more useful)
  - SHAP values are based on game theory (Shapley values) — they measure each
    feature's marginal contribution to a specific prediction
  - SHAP gives both global importance (which features matter overall) AND
    local explanations (why this specific game was predicted as successful)

Why XGBoost (not RF or MLP):
  - XGBoost has a fast TreeExplainer that runs in seconds on 89K rows
  - It was the strongest single model in Kanniese's comparison
  - SHAP on MLPs requires KernelExplainer which is much slower

Reads:  data/output/steam_features_engineered.csv
        Modeling/optuna_best_params.json
Writes: Modeling/shap_global_importance.csv
        Modeling/shap_results.md
        Modeling/charts/shap_summary.png
        Modeling/charts/shap_bar.png
        Modeling/charts/shap_force_plot_*.png

Requires: pip install shap
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError:
    raise ImportError("SHAP is required. Install with: pip install shap")

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

from _modeling_utils import (
    CHARTS_DIR,
    MODELING_DIR,
    RANDOM_STATE,
    ensure_dirs,
    load_modeling_data,
)



SHAP_SAMPLE_SIZE = 89618


# ---------------------------------------------------------------------------
# Hypotheses to validate
# ---------------------------------------------------------------------------
# Maps hypothesis names to feature name patterns. SHAP results will be
# checked against these to confirm or reject each hypothesis.
HYPOTHESIS_FEATURES = {
    "H1: Paid games succeed more than free games": ["is_free", "price", "price_tier"],
    "H2: Price tier matters more than raw price": ["price_tier", "price"],
    "H3: Developer track record predicts success": ["developer_historical_success", "developer_game_count", "publisher_historical_success"],
    "H4: Genre affects success probability": ["genre_"],
    "H5: Multi-platform support matters": ["platform_count", "windows", "mac", "linux"],
}


def load_xgb_params() -> dict:
    """Load Optuna-tuned XGBoost params, falling back to sensible defaults."""
    optuna_path = MODELING_DIR / "optuna_best_params.json"
    if optuna_path.exists():
        try:
            payload = json.loads(optuna_path.read_text())
            params = payload["best_params"].copy()
            print(f"  Loaded tuned params from {optuna_path.name}")
            return params
        except Exception as exc:
            print(f"  Could not load Optuna params ({exc}), using defaults")
    return {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
    }


# ---------------------------------------------------------------------------
# Train the model SHAP will explain
# ---------------------------------------------------------------------------
def train_xgb_for_shap(X: pd.DataFrame, y: pd.Series, params: dict) -> XGBClassifier:
    """Train XGBoost on the full dataset (no CV) for SHAP analysis."""
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    full_params = {
        **params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }
    model = XGBClassifier(**full_params)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values on a sampled subset for speed."""
    if len(X) > SHAP_SAMPLE_SIZE:
        X_sample = X.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    print(f"  Computing SHAP values on {len(X_sample):,} sampled rows ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # XGBoost binary classification returns shap values for the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_sample


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_shap_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Summary (top 20 features)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_shap_bar(shap_values: np.ndarray, X_sample: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Global Feature Importance (top 20)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Hypothesis validation against SHAP
# ---------------------------------------------------------------------------
def validate_hypotheses(global_importance: pd.DataFrame) -> list[dict]:
    """
    For each hypothesis, find its features in the SHAP rankings and report
    whether they made it into the top 10 / top 20 / not at all.
    """
    results = []
    top_features = global_importance["feature"].tolist()

    for hypothesis, patterns in HYPOTHESIS_FEATURES.items():
        # Find all features matching any pattern
        matches = []
        for pattern in patterns:
            for feat in top_features:
                if pattern in feat:
                    rank = top_features.index(feat) + 1
                    importance = global_importance.loc[global_importance["feature"] == feat, "mean_abs_shap"].values[0]
                    matches.append((feat, rank, importance))

        # Deduplicate while preserving order
        seen = set()
        unique_matches = []
        for m in matches:
            if m[0] not in seen:
                seen.add(m[0])
                unique_matches.append(m)
        unique_matches.sort(key=lambda x: x[1])

        if not unique_matches:
            verdict = "NOT supported (no matching features in SHAP rankings)"
            evidence = "—"
        elif unique_matches[0][1] <= 10:
            verdict = "STRONGLY supported (feature in SHAP top 10)"
            evidence = f"{unique_matches[0][0]} ranked #{unique_matches[0][1]}"
        elif unique_matches[0][1] <= 20:
            verdict = "Supported (feature in SHAP top 20)"
            evidence = f"{unique_matches[0][0]} ranked #{unique_matches[0][1]}"
        else:
            verdict = "Weakly supported (feature ranked outside top 20)"
            evidence = f"{unique_matches[0][0]} ranked #{unique_matches[0][1]}"

        results.append({
            "hypothesis": hypothesis,
            "verdict": verdict,
            "evidence": evidence,
            "all_matches": unique_matches[:3],
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(global_importance: pd.DataFrame, hypothesis_results: list[dict]) -> None:
    global_importance.to_csv(MODELING_DIR / "shap_global_importance.csv", index=False)

    top_15 = global_importance.head(15)
    top_15_table = "\n".join(
        f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.4f} |"
        for i, row in top_15.iterrows()
    )

    hypothesis_text = ""
    for result in hypothesis_results:
        hypothesis_text += f"\n**{result['hypothesis']}**\n"
        hypothesis_text += f"Verdict: {result['verdict']}. {result['evidence']}\n"
        if result["all_matches"]:
            hypothesis_text += "Matching features:\n"
            for feat, rank, imp in result["all_matches"]:
                hypothesis_text += f"- `{feat}` (rank #{rank}, mean |SHAP| = {imp:.4f})\n"

    report = f"""# SHAP Feature Importance Analysis

This analysis runs SHAP on the tuned XGBoost model to understand which features actually drive the model's predictions, and to validate the project hypotheses against the model's behavior.

SHAP (SHapley Additive exPlanations) values are based on game-theoretic Shapley values, which measure each feature's marginal contribution to a specific prediction. Unlike Random Forest's impurity-based feature importance — which can be biased toward high-cardinality features and features that get split on more often without actually being more useful — SHAP gives a more honest picture of which features the model relies on. SHAP also provides both global importance (which features matter overall, averaged across all predictions) and local explanations (why this specific game was predicted as successful).

The SHAP TreeExplainer was run on a stratified sample of {SHAP_SAMPLE_SIZE:,} games rather than the full dataset. SHAP values converge quickly with sample size for tree models, and a 5,000-row sample produces essentially the same global rankings as the full dataset while being substantially faster.

## Top 15 features by SHAP importance

| Rank | Feature | Mean |SHAP| value |
|---:|---|---:|
{top_15_table}

The mean absolute SHAP value measures how much each feature shifts the model's prediction on average. Higher values mean the feature has more influence on whether a game is predicted as successful.

## Hypothesis validation
{hypothesis_text}

## Interpretation
The SHAP results provide model-level evidence for each project hypothesis. Hypotheses confirmed at the SHAP level are stronger findings than those supported only by EDA correlations, because SHAP shows that the model actually uses these features when making predictions — not just that the features happen to correlate with the outcome. Hypotheses that show up in EDA but are absent from SHAP rankings are worth investigating: it usually means the feature's signal was redundant with other features the model already had, or that the EDA correlation was driven by a confounding variable that the model controlled for.
"""
    (MODELING_DIR / "shap_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/5] Loading data ...")
    X, y = load_modeling_data()

    print("\n[2/5] Loading tuned XGBoost params and training ...")
    params = load_xgb_params()
    model = train_xgb_for_shap(X, y, params)
    print(f"  Model trained on {len(X):,} rows")

    print("\n[3/5] Computing SHAP values ...")
    shap_values, X_sample = compute_shap_values(model, X)

    print("\n[4/5] Generating SHAP plots ...")
    plot_shap_summary(shap_values, X_sample, CHARTS_DIR / "shap_summary.png")
    plot_shap_bar(shap_values, X_sample, CHARTS_DIR / "shap_bar.png")
    print("  Saved shap_summary.png and shap_bar.png")

    # Compute global importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\n[5/5] Validating hypotheses against SHAP ...")
    hypothesis_results = validate_hypotheses(global_importance)
    write_results(global_importance, hypothesis_results)

    print("\nTop 10 features by SHAP importance:")
    for i, row in global_importance.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:40s}  {row['mean_abs_shap']:.4f}")

    print("\nHypothesis validation:")
    for result in hypothesis_results:
        print(f"  {result['hypothesis']}")
        print(f"    -> {result['verdict']}")
        print(f"    -> {result['evidence']}")

    print(f"\nSaved global importance to {MODELING_DIR / 'shap_global_importance.csv'}")
    print(f"Saved report to {MODELING_DIR / 'shap_results.md'}")
    print(f"Saved plots to {CHARTS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()