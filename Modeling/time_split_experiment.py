"""
Time-Based vs Random Split Experiment
======================================

Tests whether the temporal pattern in game release dates causes a meaningful
difference between random splits and time-based splits.

Why this matters: Our EDA showed success rates have shifted across years
(concept drift). If we train on a random sample and test on a random sample,
the model sees a mix of old and new games in both. But in real deployment,
we would predict success for new releases using past data — that's a
time-based split. If the time-based split performs much worse than random,
it means the model is partly learning patterns that don't transfer to
future games.

Reads:  data/output/steam_features_engineered.csv
Writes: Modeling/time_split_results.csv
        Modeling/time_split_results.md
        Modeling/charts/time_split_comparison.png

Requires: pip install xgboost
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

from _modeling_utils import (
    CHARTS_DIR,
    DATA_PATH,
    MODELING_DIR,
    RANDOM_STATE,
    ensure_dirs,
    get_cv_splitter,
    load_modeling_data,
)


SPLIT_YEAR = 2022


# ---------------------------------------------------------------------------
# Time-based split evaluation
# ---------------------------------------------------------------------------
def evaluate_time_split(X: pd.DataFrame, y: pd.Series, release_year: pd.Series) -> dict:
    """Train on games before SPLIT_YEAR, test on SPLIT_YEAR and later."""
    train_mask = release_year < SPLIT_YEAR
    test_mask = release_year >= SPLIT_YEAR

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} games (pre-{SPLIT_YEAR})")
    print(f"  Test:  {len(X_test):,} games ({SPLIT_YEAR}+)")
    print(f"  Train class balance: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test class balance:  {dict(y_test.value_counts().sort_index())}")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "split_type": "time_based",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Random split evaluation (for comparison)
# ---------------------------------------------------------------------------
def evaluate_random_split(X: pd.DataFrame, y: pd.Series) -> dict:
    """Use the standard 5-fold CV and average across folds."""
    cv = get_cv_splitter()
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    roc_list, f1_list = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        roc_list.append(roc_auc_score(y_val, y_prob))
        f1_list.append(f1_score(y_val, y_pred, zero_division=0))
        print(f"  Random split fold {fold}/5: ROC-AUC={roc_list[-1]:.4f}")

    return {
        "split_type": "random_stratified",
        "train_size": int(len(X) * 0.8),
        "test_size": int(len(X) * 0.2),
        "roc_auc": float(np.mean(roc_list)),
        "f1": float(np.mean(f1_list)),
        "roc_auc_std": float(np.std(roc_list, ddof=1)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(time_result: dict, random_result: dict, output_path: Path) -> None:
    splits = ["Random Stratified\n(5-fold CV)", f"Time-based\n(<{SPLIT_YEAR} train, {SPLIT_YEAR}+ test)"]
    roc_values = [random_result["roc_auc"], time_result["roc_auc"]]
    f1_values = [random_result["f1"], time_result["f1"]]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, roc_values, width, label="ROC-AUC", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, f1_values, width, label="F1", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Score")
    ax.set_title("Random Split vs Time-based Split (XGBoost)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars, vals in [(bars1, roc_values), (bars2, f1_values)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(time_result: dict, random_result: dict) -> None:
    df = pd.DataFrame([random_result, time_result])
    df.to_csv(MODELING_DIR / "time_split_results.csv", index=False)

    delta_roc = time_result["roc_auc"] - random_result["roc_auc"]
    pct_change = (delta_roc / random_result["roc_auc"]) * 100

    if abs(delta_roc) < 0.01:
        verdict = "No meaningful difference. The model generalizes equally well across both split strategies, which means there is no significant temporal drift affecting model performance and a random stratified split is appropriate for this project."
    elif delta_roc < -0.05:
        verdict = "Substantial degradation under time-based split. The model performs noticeably worse when forced to predict on future games using only past data. This indicates real temporal concept drift — the patterns that predict success have shifted over time. The final report should use time-based evaluation to give an honest picture of how the model would perform in real deployment."
    elif delta_roc < 0:
        verdict = "Slight degradation under time-based split. There is a small temporal effect, but it is not severe. Random stratified CV is acceptable as the primary evaluation, with the time-based result reported as a sanity check on real-world deployment performance."
    else:
        verdict = "Time-based split actually performs better. This is unusual and may indicate that the older training games have cleaner labels or that the test set happens to contain games similar to the training distribution. Worth investigating but not a cause for concern."

    report = f"""# Time-based vs Random Split Comparison

The EDA revealed a temporal pattern in success rates across release years. This experiment tests whether that temporal pattern causes a meaningful difference between random stratified splits and time-based splits, using the same XGBoost model with `scale_pos_weight` for imbalance handling.

The random stratified split was evaluated using the standard 5-fold cross-validation that every other model in this project uses, with results averaged across folds. The time-based split trained on games released before {SPLIT_YEAR} ({random_result['train_size']:,} games) and tested on games released in {SPLIT_YEAR} or later ({time_result['test_size']:,} games). Both splits used identical model hyperparameters and the same `random_state=42`.

| Split type | Train size | Test size | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| Random stratified (5-fold CV) | {random_result['train_size']:,} | {random_result['test_size']:,} | {random_result['roc_auc']:.4f} | {random_result['f1']:.4f} |
| Time-based ({SPLIT_YEAR} cutoff) | {time_result['train_size']:,} | {time_result['test_size']:,} | {time_result['roc_auc']:.4f} | {time_result['f1']:.4f} |

The time-based split produced a ROC-AUC change of {delta_roc:+.4f} ({pct_change:+.2f}%) compared to the random stratified split. {verdict}

For the final modeling pipeline, the random stratified 5-fold CV is the primary evaluation strategy because it produces the most stable estimates and matches the convention for tabular classification benchmarks. The time-based result is reported as a secondary check to demonstrate awareness of potential concept drift and to give the reader an honest picture of how the model would perform if deployed on truly unseen future games.
"""
    (MODELING_DIR / "time_split_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("TIME-BASED VS RANDOM SPLIT EXPERIMENT")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/4] Loading data ...")
    X, y = load_modeling_data()

    # Read release_year from the raw CSV (it's in X but we want it explicit)
    if "release_year" not in X.columns:
        raise ValueError(
            "release_year column not found in features. The time-based split needs it."
        )
    release_year = X["release_year"].copy()
    print(f"  release_year range: {int(release_year.min())} to {int(release_year.max())}")

    print("\n[2/4] Running random stratified split (5-fold CV) ...")
    random_result = evaluate_random_split(X, y)

    print(f"\n[3/4] Running time-based split (pre-{SPLIT_YEAR} train, {SPLIT_YEAR}+ test) ...")
    time_result = evaluate_time_split(X, y, release_year)

    print("\n[4/4] Saving results and plot ...")
    plot_comparison(time_result, random_result, CHARTS_DIR / "time_split_comparison.png")
    write_results(time_result, random_result)

    print("\nFinal comparison:")
    print(f"  Random stratified: ROC-AUC={random_result['roc_auc']:.4f}, F1={random_result['f1']:.4f}")
    print(f"  Time-based:        ROC-AUC={time_result['roc_auc']:.4f}, F1={time_result['f1']:.4f}")
    print(f"  Delta: {time_result['roc_auc'] - random_result['roc_auc']:+.4f}")

    print(f"\nSaved results to {MODELING_DIR / 'time_split_results.csv'}")
    print(f"Saved report to {MODELING_DIR / 'time_split_results.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()