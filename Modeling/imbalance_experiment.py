"""
Class Imbalance Experiment
===========================

Compares three approaches to handling the 16.3% positive class imbalance:
  1. None — train XGBoost as-is, no balancing
  2. Weighted — use scale_pos_weight to penalize minority class errors
  3. SMOTE — synthetic oversampling of the minority class on training folds only

Critical: SMOTE is applied INSIDE the CV loop, only to the training fold,
never to validation. Applying SMOTE before splitting would leak synthetic
samples into validation and inflate the metrics.

Reads:  data/output/steam_features_engineered.csv
Writes: Modeling/imbalance_experiment_results.csv
        Modeling/imbalance_experiment_results.md
        Modeling/charts/imbalance_comparison.png

Requires: pip install xgboost imbalanced-learn
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

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "imbalanced-learn is required. Install with: pip install imbalanced-learn"
    )

from _modeling_utils import (
    CHARTS_DIR,
    MODELING_DIR,
    RANDOM_STATE,
    ensure_dirs,
    get_cv_splitter,
    load_modeling_data,
)


# ---------------------------------------------------------------------------
# Run a single experiment mode
# ---------------------------------------------------------------------------
def run_experiment(mode: str, X: pd.DataFrame, y: pd.Series, scale_pos_weight: float) -> dict:
    """
    mode: 'none', 'weighted', or 'smote'
    Returns dict with mean ROC-AUC, mean F1, and per-fold lists.
    """
    cv = get_cv_splitter()
    roc_list, f1_list = [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE only to the training fold (never validation)
        if mode == "smote":
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Build model with appropriate balancing
        if mode == "weighted":
            model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            )
        else:
            # Both 'none' and 'smote' use unweighted XGBoost
            model = XGBClassifier(
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
        print(f"  {mode:10s} Fold {fold}/5: ROC-AUC={roc_list[-1]:.4f}, F1={f1_list[-1]:.4f}")

    return {
        "mode": mode,
        "roc_auc_mean": float(np.mean(roc_list)),
        "roc_auc_std": float(np.std(roc_list, ddof=1)),
        "f1_mean": float(np.mean(f1_list)),
        "f1_std": float(np.std(f1_list, ddof=1)),
        "roc_per_fold": roc_list,
        "f1_per_fold": f1_list,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(results: list, output_path: Path) -> None:
    modes = [r["mode"] for r in results]
    roc_means = [r["roc_auc_mean"] for r in results]
    roc_stds = [r["roc_auc_std"] for r in results]
    f1_means = [r["f1_mean"] for r in results]
    f1_stds = [r["f1_std"] for r in results]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, roc_means, width, yerr=roc_stds, capsize=4,
                   label="ROC-AUC", color="#1f77b4")
    bars2 = ax.bar(x + width / 2, f1_means, width, yerr=f1_stds, capsize=4,
                   label="F1", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in modes])
    ax.set_ylabel("Score")
    ax.set_title("Class Imbalance Strategy Comparison (XGBoost)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for bars, means in [(bars1, roc_means), (bars2, f1_means)]:
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(results: list) -> None:
    df = pd.DataFrame([{
        "mode": r["mode"],
        "roc_auc_mean": r["roc_auc_mean"],
        "roc_auc_std": r["roc_auc_std"],
        "f1_mean": r["f1_mean"],
        "f1_std": r["f1_std"],
    } for r in results])
    df.to_csv(MODELING_DIR / "imbalance_experiment_results.csv", index=False)

    # Find the best mode
    best_roc_mode = max(results, key=lambda r: r["roc_auc_mean"])["mode"]
    best_f1_mode = max(results, key=lambda r: r["f1_mean"])["mode"]

    table_rows = "\n".join(
        f"| {r['mode'].title()} | {r['roc_auc_mean']:.4f} +/- {r['roc_auc_std']:.4f} | {r['f1_mean']:.4f} +/- {r['f1_std']:.4f} |"
        for r in results
    )

    report = f"""# Class Imbalance Experiment Results

The Steam dataset has a 16.3% positive class (14,617 successful games out of 89,618 total). This experiment compares three strategies for handling that imbalance, all using XGBoost as the base model and the same stratified 5-fold cross-validation as every other model in the project.

The three strategies tested were: no balancing (let XGBoost train on the raw distribution), weighted (use `scale_pos_weight` to penalize minority class errors more heavily), and SMOTE (synthetic minority oversampling applied only to the training fold within each CV iteration). The SMOTE application is critical: synthetic samples must never appear in the validation fold, or the metrics become inflated by leakage. This implementation applies SMOTE inside the CV loop, on the training data only, generating new training data fresh for each fold.

| Strategy | ROC-AUC | F1 |
|---|---|---|
{table_rows}

The best ROC-AUC came from the **{best_roc_mode}** strategy and the best F1 came from the **{best_f1_mode}** strategy. ROC-AUC is the primary metric for this project because it is invariant to the prediction threshold, but F1 is also informative because it directly reflects the precision-recall tradeoff at the default 0.5 threshold.

The general pattern in tabular classification with moderate imbalance (10-20% minority class) is that `scale_pos_weight` typically matches or beats SMOTE while being far simpler and faster. SMOTE can help in extreme imbalance situations (under 5% minority class) but tends to introduce noise when there are already plenty of minority examples to learn from. The recommendation for this project is to use `scale_pos_weight` in tree-based models because it is computationally cheap, mathematically clean, and produces results competitive with more expensive resampling methods.
"""
    (MODELING_DIR / "imbalance_experiment_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("CLASS IMBALANCE EXPERIMENT")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/3] Loading data ...")
    X, y = load_modeling_data()

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"  scale_pos_weight = {scale_pos_weight:.3f}")

    print("\n[2/3] Running three imbalance strategies ...")
    results = []
    for mode in ["none", "weighted", "smote"]:
        print(f"\n  --- {mode.upper()} ---")
        result = run_experiment(mode, X, y, scale_pos_weight)
        results.append(result)

    print("\n[3/3] Saving results and plot ...")
    plot_comparison(results, CHARTS_DIR / "imbalance_comparison.png")
    write_results(results)

    print("\nFinal comparison:")
    for r in results:
        print(f"  {r['mode']:10s}  ROC-AUC={r['roc_auc_mean']:.4f}, F1={r['f1_mean']:.4f}")

    print(f"\nSaved results to {MODELING_DIR / 'imbalance_experiment_results.csv'}")
    print(f"Saved report to {MODELING_DIR / 'imbalance_experiment_results.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()