"""
XGBoost & LightGBM Training
============================

The primary boosted tree models — second tier of the model ladder.
Both use the same data, splits, and metrics as the RF baseline so the
comparison is fair.

Why two boosted models: XGBoost and LightGBM are the two dominant
gradient boosting libraries. They have slightly different algorithms
(level-wise vs leaf-wise tree growth) and we want to know which performs
better on this dataset before tuning.

Reads:  data/output/steam_features_engineered.csv
Writes: Modeling/xgb_lgbm_cv_metrics.csv
        Modeling/xgb_lgbm_summary.json
        Modeling/xgb_lgbm_results.md
        Modeling/charts/xgb_lgbm_roc_curve.png
        Modeling/charts/xgb_lgbm_confusion_matrix.png

Requires: pip install xgboost lightgbm
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm")

from _modeling_utils import (
    CHARTS_DIR,
    MODELING_DIR,
    RANDOM_STATE,
    ensure_dirs,
    format_metrics,
    get_cv_splitter,
    load_modeling_data,
    summarize_cv_results,
)


# ---------------------------------------------------------------------------
# Cross-validation training
# ---------------------------------------------------------------------------
def evaluate_model(model_class, model_kwargs: dict, X: pd.DataFrame, y: pd.Series, model_name: str):
    """Train a single model class across all CV folds and return metrics."""
    cv = get_cv_splitter()
    metrics_by_fold = []
    roc_curves = []
    confusion_total = np.zeros((2, 2), dtype=int)
    last_model = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"  {model_name} Fold {fold}/5 ...", end=" ", flush=True)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_curves.append((fpr, tpr, auc(fpr, tpr)))
        confusion_total += confusion_matrix(y_val, y_pred, labels=[0, 1])

        fold_metrics = {
            "fold": fold,
            "roc_auc": roc_auc_score(y_val, y_prob),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_val, y_pred),
        }
        metrics_by_fold.append(fold_metrics)
        last_model = model
        print(f"ROC-AUC={fold_metrics['roc_auc']:.4f}, F1={fold_metrics['f1']:.4f}")

    return metrics_by_fold, roc_curves, confusion_total, last_model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_combined_roc(xgb_roc, lgbm_roc, output_path: Path) -> None:
    """Plot mean ROC for both XGBoost and LightGBM on the same chart."""
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 200)

    for label, roc_data, color in [("XGBoost", xgb_roc, "#1f77b4"), ("LightGBM", lgbm_roc, "#ff7f0e")]:
        interpolated = []
        for fpr, tpr, _ in roc_data:
            interp = np.interp(mean_fpr, fpr, tpr)
            interp[0] = 0.0
            interpolated.append(interp)
        mean_tpr = np.mean(interpolated, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, linewidth=2, label=f"{label} (AUC={mean_auc:.3f})", color=color)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost vs LightGBM ROC Curves (mean across folds)")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion(confusion_total: np.ndarray, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_total, display_labels=["Fail", "Success"])
    disp.plot(ax=ax, cmap="Oranges", colorbar=False, values_format="d")
    ax.set_title(f"{model_name} Confusion Matrix (sum across folds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(X: pd.DataFrame, y: pd.Series, xgb_summary: dict, lgbm_summary: dict,
                  combined_metrics: pd.DataFrame) -> None:
    combined_metrics.to_csv(MODELING_DIR / "xgb_lgbm_cv_metrics.csv", index=False)

    payload = {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "xgboost": {
            metric: {
                "mean": round(xgb_summary[f"{metric}_mean"], 6),
                "std": round(xgb_summary[f"{metric}_std"], 6),
            }
            for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
        },
        "lightgbm": {
            metric: {
                "mean": round(lgbm_summary[f"{metric}_mean"], 6),
                "std": round(lgbm_summary[f"{metric}_std"], 6),
            }
            for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
        },
    }
    (MODELING_DIR / "xgb_lgbm_summary.json").write_text(json.dumps(payload, indent=2))

    winner = "XGBoost" if xgb_summary["roc_auc_mean"] >= lgbm_summary["roc_auc_mean"] else "LightGBM"

    report = f"""# XGBoost and LightGBM Results

The two gradient boosting models were trained on the same engineered dataset (`data/output/steam_features_engineered.csv`) as the Random Forest baseline, using identical stratified 5-fold cross-validation splits with `random_state=42`. Both models received imbalance handling: XGBoost via `scale_pos_weight = (n_neg / n_pos)`, and LightGBM via `is_unbalance=True`. No hyperparameter tuning was applied at this stage — the goal here is to establish the strongest untuned tree model before passing it to Optuna.

The final modeling matrix contained {X.shape[0]:,} rows and {X.shape[1]:,} usable features, with class balance of {int((y == 0).sum()):,} unsuccessful games versus {int((y == 1).sum()):,} successful games.

**XGBoost results:** ROC-AUC of {xgb_summary['roc_auc_mean']:.3f} +/- {xgb_summary['roc_auc_std']:.3f}, F1 of {xgb_summary['f1_mean']:.3f} +/- {xgb_summary['f1_std']:.3f}, precision of {xgb_summary['precision_mean']:.3f} +/- {xgb_summary['precision_std']:.3f}, recall of {xgb_summary['recall_mean']:.3f} +/- {xgb_summary['recall_std']:.3f}, accuracy of {xgb_summary['accuracy_mean']:.3f} +/- {xgb_summary['accuracy_std']:.3f}.

**LightGBM results:** ROC-AUC of {lgbm_summary['roc_auc_mean']:.3f} +/- {lgbm_summary['roc_auc_std']:.3f}, F1 of {lgbm_summary['f1_mean']:.3f} +/- {lgbm_summary['f1_std']:.3f}, precision of {lgbm_summary['precision_mean']:.3f} +/- {lgbm_summary['precision_std']:.3f}, recall of {lgbm_summary['recall_mean']:.3f} +/- {lgbm_summary['recall_std']:.3f}, accuracy of {lgbm_summary['accuracy_mean']:.3f} +/- {lgbm_summary['accuracy_std']:.3f}.

{winner} performed slightly better and will be the model passed to Optuna hyperparameter tuning in the next stage. Both models substantially outperform what would be expected from random guessing (ROC-AUC of 0.5) and demonstrate that the engineered features carry meaningful predictive signal. The next step is to tune the stronger model with Bayesian optimization to extract additional performance, then combine all base models in a stacking ensemble.
"""
    (MODELING_DIR / "xgb_lgbm_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("XGBOOST AND LIGHTGBM TRAINING")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/4] Loading data ...")
    X, y = load_modeling_data()

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"  scale_pos_weight = {scale_pos_weight:.3f}")

    print("\n[2/4] Training XGBoost with 5-fold CV ...")
    xgb_kwargs = {
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }
    xgb_metrics, xgb_roc, xgb_cm, _ = evaluate_model(XGBClassifier, xgb_kwargs, X, y, "XGB")
    xgb_summary = summarize_cv_results(xgb_metrics)

    print("\n[3/4] Training LightGBM with 5-fold CV ...")
    lgbm_kwargs = {
        "is_unbalance": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgbm_metrics, lgbm_roc, lgbm_cm, _ = evaluate_model(LGBMClassifier, lgbm_kwargs, X, y, "LGBM")
    lgbm_summary = summarize_cv_results(lgbm_metrics)

    print("\n[4/4] Saving results and plots ...")
    plot_combined_roc(xgb_roc, lgbm_roc, CHARTS_DIR / "xgb_lgbm_roc_curve.png")
    plot_confusion(xgb_cm, "XGBoost", CHARTS_DIR / "xgb_confusion_matrix.png")
    plot_confusion(lgbm_cm, "LightGBM", CHARTS_DIR / "lgbm_confusion_matrix.png")

    # Combine metrics into one CSV for easy comparison later
    xgb_df = pd.DataFrame(xgb_metrics).add_prefix("xgb_")
    xgb_df = xgb_df.rename(columns={"xgb_fold": "fold"})
    lgbm_df = pd.DataFrame(lgbm_metrics).add_prefix("lgbm_")
    lgbm_df = lgbm_df.drop(columns=["lgbm_fold"])
    combined = pd.concat([xgb_df, lgbm_df], axis=1)
    write_results(X, y, xgb_summary, lgbm_summary, combined)

    print("\nXGBoost results")
    print(format_metrics(xgb_summary))
    print("\nLightGBM results")
    print(format_metrics(lgbm_summary))

    print(f"\nSaved metrics to {MODELING_DIR / 'xgb_lgbm_cv_metrics.csv'}")
    print(f"Saved report to {MODELING_DIR / 'xgb_lgbm_results.md'}")
    print(f"Saved plots to {CHARTS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()