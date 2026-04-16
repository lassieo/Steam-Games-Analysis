"""
Random Forest Baseline
=======================

The first model in the ladder. Uses the engineered dataset from the EDA
pipeline and the shared CV splits from
_modeling_utils.py to ensure all models compare fairly.

Reads:  data/output/steam_features_engineered.csv
Writes: Modeling/rf_baseline_cv_metrics.csv
        Modeling/rf_baseline_summary.json
        Modeling/rf_baseline_results.md
        Modeling/charts/rf_baseline_roc_curve.png
        Modeling/charts/rf_baseline_confusion_matrix.png
        Modeling/charts/rf_baseline_feature_importance.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


# RF baseline parameters — reasonable defaults, NOT tuned
# The point of the baseline is to establish the floor, not to optimize
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Cross-validation training
# ---------------------------------------------------------------------------
def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    cv = get_cv_splitter()
    metrics_by_fold = []
    roc_curves = []
    confusion_total = np.zeros((2, 2), dtype=int)
    last_model = None

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"  Fold {fold}/5 ...", end=" ", flush=True)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(**RF_PARAMS)
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

    metrics_df = pd.DataFrame(metrics_by_fold)
    summary = summarize_cv_results(metrics_by_fold)

    return metrics_df, summary, roc_curves, confusion_total, last_model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc_curves(roc_curves, output_path: Path) -> None:
    mean_fpr = np.linspace(0, 1, 200)
    interpolated_tprs = []

    plt.figure(figsize=(8, 6))
    for idx, (fpr, tpr, fold_auc) in enumerate(roc_curves, start=1):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)
        plt.plot(fpr, tpr, alpha=0.35, label=f"Fold {idx} (AUC={fold_auc:.3f})")

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color="black", linewidth=2.5, label=f"Mean ROC (AUC={mean_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest Baseline ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion(confusion_total: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_total, display_labels=["Fail", "Success"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Random Forest Baseline Confusion Matrix (sum across folds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(model: RandomForestClassifier, feature_names: list, output_path: Path) -> None:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    top_features = importance_df.head(20).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#2c7fb8")
    plt.xlabel("Impurity Importance")
    plt.title("Random Forest Baseline Top 20 Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    importance_df.to_csv(output_path.with_suffix(".csv"), index=False)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(X: pd.DataFrame, y: pd.Series, metrics_df: pd.DataFrame, summary: dict) -> None:
    metrics_df.to_csv(MODELING_DIR / "rf_baseline_cv_metrics.csv", index=False)

    payload = {
        "model": "RandomForestClassifier",
        "params": RF_PARAMS,
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "metrics_mean_std": {
            metric: {
                "mean": round(summary[f"{metric}_mean"], 6),
                "std": round(summary[f"{metric}_std"], 6),
            }
            for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
        },
    }
    (MODELING_DIR / "rf_baseline_summary.json").write_text(json.dumps(payload, indent=2))

    # Read top features for the report
    top_features_path = CHARTS_DIR / "rf_baseline_feature_importance.csv"
    if top_features_path.exists():
        top_features = pd.read_csv(top_features_path).head(5)
        top_feature_text = ", ".join(top_features["feature"].tolist())
    else:
        top_feature_text = "(see feature importance chart)"

    report = f"""# Random Forest Baseline Results

The baseline model used `RandomForestClassifier` with `n_estimators=300`, `max_depth=None`, `class_weight='balanced'`, and `random_state=42`. Evaluation used stratified 5-fold cross-validation on the engineered dataset at `data/output/steam_features_engineered.csv`, with ID columns (`appid`, `name`) and any `eda_` prefixed columns removed before training. The final modeling matrix contained {X.shape[0]:,} rows and {X.shape[1]:,} usable features, with class balance of {int((y == 0).sum()):,} unsuccessful games versus {int((y == 1).sum()):,} successful games.

Across folds, the baseline achieved ROC-AUC of {summary['roc_auc_mean']:.3f} +/- {summary['roc_auc_std']:.3f}, F1 of {summary['f1_mean']:.3f} +/- {summary['f1_std']:.3f}, precision of {summary['precision_mean']:.3f} +/- {summary['precision_std']:.3f}, recall of {summary['recall_mean']:.3f} +/- {summary['recall_std']:.3f}, and accuracy of {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}. Because only about 16% of games are labeled successful, using `class_weight='balanced'` was important for keeping the model from collapsing into an almost-all-negative classifier. The resulting recall and F1 are more meaningful than raw accuracy alone for this project.

The strongest impurity-based feature importance signals came from {top_feature_text}. These are useful baseline signals, but they should still be interpreted cautiously because impurity importance can favor higher-cardinality or more frequently splitting variables. SHAP analysis on the best model later in the pipeline will give a more reliable picture of feature importance. This run establishes the floor for later comparisons, while Optuna tuning and stronger boosted models will be evaluated against the same cross-validation setup.
"""
    (MODELING_DIR / "rf_baseline_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("RANDOM FOREST BASELINE")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/3] Loading data ...")
    X, y = load_modeling_data()

    print("\n[2/3] Training Random Forest with 5-fold CV ...")
    metrics_df, summary, roc_curves, confusion_total, last_model = train_and_evaluate(X, y)

    print("\n[3/3] Saving results and plots ...")
    plot_roc_curves(roc_curves, CHARTS_DIR / "rf_baseline_roc_curve.png")
    plot_confusion(confusion_total, CHARTS_DIR / "rf_baseline_confusion_matrix.png")
    if last_model is not None:
        plot_feature_importance(last_model, X.columns.tolist(), CHARTS_DIR / "rf_baseline_feature_importance.png")
    write_results(X, y, metrics_df, summary)

    print("\nCross-validation results")
    print(format_metrics(summary))
    print(f"\nSaved metrics to {MODELING_DIR / 'rf_baseline_cv_metrics.csv'}")
    print(f"Saved report to {MODELING_DIR / 'rf_baseline_results.md'}")
    print(f"Saved plots to {CHARTS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()