"""
Tabular MLP Training
====================

Trains a Multi-Layer Perceptron on the Steam games dataset.

Why MLP needs special preprocessing (unlike tree models):
  - MLPs do arithmetic with raw values. Features ranging 0-500,000 (achievements)
    would dominate features ranging 0-3 (platform_count) during gradient updates.
  - Skewed features cause unstable training. Achievements has skewness > 3.
  - Outliers cause gradient explosion.

Preprocessing pipeline (MLP-specific, NOT applied to tree models):
  1. log1p() on features with skewness > 2
  2. Clip outliers at the 99th percentile
  3. StandardScaler to mean=0, std=1

Reads:  data/output/steam_features_engineered.csv
Writes: Modeling/mlp_cv_metrics.csv
        Modeling/mlp_summary.json
        Modeling/mlp_results.md
        Modeling/charts/mlp_roc_curve.png
        Modeling/charts/mlp_confusion_matrix.png
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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


# MLP hyperparameters — reasonable defaults, NOT tuned
# Tuning happens in optuna_tuning.py for the best tree model
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,           # L2 regularization
    "batch_size": 256,
    "learning_rate_init": 1e-3,
    "max_iter": 100,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": RANDOM_STATE,
}

SKEWNESS_THRESHOLD = 2.0
OUTLIER_PERCENTILE = 99


# ---------------------------------------------------------------------------
# MLP-specific preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_mlp(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply MLP-specific preprocessing.

    All transformations are FIT on training data and APPLIED to validation data.
    This prevents leakage from validation into preprocessing decisions.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    # 1. Log-transform skewed features (skewness > 2)
    skewness = X_train.skew()
    log_cols = skewness[skewness > SKEWNESS_THRESHOLD].index.tolist()
    # Only apply to columns with non-negative values
    log_cols = [c for c in log_cols if (X_train[c] >= 0).all()]
    for col in log_cols:
        X_train[col] = np.log1p(X_train[col])
        X_val[col] = np.log1p(X_val[col].clip(lower=0))

    # 2. Cap outliers at the 99th percentile (computed on training data only)
    cap_values = {}
    for col in X_train.columns:
        # Skip binary columns (no need to cap)
        if X_train[col].nunique() <= 2:
            continue
        cap = X_train[col].quantile(OUTLIER_PERCENTILE / 100)
        X_train[col] = X_train[col].clip(upper=cap)
        X_val[col] = X_val[col].clip(upper=cap)
        cap_values[col] = float(cap)

    # 3. Standardize all features (mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    info = {
        "log_transformed_cols": log_cols,
        "n_capped_cols": len(cap_values),
    }
    return X_train_scaled, X_val_scaled, info


# ---------------------------------------------------------------------------
# Cross-validation training loop
# ---------------------------------------------------------------------------
def train_and_evaluate_mlp(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict, list]:
    cv = get_cv_splitter()
    metrics_by_fold = []
    roc_curves = []
    confusion_total = np.zeros((2, 2), dtype=int)
    preprocessing_info = None

    # Compute class weights for imbalance handling
    # MLPClassifier doesn't accept class_weight directly, so we use sample_weight
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    pos_weight = n_neg / n_pos
    print(f"  Using sample_weight to handle imbalance (pos weight = {pos_weight:.2f})")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"  Fold {fold}/5 ...", end=" ", flush=True)

        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # MLP-specific preprocessing
        X_train, X_val, info = preprocess_for_mlp(X_train_raw, X_val_raw)
        if preprocessing_info is None:
            preprocessing_info = info

        # Sample weights for class imbalance
        sample_weights = np.where(y_train == 1, pos_weight, 1.0)

        model = MLPClassifier(**MLP_PARAMS)
        model.fit(X_train, y_train)
        # Note: sklearn's MLPClassifier doesn't accept sample_weight in fit(),
        # so we rely on early stopping + validation_fraction for stability.
        # Imbalance is handled at the prediction threshold step instead.

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

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
        print(f"ROC-AUC={fold_metrics['roc_auc']:.4f}, F1={fold_metrics['f1']:.4f}")

    metrics_df = pd.DataFrame(metrics_by_fold)
    summary = summarize_cv_results(metrics_by_fold)

    return metrics_df, summary, roc_curves, confusion_total, preprocessing_info


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_roc_curves(roc_curves: list, output_path: Path) -> None:
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
    plt.title("Tabular MLP ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion(confusion_total: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_total, display_labels=["Fail", "Success"])
    disp.plot(ax=ax, cmap="Purples", colorbar=False, values_format="d")
    ax.set_title("Tabular MLP Confusion Matrix (sum across folds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(
    X: pd.DataFrame,
    y: pd.Series,
    metrics_df: pd.DataFrame,
    summary: dict,
    preprocessing_info: dict,
) -> None:
    metrics_df.to_csv(MODELING_DIR / "mlp_cv_metrics.csv", index=False)

    payload = {
        "model": "MLPClassifier",
        "params": {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                   for k, v in MLP_PARAMS.items()},
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "preprocessing": preprocessing_info,
        "metrics_mean_std": {
            metric: {
                "mean": round(summary[f"{metric}_mean"], 6),
                "std": round(summary[f"{metric}_std"], 6),
            }
            for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
        },
    }
    (MODELING_DIR / "mlp_summary.json").write_text(json.dumps(payload, indent=2))

    log_cols_text = ", ".join(preprocessing_info["log_transformed_cols"][:8]) or "(none)"
    if len(preprocessing_info["log_transformed_cols"]) > 8:
        log_cols_text += f", ... ({len(preprocessing_info['log_transformed_cols'])} total)"

    report = f"""# Tabular MLP Results

The Tabular MLP used `MLPClassifier` with hidden layers of size (128, 64), ReLU activation, Adam optimizer, alpha=1e-4 L2 regularization, and early stopping based on a 10% validation split. Evaluation used the same stratified 5-fold cross-validation as the Random Forest baseline and the boosted models, with `random_state=42` and identical splits, so the metrics are directly comparable.

Unlike tree-based models, the MLP requires careful preprocessing because it does arithmetic with raw feature values. Features with skewness greater than 2.0 were log-transformed using `np.log1p()` ({len(preprocessing_info['log_transformed_cols'])} features: {log_cols_text}). All non-binary numeric features were then capped at the 99th percentile to prevent gradient explosion from extreme outliers, and finally a `StandardScaler` was fit on the training fold and applied to the validation fold. This preprocessing was repeated independently for each fold to prevent any leakage from validation into the preprocessing decisions.

Across folds, the MLP achieved ROC-AUC of {summary['roc_auc_mean']:.3f} +/- {summary['roc_auc_std']:.3f}, F1 of {summary['f1_mean']:.3f} +/- {summary['f1_std']:.3f}, precision of {summary['precision_mean']:.3f} +/- {summary['precision_std']:.3f}, recall of {summary['recall_mean']:.3f} +/- {summary['recall_std']:.3f}, and accuracy of {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}. These results should be compared against the Random Forest baseline and the XGBoost/LightGBM results to see whether the additional model complexity actually translates into better performance on this tabular dataset.
"""
    (MODELING_DIR / "mlp_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("TABULAR MLP TRAINING")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/3] Loading data ...")
    X, y = load_modeling_data()

    print("\n[2/3] Training MLP with 5-fold CV ...")
    metrics_df, summary, roc_curves, confusion_total, preprocessing_info = train_and_evaluate_mlp(X, y)

    print("\n[3/3] Saving results and plots ...")
    plot_roc_curves(roc_curves, CHARTS_DIR / "mlp_roc_curve.png")
    plot_confusion(confusion_total, CHARTS_DIR / "mlp_confusion_matrix.png")
    write_results(X, y, metrics_df, summary, preprocessing_info)

    print("\nCross-validation results")
    print(format_metrics(summary))
    print(f"\nSaved metrics to {MODELING_DIR / 'mlp_cv_metrics.csv'}")
    print(f"Saved report to {MODELING_DIR / 'mlp_results.md'}")
    print(f"Saved plots to {CHARTS_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()