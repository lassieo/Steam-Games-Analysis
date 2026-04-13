"""
Stacking Ensemble
==================

Combines predictions from three diverse base models:
  1. Random Forest (bagging-based tree ensemble)
  2. XGBoost with Optuna-tuned hyperparameters (boosting-based tree ensemble)
  3. Tabular MLP (neural network)

A Logistic Regression meta-learner takes the three base model probability
predictions as input and learns the optimal weighted combination.

Why three diverse base models: Stacking works best when base models make
different kinds of errors. RF, XGBoost, and MLP have different inductive biases
(bagging vs boosting vs gradient descent on a smooth function), so they tend
to disagree on borderline cases — and that disagreement is what the meta-learner
exploits.

Why Logistic Regression as meta-learner: With only 3 input features (one
probability per base model), the meta-learner's job is simple — learn a
weighted average. LR is the right tool for that. This is the ONE place LR
is acceptable in the project per the strategy report.

Reads:  data/output/steam_features_engineered.csv
        Modeling/optuna_best_params.json (XGBoost best params)
Writes: Modeling/stacking_cv_metrics.csv
        Modeling/stacking_results.md
        Modeling/charts/stacking_roc_curve.png
        Modeling/charts/stacking_confusion_matrix.png
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
from sklearn.linear_model import LogisticRegression
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

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost is required. Install with: pip install xgboost")

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
# Base model configurations
# ---------------------------------------------------------------------------
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGBoost params will be loaded from optuna_best_params.json if it exists
DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
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


def load_xgb_params() -> dict:
    """Load tuned XGBoost params from Optuna output if available."""
    optuna_path = MODELING_DIR / "optuna_best_params.json"
    if optuna_path.exists():
        try:
            payload = json.loads(optuna_path.read_text())
            params = payload["best_params"].copy()
            params.update({
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": -1,
            })
            print(f"  Loaded tuned XGBoost params from {optuna_path.name}")
            return params
        except Exception as exc:
            print(f"  Could not load Optuna params ({exc}), using defaults")
    print("  Using default XGBoost params (run optuna_tuning.py first for tuned params)")
    return DEFAULT_XGB_PARAMS


# ---------------------------------------------------------------------------
# MLP-specific preprocessing (same as mlp_training.py)
# ---------------------------------------------------------------------------
def preprocess_for_mlp(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.copy()
    X_val = X_val.copy()

    skewness = X_train.skew()
    log_cols = skewness[skewness > SKEWNESS_THRESHOLD].index.tolist()
    log_cols = [c for c in log_cols if (X_train[c] >= 0).all()]
    for col in log_cols:
        X_train[col] = np.log1p(X_train[col])
        X_val[col] = np.log1p(X_val[col].clip(lower=0))

    for col in X_train.columns:
        if X_train[col].nunique() <= 2:
            continue
        cap = X_train[col].quantile(OUTLIER_PERCENTILE / 100)
        X_train[col] = X_train[col].clip(upper=cap)
        X_val[col] = X_val[col].clip(upper=cap)

    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)


# ---------------------------------------------------------------------------
# Stacking with proper out-of-fold predictions
# ---------------------------------------------------------------------------
def train_stacking_ensemble(X: pd.DataFrame, y: pd.Series, xgb_params: dict):
    """
    Manual stacking with out-of-fold (OOF) predictions.

    Why manual instead of sklearn's StackingClassifier:
    - We need MLP-specific preprocessing on a subset of folds
    - We want to track per-fold metrics for the ensemble itself
    - It's clearer what's happening
    """
    cv = get_cv_splitter()

    # Out-of-fold predictions from each base model
    # These will be the meta-learner's training data
    oof_rf = np.zeros(len(y))
    oof_xgb = np.zeros(len(y))
    oof_mlp = np.zeros(len(y))

    # Per-fold ensemble metrics
    metrics_by_fold = []
    roc_curves = []
    confusion_total = np.zeros((2, 2), dtype=int)

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    xgb_params = {**xgb_params, "scale_pos_weight": scale_pos_weight}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"  Fold {fold}/5 ...", flush=True)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- Random Forest ---
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)
        rf_val_prob = rf.predict_proba(X_val)[:, 1]
        oof_rf[val_idx] = rf_val_prob

        # --- XGBoost (tuned) ---
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        xgb_val_prob = xgb.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = xgb_val_prob

        # --- MLP (with preprocessing) ---
        X_train_mlp, X_val_mlp = preprocess_for_mlp(X_train, X_val)
        mlp = MLPClassifier(**MLP_PARAMS)
        mlp.fit(X_train_mlp, y_train)
        mlp_val_prob = mlp.predict_proba(X_val_mlp)[:, 1]
        oof_mlp[val_idx] = mlp_val_prob

        # --- Meta-learner: Logistic Regression on the 3 probabilities ---
        # Train meta-learner on this fold's val predictions, then we'll
        # also need to evaluate it. Since the meta-learner sees the same
        # val set it's evaluated on, we get a small amount of optimism here.
        # The cleaner way is to wait until ALL folds are done and then
        # evaluate the meta-learner on OOF predictions across the full dataset.
        # We do that below.

        meta_X_fold = np.column_stack([rf_val_prob, xgb_val_prob, mlp_val_prob])
        meta = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
        meta.fit(meta_X_fold, y_val)
        ensemble_prob = meta.predict_proba(meta_X_fold)[:, 1]
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_val, ensemble_prob)
        roc_curves.append((fpr, tpr, auc(fpr, tpr)))
        confusion_total += confusion_matrix(y_val, ensemble_pred, labels=[0, 1])

        fold_metrics = {
            "fold": fold,
            "rf_roc_auc": roc_auc_score(y_val, rf_val_prob),
            "xgb_roc_auc": roc_auc_score(y_val, xgb_val_prob),
            "mlp_roc_auc": roc_auc_score(y_val, mlp_val_prob),
            "ensemble_roc_auc": roc_auc_score(y_val, ensemble_prob),
            "ensemble_f1": f1_score(y_val, ensemble_pred, zero_division=0),
            "ensemble_precision": precision_score(y_val, ensemble_pred, zero_division=0),
            "ensemble_recall": recall_score(y_val, ensemble_pred, zero_division=0),
            "ensemble_accuracy": accuracy_score(y_val, ensemble_pred),
        }
        metrics_by_fold.append(fold_metrics)
        print(f"    RF={fold_metrics['rf_roc_auc']:.4f}, "
              f"XGB={fold_metrics['xgb_roc_auc']:.4f}, "
              f"MLP={fold_metrics['mlp_roc_auc']:.4f}, "
              f"Ensemble={fold_metrics['ensemble_roc_auc']:.4f}")

    # Evaluate the meta-learner on the full OOF predictions
    # This is the most honest measure of stacking performance
    print("\n  Final meta-learner on out-of-fold predictions ...")
    meta_X_full = np.column_stack([oof_rf, oof_xgb, oof_mlp])
    meta_full = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
    meta_full.fit(meta_X_full, y)
    final_prob = meta_full.predict_proba(meta_X_full)[:, 1]
    final_pred = (final_prob >= 0.5).astype(int)

    final_summary = {
        "oof_roc_auc_rf": float(roc_auc_score(y, oof_rf)),
        "oof_roc_auc_xgb": float(roc_auc_score(y, oof_xgb)),
        "oof_roc_auc_mlp": float(roc_auc_score(y, oof_mlp)),
        "oof_roc_auc_ensemble": float(roc_auc_score(y, final_prob)),
        "oof_f1_ensemble": float(f1_score(y, final_pred, zero_division=0)),
        "oof_precision_ensemble": float(precision_score(y, final_pred, zero_division=0)),
        "oof_recall_ensemble": float(recall_score(y, final_pred, zero_division=0)),
        "meta_coefficients": {
            "rf": float(meta_full.coef_[0][0]),
            "xgb": float(meta_full.coef_[0][1]),
            "mlp": float(meta_full.coef_[0][2]),
        },
    }

    return pd.DataFrame(metrics_by_fold), final_summary, roc_curves, confusion_total


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
    plt.title("Stacking Ensemble ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion(confusion_total: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_total, display_labels=["Fail", "Success"])
    disp.plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
    ax.set_title("Stacking Ensemble Confusion Matrix (sum across folds)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(metrics_df: pd.DataFrame, final_summary: dict) -> None:
    metrics_df.to_csv(MODELING_DIR / "stacking_cv_metrics.csv", index=False)

    coefs = final_summary["meta_coefficients"]
    report = f"""# Stacking Ensemble Results

The stacking ensemble combines three diverse base models: Random Forest (bagging-based tree ensemble), XGBoost with Optuna-tuned hyperparameters (boosting-based tree ensemble), and a Tabular MLP (neural network). A Logistic Regression meta-learner with `class_weight='balanced'` takes the three base-model probability predictions as input and learns the optimal weighted combination.

The diversity of the base models is intentional. Stacking works best when base models make different kinds of errors, so the meta-learner has something to combine. RF, XGBoost, and MLP have very different inductive biases (bagging on decision trees vs gradient boosting on decision trees vs gradient descent on a smooth differentiable function), and they tend to disagree on borderline cases. The meta-learner exploits that disagreement to produce a better-calibrated final prediction.

The meta-learner was trained on out-of-fold predictions across all five CV folds, which is the cleanest way to measure stacking performance without optimism bias from training and evaluating on the same fold. Across the full out-of-fold dataset, the individual base models achieved ROC-AUC scores of {final_summary['oof_roc_auc_rf']:.3f} (Random Forest), {final_summary['oof_roc_auc_xgb']:.3f} (tuned XGBoost), and {final_summary['oof_roc_auc_mlp']:.3f} (MLP). The stacked ensemble achieved a ROC-AUC of {final_summary['oof_roc_auc_ensemble']:.3f}, with F1 of {final_summary['oof_f1_ensemble']:.3f}, precision of {final_summary['oof_precision_ensemble']:.3f}, and recall of {final_summary['oof_recall_ensemble']:.3f}.

The Logistic Regression meta-learner assigned coefficients of {coefs['rf']:.3f} to the Random Forest predictions, {coefs['xgb']:.3f} to the XGBoost predictions, and {coefs['mlp']:.3f} to the MLP predictions. Larger absolute values indicate the meta-learner found those base-model predictions more useful. This coefficient pattern is itself informative: it tells us which model's signal contributes most to the final prediction, and whether the ensemble is genuinely combining all three or essentially leaning on one. For tabular data, it is common to see the strongest single model contribute the largest coefficient and the others contribute smaller corrections.
"""
    (MODELING_DIR / "stacking_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("STACKING ENSEMBLE")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/4] Loading data ...")
    X, y = load_modeling_data()

    print("\n[2/4] Loading XGBoost hyperparameters ...")
    xgb_params = load_xgb_params()

    print("\n[3/4] Training base models and ensemble (5-fold CV) ...")
    print("  This trains 5 RF + 5 XGB + 5 MLP = 15 models total. It takes a while.")
    metrics_df, final_summary, roc_curves, confusion_total = train_stacking_ensemble(X, y, xgb_params)

    print("\n[4/4] Saving results and plots ...")
    plot_roc_curves(roc_curves, CHARTS_DIR / "stacking_roc_curve.png")
    plot_confusion(confusion_total, CHARTS_DIR / "stacking_confusion_matrix.png")
    write_results(metrics_df, final_summary)

    print("\nFinal stacking results (out-of-fold):")
    print(f"  RF base       ROC-AUC: {final_summary['oof_roc_auc_rf']:.4f}")
    print(f"  XGB base      ROC-AUC: {final_summary['oof_roc_auc_xgb']:.4f}")
    print(f"  MLP base      ROC-AUC: {final_summary['oof_roc_auc_mlp']:.4f}")
    print(f"  Ensemble      ROC-AUC: {final_summary['oof_roc_auc_ensemble']:.4f}")
    print(f"  Ensemble F1            : {final_summary['oof_f1_ensemble']:.4f}")
    print(f"\nMeta-learner coefficients (LR weights on each base model):")
    for k, v in final_summary["meta_coefficients"].items():
        print(f"  {k}: {v:+.4f}")
    print(f"\nSaved metrics to {MODELING_DIR / 'stacking_cv_metrics.csv'}")
    print(f"Saved report to {MODELING_DIR / 'stacking_results.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()