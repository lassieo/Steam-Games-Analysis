"""
Shared utilities for the modeling pipeline.

Every modeling script (rf_baseline, xgb_lgbm, mlp, optuna, stacking, shap)
imports from here so they all use:
  - The same data file (data/output/steam_features_engineered.csv)
  - The same X/y construction (drop ID, target, eda_ columns)
  - The same cross-validation splits (StratifiedKFold, 5 folds, random_state=42)

This guarantees model comparisons are valid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Constants — every script uses these exact values
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "output" / "steam_features_engineered.csv"
MODELING_DIR = REPO_ROOT / "Modeling"
CHARTS_DIR = MODELING_DIR / "charts"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_modeling_data(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered dataset and return (X, y) ready for modeling.

    Drops:
      - 'success' (target, becomes y)
      - 'appid', 'name' (identifiers)
      - any column starting with 'eda_' (post-launch leakage)
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found at {DATA_PATH}.\n"
            f"Run 'python EDA/steam_eda_feature_engineering.py' first to generate it."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "success" not in df.columns:
        raise ValueError("Expected a 'success' column in the engineered dataset.")

    drop_cols = ["success", "appid", "name"]
    eda_cols = [col for col in df.columns if col.startswith("eda_")]
    X = df.drop(columns=drop_cols + eda_cols, errors="ignore").copy()
    y = df["success"].astype(int).copy()

    # Coerce any non-numeric columns to numeric, fill nulls with 0
    object_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    if verbose:
        print(f"  Loaded: {DATA_PATH}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Class balance: {dict(y.value_counts().sort_index())}")
        print(f"  Excluded {len(eda_cols)} eda_ columns from training")

    return X, y


def get_cv_splitter() -> StratifiedKFold:
    """
    Return the standard cross-validation splitter used by every model.

    Using the same splitter (with the same random_state) means every model
    sees identical train/validation folds, so metric comparisons are valid.
    """
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def ensure_dirs() -> None:
    """Create Modeling/ and Modeling/charts/ if they don't exist."""
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def summarize_cv_results(metrics_by_fold: list[dict]) -> dict:
    """
    Convert per-fold metrics into mean +/- std summary.
    Input: list of dicts like [{'roc_auc': 0.87, 'f1': 0.45, ...}, ...]
    Output: dict like {'roc_auc_mean': 0.87, 'roc_auc_std': 0.003, ...}
    """
    df = pd.DataFrame(metrics_by_fold)
    summary = {}
    for col in df.columns:
        if col == "fold":
            continue
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1))
    return summary


def format_metrics(summary: dict, metrics: list[str] = None) -> str:
    """Format metric summary as readable text for printing."""
    if metrics is None:
        metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    lines = []
    for m in metrics:
        mean_key = f"{m}_mean"
        std_key = f"{m}_std"
        if mean_key in summary:
            lines.append(f"  {m:12s}: {summary[mean_key]:.4f} +/- {summary[std_key]:.4f}")
    return "\n".join(lines)