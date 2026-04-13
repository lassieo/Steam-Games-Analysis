"""
Export Trained Model for Streamlit App
=======================================

Run this ONCE locally to produce the model pickle that the Streamlit app uses.

What it does:
  1. Loads steam_features_engineered.csv (drops eda_ columns, ID, target)
  2. Reads the Optuna best params from Modeling/optuna_best_params.json
  3. Trains the tuned XGBoost on the FULL dataset (no CV split — we want
     the final production model, not a cross-validation average)
  4. Saves the model, feature list, and metadata to streamlit_app/models/

Usage:
  python streamlit_app/export_model.py

Produces:
  streamlit_app/models/xgb_tuned.pkl       (the trained model)
  streamlit_app/models/feature_names.json  (column order for predictions)
  streamlit_app/models/model_metadata.json (training info, metrics)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("XGBoost required. Install with: pip install xgboost")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "output" / "steam_features_engineered.csv"
OPTUNA_PATH = REPO_ROOT / "Modeling" / "optuna_best_params.json"
MODELS_DIR = REPO_ROOT / "streamlit_app" / "models"

RANDOM_STATE = 42


def main() -> None:
    print("=" * 60)
    print("EXPORT TRAINED MODEL FOR STREAMLIT APP")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"\n[1/4] Loading data from {DATA_PATH.name} ...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run the EDA script first to generate it."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    drop_cols = ["success", "appid", "name"]
    eda_cols = [c for c in df.columns if c.startswith("eda_")]
    X = df.drop(columns=drop_cols + eda_cols, errors="ignore")
    y = df["success"].astype(int)

    # Coerce any non-numeric columns
    object_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    print(f"  Loaded {len(X):,} rows with {X.shape[1]} features")
    print(f"  Class balance: {dict(y.value_counts().sort_index())}")

    # --- Load Optuna best params ---
    print(f"\n[2/4] Loading tuned hyperparameters ...")
    if OPTUNA_PATH.exists():
        optuna_payload = json.loads(OPTUNA_PATH.read_text())
        best_params = optuna_payload["best_params"]
        optuna_roc = optuna_payload.get("best_roc_auc")
        print(f"  Using Optuna-tuned params (CV ROC-AUC: {optuna_roc:.4f})")
    else:
        print(f"  {OPTUNA_PATH.name} not found, using defaults")
        best_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
        }
        optuna_roc = None

    # Add production params
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    full_params = {
        **best_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }

    # --- Train on full dataset ---
    print(f"\n[3/4] Training on full dataset ({len(X):,} rows) ...")
    model = XGBClassifier(**full_params)
    model.fit(X, y)
    print(f"  Model trained. Feature count: {model.n_features_in_}")

    # --- Save artifacts ---
    print(f"\n[4/4] Saving model artifacts to {MODELS_DIR.relative_to(REPO_ROOT)}/ ...")

    model_path = MODELS_DIR / "xgb_tuned.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model: {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB)")

    feature_names = X.columns.tolist()
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    print(f"  Saved feature names: feature_names.json ({len(feature_names)} features)")

    # Feature statistics for the predictor UI (mean/median/min/max)
    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
        }
    (MODELS_DIR / "feature_stats.json").write_text(json.dumps(feature_stats, indent=2))
    print(f"  Saved feature stats: feature_stats.json")

    metadata = {
        "model_type": "XGBClassifier (tuned)",
        "n_training_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "optuna_cv_roc_auc": optuna_roc,
        "class_balance": {
            "not_successful": int((y == 0).sum()),
            "successful": int((y == 1).sum()),
            "positive_rate": float((y == 1).mean()),
        },
        "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                           for k, v in best_params.items()},
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  Saved metadata: model_metadata.json")

    print("\nDone. Commit streamlit_app/models/ to git so Streamlit Cloud can load it.")


if __name__ == "__main__":
    main()