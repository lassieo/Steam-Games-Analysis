"""
Shared utilities for the Streamlit app.

Handles data loading, model loading, and path resolution. Uses @st.cache_resource
and @st.cache_data so expensive loads happen once per session.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent
DATA_PATH = REPO_ROOT / "data" / "output" / "steam_features_engineered.csv"
MODELS_DIR = APP_ROOT / "models"

MODEL_PATH = MODELS_DIR / "xgb_tuned.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
FEATURE_STATS_PATH = MODELS_DIR / "feature_stats.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading game data...")
def load_data() -> pd.DataFrame:
    """Load the engineered dataset. Cached across the session."""
    if not DATA_PATH.exists():
        st.error(
            f"Data file not found at {DATA_PATH.relative_to(REPO_ROOT)}. "
            "Make sure the engineered CSV is committed to the repo."
        )
        st.stop()
    return pd.read_csv(DATA_PATH, low_memory=False)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    """Load the trained XGBoost model pickle. Cached across the session."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH.relative_to(REPO_ROOT)}. "
            "Run `python streamlit_app/export_model.py` first."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_feature_names() -> list:
    if not FEATURE_NAMES_PATH.exists():
        st.error(f"feature_names.json not found. Run export_model.py first.")
        st.stop()
    return json.loads(FEATURE_NAMES_PATH.read_text())


@st.cache_data
def load_feature_stats() -> dict:
    if not FEATURE_STATS_PATH.exists():
        st.error(f"feature_stats.json not found. Run export_model.py first.")
        st.stop()
    return json.loads(FEATURE_STATS_PATH.read_text())


@st.cache_data
def load_model_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


@st.cache_data
def get_modeling_features(df: pd.DataFrame) -> list:
    """Return the list of columns that should be used as model features."""
    drop_cols = ["success", "appid", "name"]
    eda_cols = [c for c in df.columns if c.startswith("eda_")]
    return [c for c in df.columns if c not in drop_cols + eda_cols]