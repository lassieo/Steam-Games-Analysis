"""
Predictor Tab
==============

User enters game attributes via form inputs, the trained XGBoost model
predicts the success probability, and SHAP values explain which features
drove the prediction.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import load_data, load_feature_names, load_feature_stats, load_model


# Key features the user actually interacts with. Everything else gets
# a sensible default from feature_stats.
USER_INPUT_FEATURES = [
    "price",
    "required_age",
    "achievements",
    "dlc_count",
    "windows",
    "mac",
    "linux",
    "supported_language_count",
    "full_audio_language_count",
    "genre_count",
    "category_count",
    "tag_count",
    "release_year",
    "description_length",
    "developer_historical_success",
    "publisher_historical_success",
    "developer_game_count",
]

GENRE_FEATURES_PREFIX = "genre_"
CATEGORY_FEATURES_PREFIX = "cat_"
TAG_FEATURES_PREFIX = "tag_"

POPULAR_GENRES = ["action", "adventure", "casual", "indie", "rpg", "simulation", "strategy", "sports", "racing"]
POPULAR_CATEGORIES = ["single_player", "multi_player", "steam_achievements", "steam_cloud", "steam_trading_cards"]


def build_feature_vector(user_inputs: dict, feature_names: list, feature_stats: dict) -> pd.DataFrame:
    """
    Convert the user's form inputs into a full feature row matching
    the model's expected columns. Uses median values from training data
    for any feature the user didn't set.
    """
    row = {}
    for feat in feature_names:
        if feat in user_inputs:
            row[feat] = user_inputs[feat]
        else:
            # Default to median for continuous, 0 for binary
            stats = feature_stats.get(feat, {"median": 0})
            row[feat] = stats.get("median", 0)

    return pd.DataFrame([row], columns=feature_names)


def plot_shap_waterfall(model, X_row: pd.DataFrame) -> plt.Figure:
    """
    Generate a SHAP waterfall plot showing how each feature contributed
    to this specific prediction.
    """
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Build a sorted list of (feature, value, shap_value) triples
    triples = [
        (col, float(X_row.iloc[0][col]), float(shap_values[0][i]))
        for i, col in enumerate(X_row.columns)
    ]
    triples.sort(key=lambda t: abs(t[2]), reverse=True)
    top = triples[:12]

    # Horizontal bar chart: SHAP value per feature
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{name} = {val:.2f}" for name, val, _ in reversed(top)]
    values = [sv for _, _, sv in reversed(top)]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Why this prediction? (top 12 features)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return fig


def render_predictor_tab() -> None:
    st.header("Game Success Predictor")
    st.markdown(
        "Enter your game's attributes and the tuned XGBoost model will predict "
        "the probability of commercial success on Steam."
    )

    model = load_model()
    feature_names = load_feature_names()
    feature_stats = load_feature_stats()

    # --- Layout: two columns for inputs ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic attributes")
        price = st.slider("Price (USD)", 0.0, 60.0, 14.99, step=0.5,
                          help="0 means the game is free-to-play.")
        release_year = st.slider("Release year", 2010, 2026, 2024)
        required_age = st.number_input("Required age rating", 0, 21, 0)
        achievements = st.number_input("Number of achievements", 0, 500, 20)
        dlc_count = st.number_input("Number of DLCs planned", 0, 20, 0)
        description_length = st.slider("Description length (chars)", 0, 5000, 1500)

        st.subheader("Platforms")
        windows = st.checkbox("Windows support", value=True)
        mac = st.checkbox("Mac support", value=False)
        linux = st.checkbox("Linux support", value=False)

        st.subheader("Localization")
        supported_language_count = st.number_input(
            "Number of supported languages", 1, 30, 5
        )
        full_audio_language_count = st.number_input(
            "Number of full audio languages", 0, 20, 1
        )

    with col2:
        st.subheader("Developer track record")
        st.caption(
            "These features were the strongest predictors in our SHAP analysis. "
            "Leave them at 0 if you're a new developer."
        )
        developer_game_count = st.number_input(
            "Previous games by this developer", 0, 100, 0,
            help="Total number of games this developer has released before."
        )
        developer_historical_success = st.slider(
            "Developer historical success rate", 0.0, 1.0, 0.15, step=0.01,
            help="Fraction of the developer's previous games that were successful. "
                 "0.15 is the overall average."
        )
        publisher_historical_success = st.slider(
            "Publisher historical success rate", 0.0, 1.0, 0.15, step=0.01,
            help="Fraction of the publisher's previous games that were successful."
        )

        st.subheader("Content breadth")
        genre_count = st.number_input("Number of genres", 1, 10, 2)
        category_count = st.number_input("Number of categories", 1, 15, 5)
        tag_count = st.number_input(
            "Number of community tags", 0, 50, 10,
            help="Number of tags you expect the Steam community to assign. "
                 "This is the strongest feature in the model."
        )

        st.subheader("Genre selection")
        selected_genres = st.multiselect(
            "Select applicable genres",
            options=POPULAR_GENRES,
            default=["indie", "action"]
        )

    # --- Build the feature vector ---
    user_inputs = {
        "price": price,
        "required_age": required_age,
        "achievements": achievements,
        "dlc_count": dlc_count,
        "windows": int(windows),
        "mac": int(mac),
        "linux": int(linux),
        "supported_language_count": supported_language_count,
        "full_audio_language_count": full_audio_language_count,
        "genre_count": genre_count,
        "category_count": category_count,
        "tag_count": tag_count,
        "release_year": release_year,
        "description_length": description_length,
        "developer_historical_success": developer_historical_success,
        "publisher_historical_success": publisher_historical_success,
        "developer_game_count": developer_game_count,
    }

    # Engineered helpers
    user_inputs["is_free"] = 1 if price == 0 else 0
    user_inputs["has_achievements"] = 1 if achievements > 0 else 0
    user_inputs["has_tags"] = 1 if tag_count > 0 else 0
    user_inputs["has_dlc"] = 1 if dlc_count > 0 else 0
    user_inputs["log_price"] = float(np.log1p(price))
    user_inputs["platform_count"] = int(windows) + int(mac) + int(linux)

    # Genre one-hot encoding
    for g in POPULAR_GENRES:
        col = f"{GENRE_FEATURES_PREFIX}{g}"
        if col in feature_names:
            user_inputs[col] = 1 if g in selected_genres else 0

    # --- Predict ---
    st.markdown("---")
    if st.button("Predict success probability", type="primary", use_container_width=True):
        X_row = build_feature_vector(user_inputs, feature_names, feature_stats)

        prob = float(model.predict_proba(X_row)[0][1])
        pred = int(prob >= 0.5)

        # Big metric display
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                "Success probability",
                f"{prob * 100:.1f}%",
                help="Model's predicted probability this game will be successful"
            )
        with metric_col2:
            verdict = "Likely Successful" if pred == 1 else "Likely Unsuccessful"
            st.metric("Prediction", verdict)
        with metric_col3:
            confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
            st.metric("Confidence", confidence)

        # Context
        if prob >= 0.7:
            st.success(
                f"The model is confident this game has strong commercial potential. "
                f"The features you selected align with successful games in the training data."
            )
        elif prob >= 0.5:
            st.info(
                f"The model leans toward this game being successful, but the signal is not strong. "
                f"Consider adjusting the inputs to see what would push the prediction higher."
            )
        elif prob >= 0.3:
            st.warning(
                f"The model leans toward this game being unsuccessful. "
                f"Review which features are dragging the prediction down below."
            )
        else:
            st.error(
                f"The model is confident this game would not succeed under the current configuration. "
                f"The most impactful factors are typically developer track record, tag engagement, and pricing."
            )

        # --- SHAP explanation ---
        st.markdown("---")
        st.subheader("Why did the model predict this?")
        st.caption(
            "SHAP values show which features pushed the prediction up (green) or down (red). "
            "The magnitude indicates how much each feature mattered."
        )

        with st.spinner("Computing SHAP values..."):
            fig = plot_shap_waterfall(model, X_row)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("SHAP not installed. Install with: pip install shap")

    else:
        st.info("Adjust the inputs above and click the button to get a prediction.")