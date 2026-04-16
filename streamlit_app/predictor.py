"""
Predictor Tab
==============

User enters game attributes via form inputs (or clicks a preset), the trained
XGBoost model predicts the success probability, and SHAP values explain which
features drove the prediction.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import load_data, load_feature_names, load_feature_stats, load_model


POPULAR_GENRES = ["action", "adventure", "casual", "indie", "rpg", "simulation", "strategy", "sports", "racing"]
GENRE_FEATURES_PREFIX = "genre_"


# ---------------------------------------------------------------------------
# Presets — realistic game scenarios the user can click to load instantly
# ---------------------------------------------------------------------------
PRESETS = {
    "🏆 AAA blockbuster": {
        "description": "Big-budget title from an experienced studio with proven track record",
        "inputs": {
            "price": 59.99,
            "release_year": 2024,
            "required_age": 17,
            "achievements": 80,
            "dlc_count": 3,
            "description_length": 4500,
            "windows": True, "mac": True, "linux": False,
            "supported_language_count": 18,
            "full_audio_language_count": 8,
            "genre_count": 2,
            "category_count": 10,
            "tag_count": 35,
            "developer_game_count": 15,
            "developer_historical_success": 0.75,
            "publisher_historical_success": 0.80,
            "selected_genres": ["action", "adventure"],
        },
    },
    "🎨 Indie gem": {
        "description": "Polished indie game from a small but experienced developer",
        "inputs": {
            "price": 19.99,
            "release_year": 2024,
            "required_age": 0,
            "achievements": 35,
            "dlc_count": 0,
            "description_length": 2500,
            "windows": True, "mac": True, "linux": True,
            "supported_language_count": 12,
            "full_audio_language_count": 1,
            "genre_count": 3,
            "category_count": 7,
            "tag_count": 28,
            "developer_game_count": 4,
            "developer_historical_success": 0.50,
            "publisher_historical_success": 0.50,
            "selected_genres": ["indie", "adventure", "rpg"],
        },
    },
    "🆓 Free-to-play": {
        "description": "Free game with monetization through DLC or microtransactions",
        "inputs": {
            "price": 0.0,
            "release_year": 2024,
            "required_age": 13,
            "achievements": 50,
            "dlc_count": 5,
            "description_length": 3000,
            "windows": True, "mac": False, "linux": False,
            "supported_language_count": 15,
            "full_audio_language_count": 3,
            "genre_count": 2,
            "category_count": 8,
            "tag_count": 25,
            "developer_game_count": 6,
            "developer_historical_success": 0.40,
            "publisher_historical_success": 0.45,
            "selected_genres": ["action", "casual"],
        },
    },
    "🌱 First-time developer": {
        "description": "Typical first release from a new indie developer with no track record",
        "inputs": {
            "price": 9.99,
            "release_year": 2024,
            "required_age": 0,
            "achievements": 15,
            "dlc_count": 0,
            "description_length": 1200,
            "windows": True, "mac": False, "linux": False,
            "supported_language_count": 2,
            "full_audio_language_count": 0,
            "genre_count": 2,
            "category_count": 4,
            "tag_count": 8,
            "developer_game_count": 0,
            "developer_historical_success": 0.0,
            "publisher_historical_success": 0.0,
            "selected_genres": ["indie", "casual"],
        },
    },
}


def apply_preset(preset_name: str) -> None:
    """Load a preset's values into session state so the widgets pick them up."""
    preset = PRESETS[preset_name]
    for key, value in preset["inputs"].items():
        st.session_state[f"pred_{key}"] = value
    st.session_state["preset_applied"] = preset_name


def build_feature_vector(user_inputs: dict, feature_names: list, feature_stats: dict) -> pd.DataFrame:
    row = {}
    for feat in feature_names:
        if feat in user_inputs:
            row[feat] = user_inputs[feat]
        else:
            stats = feature_stats.get(feat, {"median": 0})
            row[feat] = stats.get("median", 0)
    return pd.DataFrame([row], columns=feature_names)


def plot_shap_waterfall(model, X_row: pd.DataFrame) -> plt.Figure:
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    triples = [
        (col, float(X_row.iloc[0][col]), float(shap_values[0][i]))
        for i, col in enumerate(X_row.columns)
    ]
    triples.sort(key=lambda t: abs(t[2]), reverse=True)
    top = triples[:12]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{name} = {val:.2f}" for name, val, _ in reversed(top)]
    values = [sv for _, _, sv in reversed(top)]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title("Top 12 features driving this prediction", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    return fig


def render_predictor_tab() -> None:
    st.header("Game Success Predictor")
    st.markdown(
        "Enter your game's attributes and the tuned XGBoost model will predict "
        "the probability of commercial success on Steam. "
        "**Start with a preset below** to see the model in action instantly."
    )

    model = load_model()
    feature_names = load_feature_names()
    feature_stats = load_feature_stats()

    # --- Preset buttons ---
    st.subheader("Quick start: load a preset scenario")
    preset_cols = st.columns(4)
    preset_names = list(PRESETS.keys())
    for i, name in enumerate(preset_names):
        with preset_cols[i]:
            if st.button(name, use_container_width=True, key=f"preset_btn_{i}"):
                apply_preset(name)
                st.rerun()
            st.caption(PRESETS[name]["description"])

    if "preset_applied" in st.session_state:
        st.info(
            f"Loaded preset: **{st.session_state['preset_applied']}**. "
            "Adjust any values below and click Predict."
        )

    st.markdown("---")

    # --- Input form ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Basic attributes")
        price = st.slider(
            "Price (USD)",
            0.0, 100.0, st.session_state.get("pred_price", 14.99), step=0.5,
            help="0 = Free | <$15 = Budget | $15-50 = Mid | $50-85 = Premium | $85+ = AAA",
            key="pred_price_widget",
        )
        release_year = st.slider(
            "Release year",
            2010, 2026, st.session_state.get("pred_release_year", 2024),
            key="pred_release_year_widget",
        )
        required_age = st.number_input(
            "Required age rating",
            0, 21, st.session_state.get("pred_required_age", 0),
            key="pred_required_age_widget",
        )
        achievements = st.number_input(
            "Number of achievements",
            0, 500, st.session_state.get("pred_achievements", 20),
            key="pred_achievements_widget",
        )
        dlc_count = st.number_input(
            "Number of DLCs planned",
            0, 20, st.session_state.get("pred_dlc_count", 0),
            key="pred_dlc_count_widget",
        )
        description_length = st.slider(
            "Description length (chars)",
            0, 5000, st.session_state.get("pred_description_length", 1500),
            key="pred_description_length_widget",
        )

        st.markdown("### Platforms")
        windows = st.checkbox("Windows support", value=st.session_state.get("pred_windows", True), key="pred_windows_widget")
        mac = st.checkbox("Mac support", value=st.session_state.get("pred_mac", False), key="pred_mac_widget")
        linux = st.checkbox("Linux support", value=st.session_state.get("pred_linux", False), key="pred_linux_widget")

        st.markdown("### Localization")
        supported_language_count = st.number_input(
            "Number of supported languages",
            1, 30, st.session_state.get("pred_supported_language_count", 5),
            key="pred_supported_language_count_widget",
        )
        full_audio_language_count = st.number_input(
            "Number of full audio languages",
            0, 20, st.session_state.get("pred_full_audio_language_count", 1),
            key="pred_full_audio_language_count_widget",
        )

    with col2:
        st.markdown("### Developer track record")
        st.caption(
            "These were the strongest predictors in our SHAP analysis. "
            "Leave at 0 if you're a first-time developer."
        )
        developer_game_count = st.number_input(
            "Previous games by this developer",
            0, 100, st.session_state.get("pred_developer_game_count", 0),
            help="Total games released by this developer before this one.",
            key="pred_developer_game_count_widget",
        )
        developer_historical_success = st.slider(
            "Developer historical success rate",
            0.0, 1.0, st.session_state.get("pred_developer_historical_success", 0.15), step=0.01,
            help="Fraction of the developer's previous games that were successful. 0.15 is the overall average.",
            key="pred_developer_historical_success_widget",
        )
        publisher_historical_success = st.slider(
            "Publisher historical success rate",
            0.0, 1.0, st.session_state.get("pred_publisher_historical_success", 0.15), step=0.01,
            help="Fraction of the publisher's previous games that were successful.",
            key="pred_publisher_historical_success_widget",
        )

        st.markdown("### Content breadth")
        genre_count = st.number_input(
            "Number of genres",
            1, 10, st.session_state.get("pred_genre_count", 2),
            key="pred_genre_count_widget",
        )
        category_count = st.number_input(
            "Number of categories",
            1, 15, st.session_state.get("pred_category_count", 5),
            key="pred_category_count_widget",
        )
        tag_count = st.number_input(
            "Number of community tags",
            0, 50, st.session_state.get("pred_tag_count", 10),
            help="Community tags are the #1 predictor in SHAP. More tags = more community engagement.",
            key="pred_tag_count_widget",
        )

        st.markdown("### Genre selection")
        selected_genres = st.multiselect(
            "Select applicable genres",
            options=POPULAR_GENRES,
            default=st.session_state.get("pred_selected_genres", ["indie", "action"]),
            key="pred_selected_genres_widget",
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
    user_inputs["is_free"] = 1 if price == 0 else 0
    user_inputs["has_achievements"] = 1 if achievements > 0 else 0
    user_inputs["has_tags"] = 1 if tag_count > 0 else 0
    user_inputs["has_dlc"] = 1 if dlc_count > 0 else 0
    user_inputs["log_price"] = float(np.log1p(price))
    user_inputs["platform_count"] = int(windows) + int(mac) + int(linux)
    for g in POPULAR_GENRES:
        col = f"{GENRE_FEATURES_PREFIX}{g}"
        if col in feature_names:
            user_inputs[col] = 1 if g in selected_genres else 0

    # --- Predict button ---
    st.markdown("---")
    if st.button("Predict success probability", type="primary", use_container_width=True):
        X_row = build_feature_vector(user_inputs, feature_names, feature_stats)

        prob = float(model.predict_proba(X_row)[0][1])
        # Use the class base rate (16.3%) as the decision threshold instead of the
        # default 0.5, because our dataset is imbalanced (only 16.3% positive class).
        # A prediction above the base rate means "above average chance of success."
        BASE_RATE = 0.163
        pred = int(prob >= BASE_RATE)

        # Big visual result
        st.markdown("### Prediction result")
        result_col1, result_col2, result_col3 = st.columns([2, 1, 1])

        with result_col1:
            # Color based on how far above or below the base rate
            if prob >= 0.50:
                bar_color = "#2ca02c"
                emoji = "🎉"
                label = "Strong success signal"
            elif prob >= 0.30:
                bar_color = "#66c0f4"
                emoji = "👍"
                label = "Above average"
            elif prob >= BASE_RATE:
                bar_color = "#ffa500"
                emoji = "⚠️"
                label = "Slightly above average"
            else:
                bar_color = "#d62728"
                emoji = "📉"
                label = "Below base rate"

            st.markdown(
                f"""
                <div style="padding: 1rem; background: #f0f2f6; border-radius: 8px;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: {bar_color};">
                        {emoji} {prob * 100:.1f}%
                    </div>
                    <div style="color: #666; font-size: 0.9rem;">success probability &middot; {label}</div>
                    <div style="margin-top: 0.8rem; background: #e0e0e0; border-radius: 4px; height: 10px; overflow: hidden; position: relative;">
                        <div style="width: {prob * 100}%; background: {bar_color}; height: 100%;"></div>
                        <div style="position: absolute; left: {BASE_RATE * 100}%; top: -2px; width: 2px; height: 14px; background: #333;"></div>
                    </div>
                    <div style="font-size: 0.75rem; color: #888; margin-top: 0.3rem;">
                        Black line = 16.3% base rate (average for Steam games)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with result_col2:
            verdict = "Above base rate" if pred == 1 else "Below base rate"
            st.metric("vs. base rate", verdict)

        with result_col3:
            distance = abs(prob - BASE_RATE)
            if distance > 0.25:
                confidence = "High"
            elif distance > 0.10:
                confidence = "Medium"
            else:
                confidence = "Low"
            st.metric("Confidence", confidence)

        # Context message
        
        lift = prob / BASE_RATE
        if prob >= 0.50:
            st.success(
                f"✨ Strong success signal. The model predicts this game has a {prob * 100:.1f}% chance of success, "
                f"which is **{lift:.1f}x higher than the 16.3% base rate** for Steam games. "
                f"The combination of features you selected strongly aligns with successful games in the training data."
            )
        elif prob >= 0.30:
            st.info(
                f"Above average. At {prob * 100:.1f}%, this prediction is **{lift:.1f}x the base rate** of 16.3%. "
                f"The model sees meaningful signals of success but not strong enough for a confident prediction. "
                f"To push the prediction higher, try increasing developer track record or community tag count."
            )
        elif prob >= BASE_RATE:
            st.warning(
                f"Slightly above average. At {prob * 100:.1f}%, this is **{lift:.1f}x the base rate** of 16.3%. "
                f"The model sees some positive signals but they're balanced against negative ones. "
                f"Review the SHAP chart below — green bars are helping, red bars are hurting."
            )
        else:
            st.error(
                f"Below base rate. At {prob * 100:.1f}%, this game is predicted to do worse than an average "
                f"Steam release (16.3% base rate). "
                f"The most impactful factors are typically developer track record, community engagement (tags), and pricing. "
                f"Review the SHAP chart below to see which features pulled the prediction down the most."
            )

        # SHAP explanation
        st.markdown("---")
        st.markdown("### Why did the model predict this?")
        st.caption(
            "SHAP values show which features pushed the prediction up (green) or down (red). "
            "The magnitude indicates how much each feature mattered for this specific game."
        )

        with st.spinner("Computing SHAP values..."):
            fig = plot_shap_waterfall(model, X_row)
            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("SHAP not installed. Install with: pip install shap")

    else:
        st.info("👆 Click a preset above or adjust the inputs, then click Predict.")