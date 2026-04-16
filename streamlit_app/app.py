"""
Steam Game Success Predictor — Streamlit App
=============================================

Main entry point. Three tabs:
  1. Predictor — user inputs, XGBoost prediction, SHAP explanation
  2. Chatbot — natural language Q&A grounded in real data via Groq
  3. Dashboard — interactive EDA and model performance

Run locally:
  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from app_utils import load_model_metadata
from chatbot import render_chatbot_tab
from dashboard import render_dashboard_tab
from predictor import render_predictor_tab


def inject_custom_css() -> None:
    """Inject CSS for consistent branding across the app."""
    st.markdown(
        """
        <style>
        /* Hero section styling */
        .hero-container {
            background: linear-gradient(135deg, #1b2838 0%, #2a475e 100%);
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
            color: #ffffff;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #c7d5e0;
            margin-bottom: 1rem;
        }
        .hero-team {
            font-size: 0.85rem;
            color: #8f98a0;
        }
        .hero-stats {
            display: flex;
            gap: 2rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.15);
        }
        .hero-stat {
            display: flex;
            flex-direction: column;
        }
        .hero-stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #66c0f4;
        }
        .hero-stat-label {
            font-size: 0.75rem;
            color: #8f98a0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-weight: 500;
        }

        /* Section headers */
        h2, h3 {
            color: #1b2838;
        }

        /* Button styling */
        .stButton>button[kind="primary"] {
            background-color: #66c0f4;
            color: #1b2838;
            font-weight: 600;
            border: none;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #5aa0d0;
            color: #1b2838;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metadata: dict) -> None:
    """Render the hero section at the top of the app."""
    training_rows = metadata.get("n_training_rows", 89618)
    n_features = metadata.get("n_features", 72)
    roc = metadata.get("optuna_cv_roc_auc", 0.8841)

    hero_html = f"""
    <div class="hero-container">
        <div class="hero-title">🎮 Steam Game Success Predictor</div>
        <div class="hero-subtitle">
            A data-driven ML analysis of {training_rows:,} Steam games, built to predict
            commercial success from pre-launch features only.
        </div>
        <div class="hero-team">
            Applied ML Project &middot; Laasya Venugopal &middot; Pujitha Attuluri &middot; Kanniese Chen
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-stat-value">{training_rows:,}</div>
                <div class="hero-stat-label">Games analyzed</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{n_features}</div>
                <div class="hero-stat-label">Features engineered</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">{roc:.4f}</div>
                <div class="hero-stat-label">CV ROC-AUC</div>
            </div>
            <div class="hero-stat">
                <div class="hero-stat-value">6</div>
                <div class="hero-stat-label">Models compared</div>
            </div>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Steam Game Success Predictor",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    metadata = load_model_metadata()
    render_hero(metadata)

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "🎯 Predictor",
        "💬 Ask the Data",
        "📊 Dashboard",
    ])

    with tab1:
        render_predictor_tab()

    with tab2:
        render_chatbot_tab()

    with tab3:
        render_dashboard_tab()

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Based on the [Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset). "
        "Prediction model: tuned XGBoost. Best overall: stacking ensemble (ROC-AUC 0.8844). "
        "Source code: [Steam-Games-Analysis on GitHub](https://github.com/lassieo/Steam-Games-Analysis)"
    )


if __name__ == "__main__":
    main()