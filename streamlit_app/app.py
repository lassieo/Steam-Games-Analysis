"""
Steam Game Success Predictor — Streamlit App
=============================================

Main entry point. Three tabs:
  1. Predictor — user inputs, XGBoost prediction, SHAP explanation
  2. Chatbot — natural language Q&A grounded in real data via Groq
  3. Dashboard — interactive EDA and model performance

Run locally:
  streamlit run streamlit_app/app.py

Deploy to Streamlit Community Cloud:
  1. Push to GitHub
  2. Go to share.streamlit.io
  3. Point it at this file
  4. Add GROQ_API_KEY in app settings → Secrets
"""
from __future__ import annotations

import streamlit as st

from app_utils import load_model_metadata
from chatbot import render_chatbot_tab
from dashboard import render_dashboard_tab
from predictor import render_predictor_tab


def main() -> None:
    st.set_page_config(
        page_title="Steam Game Success Predictor",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- Header ---
    st.title("Steam Game Success Predictor")
    st.markdown(
        "A data-driven ML project analyzing 89,618 Steam games to predict commercial "
        "success from pre-launch features. Built for the Applied ML course at "
        "Northeastern University by Laasya Venugopal, Pujitha Attuluri, and Kanniese Chen."
    )

    metadata = load_model_metadata()
    if metadata:
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.caption(
            f"**Model:** {metadata.get('model_type', 'XGBoost')}"
        )
        info_col2.caption(
            f"**Training rows:** {metadata.get('n_training_rows', 0):,}  "
            f"**Features:** {metadata.get('n_features', 0)}"
        )
        roc = metadata.get('optuna_cv_roc_auc')
        if roc:
            info_col3.caption(f"**CV ROC-AUC:** {roc:.4f}")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "Predictor",
        "Ask the Data",
        "Dashboard",
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
        "Source code: [Steam-Games-Analysis on GitHub](https://github.com/lassieo/Steam-Games-Analysis)"
    )


if __name__ == "__main__":
    main()