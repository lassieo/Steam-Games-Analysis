"""
Dashboard Tab
==============

Interactive EDA explorer with filters, success rate charts, and model
performance visualizations.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import load_data, load_model_metadata


def render_dashboard_tab() -> None:
    st.header("Dataset Dashboard")
    st.markdown(
        "Explore the 89,618 Steam games in our dataset. Use the filters in the sidebar "
        "to slice by genre, year, and price tier."
    )

    df = load_data()
    metadata = load_model_metadata()

    # --- Sidebar filters ---
    st.sidebar.markdown("### Dashboard filters")

    year_range = st.sidebar.slider(
        "Release year range",
        int(df["release_year"].min()),
        int(df["release_year"].max()),
        (2018, int(df["release_year"].max())),
    )

    price_filter = st.sidebar.selectbox(
        "Price filter",
        ["All games", "Free only", "Paid only", "Under $10", "$10-30", "$30+"],
    )

    genre_cols = sorted([c for c in df.columns if c.startswith("genre_") and c != "genre_count"])
    genre_labels = [c.replace("genre_", "").title() for c in genre_cols]
    selected_genres = st.sidebar.multiselect(
        "Filter by genre (any match)",
        options=genre_labels,
        default=[],
    )

    # --- Apply filters ---
    filtered = df[
        (df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])
    ]
    if price_filter == "Free only":
        filtered = filtered[filtered["price"] == 0]
    elif price_filter == "Paid only":
        filtered = filtered[filtered["price"] > 0]
    elif price_filter == "Under $10":
        filtered = filtered[(filtered["price"] > 0) & (filtered["price"] < 10)]
    elif price_filter == "$10-30":
        filtered = filtered[(filtered["price"] >= 10) & (filtered["price"] < 30)]
    elif price_filter == "$30+":
        filtered = filtered[filtered["price"] >= 30]

    if selected_genres:
        mask = pd.Series(False, index=filtered.index)
        for g in selected_genres:
            col = f"genre_{g.lower()}"
            if col in filtered.columns:
                mask = mask | (filtered[col] == 1)
        filtered = filtered[mask]

    # --- Top metrics ---
    total_games = len(filtered)
    successful = int(filtered["success"].sum())
    success_rate = successful / total_games if total_games > 0 else 0
    avg_price = filtered[filtered["price"] > 0]["price"].mean() if (filtered["price"] > 0).any() else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games in view", f"{total_games:,}")
    m2.metric("Successful", f"{successful:,}")
    m3.metric("Success rate", f"{success_rate * 100:.1f}%")
    m4.metric("Avg paid price", f"${avg_price:.2f}")

    if total_games == 0:
        st.warning("No games match the current filters. Widen the filters to see data.")
        return

    st.markdown("---")

    # --- Chart 1: Success rate by genre ---
    st.subheader("Success rate by genre")
    genre_stats = []
    for col, label in zip(genre_cols, genre_labels):
        mask = filtered[col] == 1
        if mask.sum() >= 100:
            genre_stats.append({
                "genre": label,
                "count": int(mask.sum()),
                "success_rate": float(filtered.loc[mask, "success"].mean()),
            })

    if genre_stats:
        genre_df = pd.DataFrame(genre_stats).sort_values("success_rate", ascending=True).tail(12)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ca02c" if sr >= success_rate else "#d62728" for sr in genre_df["success_rate"]]
        ax.barh(genre_df["genre"], genre_df["success_rate"] * 100, color=colors)
        ax.axvline(success_rate * 100, color="gray", linestyle="--", linewidth=1,
                   label=f"Overall ({success_rate * 100:.1f}%)")
        ax.set_xlabel("Success rate (%)")
        ax.set_title("Success rate by genre (within filtered view)")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough games per genre to show the chart (need at least 100).")

    # --- Chart 2: Success rate by release year ---
    st.subheader("Success rate trend over time")
    year_stats = filtered.groupby("release_year").agg(
        count=("success", "size"),
        success_rate=("success", "mean"),
    ).reset_index()
    year_stats = year_stats[year_stats["count"] >= 50]

    if len(year_stats) > 1:
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        color1 = "#1f77b4"
        ax1.bar(year_stats["release_year"], year_stats["count"], color=color1, alpha=0.3, label="Game count")
        ax1.set_xlabel("Release year")
        ax1.set_ylabel("Game count", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#ff7f0e"
        ax2.plot(year_stats["release_year"], year_stats["success_rate"] * 100,
                 color=color2, marker="o", linewidth=2, label="Success rate")
        ax2.set_ylabel("Success rate (%)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("Game volume and success rate by release year")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Chart 3: Price distribution with success overlay ---
    st.subheader("Price distribution")
    paid = filtered[(filtered["price"] > 0) & (filtered["price"] < 80)]
    if len(paid) > 100:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        paid_success = paid[paid["success"] == 1]["price"]
        paid_fail = paid[paid["success"] == 0]["price"]

        bins = np.linspace(0, 80, 40)
        ax.hist([paid_fail, paid_success], bins=bins,
                label=["Not successful", "Successful"],
                color=["#d62728", "#2ca02c"], alpha=0.6, stacked=False)
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Game count")
        ax.set_title("Price distribution: successful vs unsuccessful games")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Model performance section ---
    st.markdown("---")
    st.subheader("Model performance")

    model_results = pd.DataFrame([
        {"model": "Random Forest (baseline)", "roc_auc": 0.8792, "f1": 0.434},
        {"model": "XGBoost (untuned)", "roc_auc": 0.8748, "f1": 0.564},
        {"model": "LightGBM", "roc_auc": 0.8799, "f1": 0.556},
        {"model": "Tabular MLP", "roc_auc": 0.8724, "f1": 0.488},
        {"model": "XGBoost (tuned)", "roc_auc": 0.8841, "f1": 0.564},
        {"model": "Stacking Ensemble", "roc_auc": 0.8844, "f1": 0.568},
    ]).sort_values("roc_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_pos = np.arange(len(model_results))
    colors = ["#2ca02c" if r == model_results["roc_auc"].max() else "#1f77b4"
              for r in model_results["roc_auc"]]
    ax.barh(y_pos, model_results["roc_auc"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_results["model"])
    ax.set_xlabel("ROC-AUC (5-fold stratified CV)")
    ax.set_title("Model comparison (best model highlighted)")
    ax.set_xlim(0.85, 0.90)
    for i, (roc, f1) in enumerate(zip(model_results["roc_auc"], model_results["f1"])):
        ax.text(roc + 0.0005, i, f"{roc:.4f} (F1: {f1:.3f})",
                va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- Top SHAP features ---
    st.subheader("What the model learned")
    st.markdown("Top 10 features by SHAP importance from the tuned XGBoost model:")

    shap_top = pd.DataFrame([
        ("tag_count", 1.188),
        ("has_tags", 0.391),
        ("publisher_historical_success", 0.354),
        ("release_year", 0.225),
        ("achievements", 0.205),
        ("price", 0.167),
        ("cat_steam_cloud", 0.156),
        ("developer_historical_success", 0.153),
        ("cat_steam_trading_cards", 0.153),
        ("language_count", 0.143),
    ], columns=["feature", "mean_abs_shap"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(shap_top["feature"][::-1], shap_top["mean_abs_shap"][::-1], color="#2c7fb8")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("Top 10 features driving predictions")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Note: tag_count dominates because well-tagged games correlate strongly "
        "with community engagement. Developer and publisher track records are the "
        "next strongest signals — confirming the EDA finding that who makes the game "
        "is one of the best predictors of success."
    )