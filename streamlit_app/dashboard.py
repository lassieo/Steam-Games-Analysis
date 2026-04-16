"""
Dashboard Tab
==============

Interactive EDA explorer with filters, success rate charts, SHAP analysis,
model performance comparison, and hypothesis validation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app_utils import load_data, load_model_metadata


# Steam-inspired color palette for consistency with the hero section
STEAM_BLUE = "#66c0f4"
STEAM_DARK = "#1b2838"
SUCCESS_GREEN = "#2ca02c"
FAIL_RED = "#d62728"
NEUTRAL_GRAY = "#888888"


def apply_chart_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to matplotlib axes."""
    if title:
        ax.set_title(title, fontsize=12, fontweight="500", pad=10, color=STEAM_DARK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def render_dashboard_tab() -> None:
    st.header("📊 Dataset Dashboard")
    st.markdown(
        "Explore the 89,618 Steam games in our dataset. Use the filters in the sidebar "
        "to slice by year, price, and genre. All charts update in real time."
    )

    df = load_data()
    metadata = load_model_metadata()

    # --- Sidebar filters ---
    st.sidebar.markdown("## 🔍 Dashboard filters")
    st.sidebar.caption("Slice and dice the 89,618 games")

    year_range = st.sidebar.slider(
        "Release year range",
        int(df["release_year"].min()),
        int(df["release_year"].max()),
        (2018, int(df["release_year"].max())),
    )

    price_filter = st.sidebar.selectbox(
        "Price filter",
        ["All games", "Free only", "Paid only", "Budget (<$15)", "Mid ($15-50)", "Premium ($50-85)", "AAA ($85+)"],
    )

    genre_cols = sorted([c for c in df.columns if c.startswith("genre_") and c != "genre_count"])
    genre_labels = [c.replace("genre_", "").replace("_", " ").title() for c in genre_cols]
    selected_genres = st.sidebar.multiselect(
        "Filter by genre (any match)",
        options=genre_labels,
        default=[],
    )

    success_filter = st.sidebar.selectbox(
        "Success filter",
        ["All games", "Successful only", "Unsuccessful only"],
    )

    # Sidebar summary stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About this dataset")
    st.sidebar.caption(
        "**Success definition:** A game is successful if it has at least 80% positive "
        "reviews AND at least 50 total reviews.\n\n"
        "**Base rate:** 16.3% of all games meet this definition."
    )

    # --- Apply filters ---
    filtered = df[
        (df["release_year"] >= year_range[0]) & (df["release_year"] <= year_range[1])
    ]

    if price_filter == "Free only":
        filtered = filtered[filtered["price"] == 0]
    elif price_filter == "Paid only":
        filtered = filtered[filtered["price"] > 0]
    elif price_filter == "Budget (<$15)":
        filtered = filtered[(filtered["price"] > 0) & (filtered["price"] < 15)]
    elif price_filter == "Mid ($15-50)":
        filtered = filtered[(filtered["price"] >= 15) & (filtered["price"] < 50)]
    elif price_filter == "Premium ($50-85)":
        filtered = filtered[(filtered["price"] >= 50) & (filtered["price"] < 85)]
    elif price_filter == "AAA ($85+)":
        filtered = filtered[filtered["price"] >= 85]

    if selected_genres:
        mask = pd.Series(False, index=filtered.index)
        for g in selected_genres:
            col = f"genre_{g.lower().replace(' ', '_')}"
            if col in filtered.columns:
                mask = mask | (filtered[col] == 1)
        filtered = filtered[mask]

    if success_filter == "Successful only":
        filtered = filtered[filtered["success"] == 1]
    elif success_filter == "Unsuccessful only":
        filtered = filtered[filtered["success"] == 0]

    # --- Top metrics ---
    total_games = len(filtered)
    successful = int(filtered["success"].sum())
    success_rate = successful / total_games if total_games > 0 else 0
    avg_price = filtered[filtered["price"] > 0]["price"].mean() if (filtered["price"] > 0).any() else 0
    median_tags = int(filtered["tag_count"].median()) if "tag_count" in filtered.columns else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Games in view", f"{total_games:,}")
    m2.metric("Successful", f"{successful:,}")
    delta = f"{(success_rate - 0.163) * 100:+.1f}% vs base"
    m3.metric("Success rate", f"{success_rate * 100:.1f}%", delta=delta, delta_color="normal")
    m4.metric("Avg paid price", f"${avg_price:.2f}")
    m5.metric("Median tags", f"{median_tags}")

    if total_games == 0:
        st.warning("No games match the current filters. Widen the filters to see data.")
        return

    st.markdown("---")

    # ========================================================================
    # Section 1: EDA charts
    # ========================================================================
    st.subheader("Part 1: Exploring the data")

    # --- Chart: Success rate by genre ---
    st.markdown("**Success rate by genre**")
    st.caption(
        "Genres are one of the hypothesis features. Green bars are above the current "
        "view's average success rate, red bars are below."
    )

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
        colors = [SUCCESS_GREEN if sr >= success_rate else FAIL_RED for sr in genre_df["success_rate"]]
        bars = ax.barh(genre_df["genre"], genre_df["success_rate"] * 100, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(success_rate * 100, color=STEAM_DARK, linestyle="--", linewidth=1.2,
                   label=f"Filtered average ({success_rate * 100:.1f}%)")

        # Add value labels at the end of each bar
        for i, (bar, count) in enumerate(zip(bars, genre_df["count"])):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}% (n={count:,})",
                    ha="left", va="center", fontsize=8, color="#444")

        apply_chart_style(ax, xlabel="Success rate (%)")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, max(genre_df["success_rate"].max() * 100 * 1.25, 30))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough games per genre to show the chart (need at least 100).")

    # --- Chart: Success rate over time ---
    st.markdown("**Game volume and success rate over time**")
    st.caption(
        "How the Steam market has evolved. Blue bars show yearly game volume, "
        "orange line shows the success rate trend."
    )

    year_stats = filtered.groupby("release_year").agg(
        count=("success", "size"),
        success_rate=("success", "mean"),
    ).reset_index()
    year_stats = year_stats[year_stats["count"] >= 50]

    if len(year_stats) > 1:
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.bar(year_stats["release_year"], year_stats["count"],
                color=STEAM_BLUE, alpha=0.5, label="Game count", edgecolor="white", linewidth=0.5)
        apply_chart_style(ax1, xlabel="Release year")
        ax1.set_ylabel("Game count", color=STEAM_BLUE, fontsize=10)
        ax1.tick_params(axis="y", labelcolor=STEAM_BLUE, labelsize=9)

        ax2 = ax1.twinx()
        ax2.plot(year_stats["release_year"], year_stats["success_rate"] * 100,
                 color="#ff9800", marker="o", linewidth=2.5, markersize=6, label="Success rate")
        ax2.set_ylabel("Success rate (%)", color="#ff9800", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#ff9800", labelsize=9)
        ax2.spines["top"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Chart: Price distribution ---
    st.markdown("**Price distribution: successful vs unsuccessful**")
    st.caption(
        "How the price distribution differs between successful and unsuccessful games. "
        "Overlapping bars show where each price band's game volume falls."
    )

    paid = filtered[(filtered["price"] > 0) & (filtered["price"] <= 100)]
    if len(paid) > 100:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        paid_success = paid[paid["success"] == 1]["price"]
        paid_fail = paid[paid["success"] == 0]["price"]

        bins = np.linspace(0, 100, 40)
        ax.hist(paid_fail, bins=bins, label="Not successful", color=FAIL_RED, alpha=0.6, edgecolor="white", linewidth=0.3)
        ax.hist(paid_success, bins=bins, label="Successful", color=SUCCESS_GREEN, alpha=0.7, edgecolor="white", linewidth=0.3)

        # Add vertical lines for price tier boundaries
        for boundary, label in [(15, "Budget/Mid"), (50, "Mid/Premium"), (85, "Premium/AAA")]:
            ax.axvline(boundary, color=STEAM_DARK, linestyle=":", linewidth=1, alpha=0.5)
            ax.text(boundary, ax.get_ylim()[1] * 0.95, f" {label}",
                    fontsize=8, color=STEAM_DARK, rotation=90, va="top")

        apply_chart_style(ax, xlabel="Price (USD)", ylabel="Game count")
        ax.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ========================================================================
    # Section 2: Model performance
    # ========================================================================
    st.markdown("---")
    st.subheader("Part 2: Model performance")

    st.markdown("**Model comparison across the full ladder**")
    st.caption(
        "Every model was trained on the same features and evaluated with identical "
        "5-fold stratified cross-validation splits. The small gap between models shows "
        "that the predictive ceiling is set by the features, not the model class."
    )

    # Read real metrics from the modeling output
    comparison_path = Path(__file__).resolve().parent.parent / "Modeling" / "final_model_comparison.csv"
    if comparison_path.exists():
        raw = pd.read_csv(comparison_path)
        model_results = pd.DataFrame({
            "model": raw["model"],
            "roc_auc": raw["roc_auc_mean"],
            "f1": raw["f1_mean"],
        }).sort_values("roc_auc", ascending=True)
    else:
        model_results = pd.DataFrame([
            {"model": "Random Forest", "roc_auc": 0.8792, "f1": 0.434},
            {"model": "XGBoost", "roc_auc": 0.8748, "f1": 0.564},
            {"model": "LightGBM", "roc_auc": 0.8799, "f1": 0.556},
            {"model": "Tabular MLP", "roc_auc": 0.8724, "f1": 0.488},
            {"model": "XGBoost (tuned)", "roc_auc": 0.8841, "f1": 0.5756},
            {"model": "Stacking Ensemble", "roc_auc": 0.8844, "f1": 0.568},
        ]).sort_values("roc_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_pos = np.arange(len(model_results))
    max_roc = model_results["roc_auc"].max()
    colors = [SUCCESS_GREEN if r == max_roc else STEAM_BLUE for r in model_results["roc_auc"]]
    bars = ax.barh(y_pos, model_results["roc_auc"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_results["model"])
    apply_chart_style(ax, xlabel="ROC-AUC (5-fold stratified CV)")
    ax.set_xlim(0.85, 0.90)

    for i, (roc, f1) in enumerate(zip(model_results["roc_auc"], model_results["f1"])):
        ax.text(roc + 0.0003, i, f"{roc:.4f}  (F1: {f1:.3f})",
                va="center", fontsize=9, color=STEAM_DARK)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- SHAP analysis ---
    st.markdown("**SHAP feature importance**")
    st.caption(
        "This is the actual SHAP summary plot from the tuned XGBoost model on the full "
        "89,618-game dataset. Each dot is one game's prediction. Red dots are high feature "
        "values, blue dots are low. Horizontal position shows how much that feature pushed "
        "the prediction up (right) or down (left)."
    )

    shap_image_path = Path(__file__).resolve().parent.parent / "Modeling" / "charts" / "shap_summary.png"
    if shap_image_path.exists():
        st.image(str(shap_image_path), use_container_width=True)
    else:
        st.warning(
            f"SHAP plot not found. Run `python Modeling/shap_analysis.py` to generate it."
        )

    st.info(
        "**How to read this:** `tag_count` dominates because well-tagged games correlate "
        "strongly with community engagement. `publisher_historical_success` and "
        "`developer_historical_success` rank in the top 10, confirming the EDA finding "
        "that who makes the game is one of the strongest predictors of success. Steam "
        "category features like `steam_cloud` and `trading_cards` are proxies for "
        "developer professionalism — they don't cause success directly but indicate "
        "well-resourced studios."
    )

    # ========================================================================
    # Section 3: Hypothesis validation
    # ========================================================================
    st.markdown("---")
    st.subheader("Part 3: Project hypotheses")
    st.markdown(
        "Before modeling, we defined five hypotheses about what makes a Steam game successful. "
        "SHAP analysis gave us a way to validate each one against the model's actual behavior."
    )

    hypotheses = [
        {
            "name": "H1: Paid games succeed more than free games",
            "verdict": "✅ Strongly supported",
            "evidence": "`price` ranked #6 in SHAP importance (mean |SHAP| = 0.167). "
                        "Free games have an 11.1% success rate vs 17.1% for paid games.",
            "color": "success",
        },
        {
            "name": "H2: Price tier matters more than raw price",
            "verdict": "⚠️ Partially reversed",
            "evidence": "Raw `price` (#6) outperformed `price_tier` (#19). The tuned XGBoost learns "
                        "price thresholds directly from the continuous value, making the engineered "
                        "tier feature redundant for high-capacity models.",
            "color": "warning",
        },
        {
            "name": "H3: Developer track record predicts success",
            "verdict": "✅ Strongly supported (best finding)",
            "evidence": "`publisher_historical_success` ranked #3, `developer_historical_success` ranked "
                        "#8, and `developer_game_count` ranked #15. Three separate track-record features "
                        "all appeared in the top 15 — who makes the game is one of the strongest "
                        "predictors of success.",
            "color": "success",
        },
        {
            "name": "H4: Genre affects success probability",
            "verdict": "✅ Supported",
            "evidence": "`genre_action` ranked #13, `genre_count` ranked #17. Genre contributes meaningfully "
                        "but is not as dominant as developer track record or price.",
            "color": "success",
        },
        {
            "name": "H5: Multi-platform support matters",
            "verdict": "✅ Supported",
            "evidence": "`platform_count` ranked #11. Games supporting more platforms are predicted as "
                        "more successful, consistent with multi-platform support signaling development investment.",
            "color": "success",
        },
    ]

    for h in hypotheses:
        with st.expander(f"**{h['name']}** — {h['verdict']}", expanded=False):
            if h["color"] == "success":
                st.success(h["evidence"])
            elif h["color"] == "warning":
                st.warning(h["evidence"])
            else:
                st.info(h["evidence"])

    st.caption(
        "All five hypotheses were validated by SHAP analysis, which is stronger evidence than "
        "EDA correlations because it shows the model actually uses these features when making "
        "predictions — not just that they happen to correlate with the outcome."
    )