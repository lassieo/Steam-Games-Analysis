"""
Steam Games — EDA & Feature Engineering
=========================================
Comprehensive exploratory analysis designed to inform modeling
decisions, plus full feature engineering pipeline.

This is NOT a surface-level EDA. Every chart and analysis answers a specific
question that affects how we build models:

  SECTION 1: Feature engineering (produces steam_features_engineered.csv)
  SECTION 2: Target analysis (class balance, imbalance strategy)
  SECTION 3: Feature distributions & skewness (log transform decisions)
  SECTION 4: Class separability (which features actually help?)
  SECTION 5: Feature-feature correlations (multicollinearity check)
  SECTION 6: Category coverage & cardinality (encoding decisions)
  SECTION 7: Outlier analysis (capping/clipping decisions)
  SECTION 8: Temporal patterns (concept drift, train/test split strategy)
  SECTION 9: Leakage proof (prove post-launch exclusion was correct)
  SECTION 10: Edge cases (missing, zero-value, rare categories)
  SECTION 11: Success rate breakdowns (hypothesis validation)
  SECTION 12: Insights and modeling recommendations

Reads:  games_march2025_cleaned_binary_success.csv
Writes: steam_features_engineered.csv
        EDA/charts/*.png
        EDA/eda_insights_report.md
"""

from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/pujithaattuluri/Desktop/Spring 2026/Applied ML/Group Project")
SOURCE_FILE = BASE_DIR / "games_march2025_cleaned_binary_success.csv"
OUTPUT_CSV = BASE_DIR / "steam_features_engineered.csv"
CHARTS_DIR = Path(__file__).resolve().parent / "charts"
OUTPUT_REPORT = Path(__file__).resolve().parent / "eda_insights_report.md"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_SUCCESS = "#1D9E75"
C_FAIL = "#E24B4A"
C_NEUTRAL = "#378ADD"
C_ACCENT = "#EF9F27"
C_PAIR = [C_FAIL, C_SUCCESS]
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      {path.name}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def safe_parse(value, fallback=None):
    if pd.isna(value): return fallback
    try: return literal_eval(str(value))
    except (ValueError, SyntaxError): return fallback


# ===================================================================
# SECTION 1: FEATURE ENGINEERING
# ===================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Returns enriched dataframe."""

    # Temporal
    dt = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = dt.dt.year
    df["release_month"] = dt.dt.month
    df["release_day_of_week"] = dt.dt.dayofweek
    df["release_quarter"] = dt.dt.quarter
    df["released_in_sale_month"] = dt.dt.month.isin([6, 11, 12]).astype(int)
    df["years_since_release"] = 2025 - df["release_year"]

    # Platform
    for col in ("windows", "mac", "linux"):
        df[col] = df[col].map({True:1, False:0, "True":1, "False":0}).fillna(0).astype(int)
    df["platform_count"] = df["windows"] + df["mac"] + df["linux"]

    # Languages
    df["language_count"] = df["supported_languages"].apply(lambda x: len(safe_parse(x, [])))
    df["audio_language_count"] = df["full_audio_languages"].apply(lambda x: len(safe_parse(x, [])))

    # Price
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    # Cap extreme prices (games > $200 are joke listings or asset flips)
    n_extreme = int((df["price"] > 200).sum())
    if n_extreme > 0:
        print(f"    Capping {n_extreme} games with price > $200")
        df["price"] = df["price"].clip(upper=200)
    df["is_free"] = (df["price"] == 0).astype(int)
    bins = [-0.01, 0, 4.99, 14.99, 29.99, np.inf]
    df["price_tier"] = pd.cut(df["price"], bins=bins, labels=[0,1,2,3,4]).astype(float).fillna(0).astype(int)

    # Genres
    df["_gl"] = df["genres"].apply(lambda x: safe_parse(x, []))
    top_genres = df[["_gl"]].explode("_gl")["_gl"].value_counts().head(15).index.tolist()
    for g in top_genres:
        df["genre_" + g.lower().replace(" ", "_").replace("-", "_")] = df["_gl"].apply(lambda lst: int(g in lst))
    df["genre_count"] = df["_gl"].apply(len)
    df["has_genres"] = (df["genre_count"] > 0).astype(int)

    # Categories
    df["_cl"] = df["categories"].apply(lambda x: safe_parse(x, []))
    top_cats = df[["_cl"]].explode("_cl")["_cl"].value_counts().head(12).index.tolist()
    for c in top_cats:
        df["cat_" + c.lower().replace(" ","_").replace("-","_").replace("'","")] = df["_cl"].apply(lambda lst: int(c in lst))
    df["category_count"] = df["_cl"].apply(len)
    df["has_categories"] = (df["category_count"] > 0).astype(int)

    # Tags
    df["_td"] = df["tags"].apply(lambda x: safe_parse(x, {}))
    tag_totals = {}
    for tags in df["_td"]:
        if isinstance(tags, dict):
            for tag, cnt in tags.items():
                tag_totals[tag] = tag_totals.get(tag, 0) + cnt
    top_tags = sorted(tag_totals, key=tag_totals.get, reverse=True)[:20]
    for tag in top_tags:
        df["tag_" + tag.lower().replace(" ","_").replace("-","_").replace("'","")] = \
            df["_td"].apply(lambda d: int(tag in d) if isinstance(d, dict) else 0)
    df["tag_count"] = df["_td"].apply(lambda d: len(d) if isinstance(d, dict) else 0)
    df["has_tags"] = (df["tag_count"] > 0).astype(int)

    # Achievements
    df["achievements"] = pd.to_numeric(df.get("achievements", 0), errors="coerce").fillna(0)
    df["has_achievements"] = (df["achievements"] > 0).astype(int)

    # Description & website
    df["description_length"] = df["short_description"].fillna("").apply(len)
    df["has_website"] = (df["website"].notna() & (df["website"] != "")).astype(int)

    # Developer track record
    print("    Computing developer/publisher track records ...")
    df["_dev"] = df["developers"].apply(lambda x: (safe_parse(x, ["Unknown"]) or ["Unknown"])[0])
    df["_pub"] = df["publishers"].apply(lambda x: (safe_parse(x, ["Unknown"]) or ["Unknown"])[0])
    df["_rdt"] = pd.to_datetime(df["release_date"], errors="coerce")
    base_rate = df["success"].mean()

    for entity_col, out_col in [("_dev", "developer_historical_success"), ("_pub", "publisher_historical_success")]:
        df_s = df.sort_values("_rdt").copy()
        hist = {}
        rates = []
        for _, row in df_s.iterrows():
            e = row[entity_col]
            h = hist.get(e, [])
            rates.append(np.mean(h) if h else base_rate)
            hist.setdefault(e, []).append(row["success"])
        df_s[out_col] = rates
        df[out_col] = df_s[out_col]

    df_s = df.sort_values("_rdt").copy()
    dev_counts = {}
    dc = []
    for _, row in df_s.iterrows():
        d = row["_dev"]
        dc.append(dev_counts.get(d, 0))
        dev_counts[d] = dev_counts.get(d, 0) + 1
    df_s["developer_game_count"] = dc
    df["developer_game_count"] = df_s["developer_game_count"]

    # EDA reference (post-launch, eda_ prefix)
    def parse_owners(v):
        if pd.isna(v) or str(v) == "0": return 0.0
        parts = str(v).replace(",","").split(" - ")
        return (float(parts[0])+float(parts[1]))/2 if len(parts)==2 else float(parts[0])

    df["eda_owners_midpoint"] = df["estimated_owners"].apply(parse_owners)
    df["eda_log_owners"] = np.log1p(df["eda_owners_midpoint"])
    for col in ("peak_ccu","average_playtime_forever","median_playtime_forever","metacritic_score","dlc_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["eda_peak_ccu"] = df["peak_ccu"]
    df["eda_log_peak_ccu"] = np.log1p(df["peak_ccu"])
    df["eda_avg_playtime"] = df["average_playtime_forever"]
    df["eda_median_playtime"] = df["median_playtime_forever"]
    df["eda_metacritic_score"] = df["metacritic_score"]
    df["eda_has_metacritic"] = (df["metacritic_score"] > 0).astype(int)
    df["eda_dlc_count"] = pd.to_numeric(df["dlc_count"], errors="coerce").fillna(0)

    df.drop(columns=["_gl","_cl","_td","_dev","_pub","_rdt"], inplace=True, errors="ignore")
    return df


# ===================================================================
# SECTION 2: TARGET ANALYSIS
# ===================================================================
def eda_target(df, charts_dir):
    """Class balance and imbalance strategy."""
    counts = df["success"].value_counts().sort_index()
    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    bars = axes[0].bar(["Unsuccessful (0)","Successful (1)"],
                       [counts.get(0,0), counts.get(1,0)], color=C_PAIR, width=0.5)
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+n*0.005,
                     f"{int(bar.get_height()):,}\n({bar.get_height()/n*100:.1f}%)",
                     ha="center", va="bottom", fontsize=9)
    style_ax(axes[0], "Class distribution", ylabel="Number of games")

    # Pie for ratio
    axes[1].pie([counts.get(0,0), counts.get(1,0)], labels=["Unsuccessful","Successful"],
                colors=C_PAIR, autopct="%1.1f%%", startangle=90, textprops={"fontsize":10})
    style_ax(axes[1], "Class ratio")

    save_fig(fig, charts_dir / "01_class_balance.png")
    return {"pct_success": counts.get(1,0)/n*100, "ratio": counts.get(0,0)/max(counts.get(1,0),1)}


# ===================================================================
# SECTION 3: FEATURE DISTRIBUTIONS & SKEWNESS
# ===================================================================
def eda_distributions(df, charts_dir):
    """Distribution of key numeric features, split by success class."""
    features = ["price","achievements","language_count","audio_language_count",
                "description_length","developer_game_count","genre_count","platform_count"]
    available = [f for f in features if f in df.columns]

    skewness = {}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, feat in enumerate(available[:8]):
        ax = axes[i]
        vals = pd.to_numeric(df[feat], errors="coerce").dropna()
        skew = vals.skew()
        skewness[feat] = skew

        for label, color, val in [(0, C_FAIL, df[df["success"]==0][feat]),
                                   (1, C_SUCCESS, df[df["success"]==1][feat])]:
            v = pd.to_numeric(val, errors="coerce").dropna()
            ax.hist(v, bins=30, alpha=0.5, color=color, label=f"{'Succ' if label else 'Fail'}", density=True)
        ax.set_title(f"{feat}\nskew={skew:.2f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(available), 8):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions by success class (density)", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, charts_dir / "02_feature_distributions.png")
    return skewness


# ===================================================================
# SECTION 4: CLASS SEPARABILITY
# ===================================================================
def eda_separability(df, charts_dir):
    """How well does each feature separate the two classes?"""
    features = ["price","achievements","language_count","platform_count",
                "description_length","developer_historical_success",
                "developer_game_count","genre_count","tag_count","has_website"]
    available = [f for f in features if f in df.columns]

    separability = {}
    for feat in available:
        s0 = pd.to_numeric(df[df["success"]==0][feat], errors="coerce").dropna()
        s1 = pd.to_numeric(df[df["success"]==1][feat], errors="coerce").dropna()
        pooled_std = np.sqrt((s0.var()*len(s0) + s1.var()*len(s1)) / (len(s0)+len(s1)))
        cohens_d = abs(s1.mean() - s0.mean()) / max(pooled_std, 0.001)
        separability[feat] = {"mean_0": s0.mean(), "mean_1": s1.mean(), "cohens_d": cohens_d}

    # Plot
    sorted_feats = sorted(separability.items(), key=lambda x: x[1]["cohens_d"], reverse=True)
    names = [f[0] for f in sorted_feats]
    d_vals = [f[1]["cohens_d"] for f in sorted_feats]
    colors = [C_SUCCESS if d > 0.2 else C_ACCENT if d > 0.1 else C_FAIL for d in d_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(names)), d_vals, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_"," ").title() for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.2, -0.5, "small effect", fontsize=7, color="gray")
    for i, v in enumerate(d_vals):
        ax.text(v+0.01, i, f"{v:.3f}", va="center", fontsize=8)
    style_ax(ax, "Class separability (Cohen's d)", xlabel="Cohen's d (higher = more separable)")
    save_fig(fig, charts_dir / "03_class_separability.png")
    return separability


# ===================================================================
# SECTION 5: FEATURE-FEATURE CORRELATIONS
# ===================================================================
def eda_multicollinearity(df, charts_dir):
    """Correlation heatmap of model features to detect redundancy."""
    features = ["price","is_free","achievements","platform_count","language_count",
                "audio_language_count","description_length","has_website",
                "developer_historical_success","publisher_historical_success",
                "developer_game_count","genre_count","category_count","tag_count",
                "release_year","years_since_release"]
    available = [f for f in features if f in df.columns]

    corr = df[available].apply(pd.to_numeric, errors="coerce").corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, square=True, linewidths=0.5, cbar_kws={"shrink":0.8},
                annot_kws={"size":7},
                xticklabels=[c.replace("_"," ").title() for c in available],
                yticklabels=[c.replace("_"," ").title() for c in available])
    ax.tick_params(labelsize=8)
    style_ax(ax, "Feature-feature correlation matrix")
    save_fig(fig, charts_dir / "04_multicollinearity.png")

    # Find highly correlated pairs
    high_corr = []
    for i in range(len(available)):
        for j in range(i+1, len(available)):
            r = abs(corr.iloc[i,j])
            if r > 0.7:
                high_corr.append((available[i], available[j], corr.iloc[i,j]))
    return high_corr


# ===================================================================
# SECTION 6: CATEGORY COVERAGE
# ===================================================================
def eda_cardinality(df, charts_dir):
    """Check how well our top-N encoding covers the data."""
    results = {}
    for prefix, label in [("genre_","Genres"), ("cat_","Categories"), ("tag_","Tags")]:
        cols = [c for c in df.columns if c.startswith(prefix) and c not in [f"{prefix}count"]]
        if not cols: continue
        has_any = df[cols].max(axis=1) > 0
        results[label] = {"n_encoded": len(cols), "coverage": has_any.mean(),
                          "games_uncovered": int((~has_any).sum())}

    # Success rate by genre
    genre_cols = [c for c in df.columns if c.startswith("genre_") and c != "genre_count"]
    genre_rates = {}
    for col in genre_cols:
        subset = df[df[col]==1]
        if len(subset) >= 30:
            genre_rates[col.replace("genre_","").replace("_"," ").title()] = {
                "rate": subset["success"].mean(), "count": len(subset)}

    if genre_rates:
        sorted_g = sorted(genre_rates.items(), key=lambda x: x[1]["rate"], reverse=True)
        fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_g)*0.4)))
        names = [g[0] for g in sorted_g]
        rates = [g[1]["rate"] for g in sorted_g]
        counts = [g[1]["count"] for g in sorted_g]
        bars = ax.barh(range(len(names)), rates, color=C_NEUTRAL, height=0.6)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_width()+0.005, i, f"{rates[i]:.0%} (n={count:,})", va="center", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        style_ax(ax, "Success rate by genre", xlabel="Success rate")
        save_fig(fig, charts_dir / "05_success_by_genre.png")

    return results


# ===================================================================
# SECTION 7: OUTLIER ANALYSIS
# ===================================================================
def eda_outliers(df, charts_dir):
    """
    Comprehensive outlier analysis that answers three questions:
      1. How many outliers exist and how extreme are they?
      2. Do outliers carry signal (are they disproportionately in one class)?
      3. What treatment is needed, and does it differ by model type?
    """
    features = ["price","achievements","language_count","description_length","developer_game_count"]
    available = [f for f in features if f in df.columns]

    outlier_info = {}
    n_total = len(df)

    # --- Chart 1: Box plots with IQR bounds ---
    fig, axes = plt.subplots(1, len(available), figsize=(3*len(available), 4))
    if len(available) == 1: axes = [axes]

    for i, feat in enumerate(available):
        vals = pd.to_numeric(df[feat], errors="coerce").dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        p95 = vals.quantile(0.95)
        p99 = vals.quantile(0.99)
        n_outliers = int(((vals < lower) | (vals > upper)).sum())

        # Check if outliers are disproportionately in one class
        outlier_mask = (df[feat].apply(pd.to_numeric, errors="coerce") < lower) | \
                       (df[feat].apply(pd.to_numeric, errors="coerce") > upper)
        s0_pct = outlier_mask[df["success"]==0].mean() * 100 if (df["success"]==0).sum() > 0 else 0
        s1_pct = outlier_mask[df["success"]==1].mean() * 100 if (df["success"]==1).sum() > 0 else 0
        signal_bias = "successful" if s1_pct > s0_pct * 1.5 else \
                      "unsuccessful" if s0_pct > s1_pct * 1.5 else "balanced"

        # What happens if we cap?
        capped_mean = vals.clip(lower=max(lower, 0), upper=upper).mean()

        outlier_info[feat] = {
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "p95": p95, "p99": p99,
            "n_outliers": n_outliers,
            "pct_outlier": n_outliers / len(vals) * 100,
            "skewness": vals.skew(),
            "outlier_in_successful_pct": s1_pct,
            "outlier_in_unsuccessful_pct": s0_pct,
            "signal_bias": signal_bias,
            "original_mean": vals.mean(),
            "capped_mean": capped_mean,
            "max_value": vals.max(),
            "treatment": "log1p" if vals.skew() > 2 else "cap_p99" if n_outliers/len(vals) > 0.05 else "none",
        }

        axes[i].boxplot(vals, vert=True, widths=0.5,
                        boxprops=dict(color=C_NEUTRAL), medianprops=dict(color=C_ACCENT, linewidth=2),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        axes[i].set_title(f"{feat}\n{n_outliers:,} outliers ({n_outliers/len(vals)*100:.1f}%)"
                          f"\nskew={vals.skew():.1f}", fontsize=8)
        axes[i].tick_params(labelsize=8)

    fig.suptitle("Outlier analysis (IQR method)", fontsize=12, fontweight="bold", y=1.02)
    save_fig(fig, charts_dir / "06a_outlier_boxplots.png")

    # --- Chart 2: Outlier signal analysis (do outliers carry class info?) ---
    fig, ax = plt.subplots(figsize=(8, max(3, len(available)*0.6)))
    y_pos = range(len(available))
    s0_pcts = [outlier_info[f]["outlier_in_unsuccessful_pct"] for f in available]
    s1_pcts = [outlier_info[f]["outlier_in_successful_pct"] for f in available]

    ax.barh([y-0.15 for y in y_pos], s0_pcts, height=0.3, color=C_FAIL, label="Unsuccessful", alpha=0.8)
    ax.barh([y+0.15 for y in y_pos], s1_pcts, height=0.3, color=C_SUCCESS, label="Successful", alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f.replace("_"," ").title() for f in available], fontsize=9)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    for i in y_pos:
        if s0_pcts[i] > 0 or s1_pcts[i] > 0:
            ax.text(max(s0_pcts[i], s1_pcts[i])+0.3, i,
                    f"bias: {outlier_info[available[i]]['signal_bias']}", fontsize=7, va="center")
    style_ax(ax, "Outlier distribution by class\n(if biased toward one class, removing outliers would destroy signal)",
             xlabel="% of class that are outliers")
    save_fig(fig, charts_dir / "06b_outlier_signal.png")

    # --- Chart 3: Treatment comparison (original vs capped vs log) ---
    log_candidates = [f for f in available if outlier_info[f]["skewness"] > 1.5]
    if log_candidates:
        fig, axes = plt.subplots(1, len(log_candidates), figsize=(4*len(log_candidates), 3.5))
        if len(log_candidates) == 1: axes = [axes]
        for i, feat in enumerate(log_candidates):
            vals = pd.to_numeric(df[feat], errors="coerce").dropna()
            log_vals = np.log1p(vals)
            axes[i].hist(vals, bins=40, alpha=0.5, color=C_FAIL, label="Original", density=True)
            ax2 = axes[i].twinx()
            ax2.hist(log_vals, bins=40, alpha=0.5, color=C_SUCCESS, label="After log1p", density=True)
            axes[i].set_title(f"{feat}\nskew: {vals.skew():.1f} → {log_vals.skew():.1f}", fontsize=9)
            axes[i].legend(loc="upper right", fontsize=7)
            ax2.legend(loc="upper left", fontsize=7)
            ax2.tick_params(labelsize=7)
            axes[i].tick_params(labelsize=7)
        fig.suptitle("Log transform effect on skewed features", fontsize=12, fontweight="bold", y=1.02)
        save_fig(fig, charts_dir / "06c_log_transform_effect.png")

    return outlier_info
    return outlier_info


# ===================================================================
# SECTION 8: TEMPORAL PATTERNS
# ===================================================================
def eda_temporal(df, charts_dir):
    """Success rate over time — concept drift check."""
    yearly = df.groupby("release_year")["success"].agg(["mean","count"])
    yearly = yearly[yearly["count"] >= 20]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.bar(yearly.index, yearly["count"], color=C_NEUTRAL, alpha=0.25, width=0.8, label="Game count")
    ax2.plot(yearly.index, yearly["mean"], color=C_SUCCESS, marker="o", linewidth=2, markersize=4, label="Success rate")

    ax1.set_xlabel("Release year")
    ax1.set_ylabel("Games released", color=C_NEUTRAL)
    ax2.set_ylabel("Success rate", color=C_SUCCESS)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.set_title("Games released and success rate over time", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper left", fontsize=8)
    save_fig(fig, charts_dir / "07_temporal_trend.png")

    # Concept drift: compare early vs late success rates
    if len(yearly) > 4:
        mid = yearly.index[len(yearly)//2]
        early = yearly.loc[yearly.index <= mid, "mean"].mean()
        late = yearly.loc[yearly.index > mid, "mean"].mean()
        return {"early_rate": early, "late_rate": late, "drift": late - early}
    return {}


# ===================================================================
# SECTION 9: LEAKAGE PROOF
# ===================================================================
def eda_leakage_proof(df, charts_dir):
    """Prove that post-launch columns would leak information."""
    eda_cols = [c for c in df.columns if c.startswith("eda_") and df[c].dtype in ["float64","int64"]]
    if not eda_cols: return {}

    corrs = {}
    for col in eda_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        corrs[col] = vals.corr(df["success"])

    # Plot
    sorted_c = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [c[0].replace("eda_","") for c in sorted_c]
    vals = [c[1] for c in sorted_c]
    colors = [C_FAIL if abs(v) > 0.3 else C_ACCENT if abs(v) > 0.15 else C_SUCCESS for v in vals]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(range(len(names)), vals, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_"," ").title() for n in names], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v+(0.01 if v>=0 else -0.01), i, f"{v:.3f}", va="center",
                ha="left" if v>=0 else "right", fontsize=8)
    style_ax(ax, "Post-launch column correlation with success\n(proves why these are excluded from modeling)",
             xlabel="Pearson correlation")
    save_fig(fig, charts_dir / "08_leakage_proof.png")
    return corrs


# ===================================================================
# SECTION 10: EDGE CASES
# ===================================================================
def eda_edge_cases(df):
    """Count problematic edge cases that could affect modeling."""
    cases = {
        "Games with 0 genres": int((df.get("genre_count", pd.Series(dtype=int)) == 0).sum()),
        "Games with 0 categories": int((df.get("category_count", pd.Series(dtype=int)) == 0).sum()),
        "Games with 0 tags": int((df.get("tag_count", pd.Series(dtype=int)) == 0).sum()),
        "Games with platform_count=0": int((df.get("platform_count", pd.Series(dtype=int)) == 0).sum()),
        "Games with price > $200": int((pd.to_numeric(df["price"], errors="coerce").fillna(0) > 200).sum()),
        "Games with 0 achievements": int((pd.to_numeric(df.get("achievements", pd.Series(dtype=int)), errors="coerce").fillna(0) == 0).sum()),
        "Games with no release date": int(pd.to_datetime(df["release_date"], errors="coerce").isna().sum()),
        "Games with description_length=0": int((df.get("description_length", pd.Series(dtype=int)) == 0).sum()),
    }
    return cases


# ===================================================================
# SECTION 11: SUCCESS RATE BREAKDOWNS
# ===================================================================
def eda_success_breakdowns(df, charts_dir):
    """Success rate by price tier, platform count, and free vs paid."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Price tier
    tier_labels = {0:"Free", 1:"Budget\n(<$5)", 2:"Mid\n($5-15)", 3:"Premium\n($15-30)", 4:"AAA\n($30+)"}
    g1 = df.groupby("price_tier")["success"].agg(["mean","count"])
    axes[0].bar(range(len(g1)), g1["mean"], color=C_NEUTRAL, width=0.5)
    axes[0].set_xticks(range(len(g1)))
    axes[0].set_xticklabels([tier_labels.get(i,"") for i in g1.index], fontsize=8)
    for i, (_, row) in enumerate(g1.iterrows()):
        axes[0].text(i, row["mean"]+0.01, f"{row['mean']:.0%}\nn={int(row['count']):,}",
                     ha="center", fontsize=7)
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(axes[0], "Success by price tier", ylabel="Success rate")

    # Platform count
    g2 = df.groupby("platform_count")["success"].agg(["mean","count"])
    axes[1].bar(range(len(g2)), g2["mean"], color=C_NEUTRAL, width=0.4)
    axes[1].set_xticks(range(len(g2)))
    axes[1].set_xticklabels([f"{int(i)} plat." for i in g2.index], fontsize=8)
    for i, (_, row) in enumerate(g2.iterrows()):
        axes[1].text(i, row["mean"]+0.01, f"{row['mean']:.0%}\nn={int(row['count']):,}",
                     ha="center", fontsize=7)
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(axes[1], "Success by platform count", ylabel="Success rate")

    # Free vs paid
    g3 = df.groupby("is_free")["success"].agg(["mean","count"])
    axes[2].bar(["Paid","Free"], [g3.loc[0,"mean"] if 0 in g3.index else 0,
                                   g3.loc[1,"mean"] if 1 in g3.index else 0],
                color=[C_SUCCESS, C_FAIL], width=0.4)
    for i, idx in enumerate([0,1]):
        if idx in g3.index:
            axes[2].text(i, g3.loc[idx,"mean"]+0.01,
                         f"{g3.loc[idx,'mean']:.0%}\nn={int(g3.loc[idx,'count']):,}",
                         ha="center", fontsize=7)
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(axes[2], "Success: paid vs free", ylabel="Success rate")

    save_fig(fig, charts_dir / "09_success_breakdowns.png")


# ===================================================================
# SECTION 12: REPORT GENERATION
# ===================================================================
def generate_report(df, balance, skewness, separability, high_corr, cardinality,
                    outliers, temporal, leakage, edge_cases, model_features, eda_cols):
    """Generate comprehensive EDA insights report."""
    n = len(df)
    n_s = int(df["success"].sum())

    # Format sections
    skew_lines = "\n".join(f"| {f} | {s:.2f} | {'Yes — use log1p' if abs(s)>2 else 'No'} |"
                           for f, s in sorted(skewness.items(), key=lambda x: abs(x[1]), reverse=True))

    sep_lines = "\n".join(f"| {f.replace('_',' ').title()} | {v['mean_0']:.2f} | {v['mean_1']:.2f} | {v['cohens_d']:.3f} | {'Strong' if v['cohens_d']>0.3 else 'Moderate' if v['cohens_d']>0.15 else 'Weak'} |"
                          for f, v in sorted(separability.items(), key=lambda x: x[1]['cohens_d'], reverse=True))

    corr_lines = "\n".join(f"| {a.replace('_',' ').title()} vs {b.replace('_',' ').title()} | {r:.3f} | Consider dropping one |"
                           for a, b, r in high_corr) if high_corr else "| (none above 0.7) | — | — |"

    outlier_lines = "\n".join(
        f"| {f} | {v['n_outliers']:,} | {v['pct_outlier']:.1f}% | {v['skewness']:.1f} | {v['signal_bias']} | {v['treatment']} |"
        for f, v in sorted(outliers.items(), key=lambda x: x[1]['pct_outlier'], reverse=True))

    edge_lines = "\n".join(f"| {k} | {v:,} |" for k, v in edge_cases.items())

    leak_lines = "\n".join(f"| {k.replace('eda_','').replace('_',' ').title()} | {v:.3f} | {'HIGH' if abs(v)>0.3 else 'moderate' if abs(v)>0.15 else 'low'} |"
                           for k, v in sorted(leakage.items(), key=lambda x: abs(x[1]), reverse=True)) if leakage else "| (not computed) | — | — |"

    drift_text = ""
    if temporal:
        drift_text = f"Early years avg success rate: {temporal.get('early_rate',0):.1%}, Late years: {temporal.get('late_rate',0):.1%}, Drift: {temporal.get('drift',0):+.1%}"

    return f"""# EDA & Insights Report

## Dataset summary
- Total games: {n:,}
- Successful: {n_s:,} ({n_s/n*100:.1f}%) | Unsuccessful: {n-n_s:,} ({(n-n_s)/n*100:.1f}%)
- Model features: {len(model_features)} | EDA reference: {len(eda_cols)}
- Class imbalance ratio: {balance['ratio']:.1f}:1

## Charts produced
All saved to `EDA/charts/`:
1. `01_class_balance.png` — Target variable distribution
2. `02_feature_distributions.png` — Feature histograms split by class
3. `03_class_separability.png` — Cohen's d for each feature
4. `04_multicollinearity.png` — Feature-feature correlation heatmap
5. `05_success_by_genre.png` — Success rate by genre
6. `06_outliers.png` — Box plots with outlier counts
7. `07_temporal_trend.png` — Success rate over release years
8. `08_leakage_proof.png` — Post-launch correlation with success
9. `09_success_breakdowns.png` — Success by price tier, platform, free/paid

---

## Insight 1: Class imbalance requires careful handling
Only {balance['pct_success']:.1f}% of games are successful. The imbalance ratio is {balance['ratio']:.1f}:1.

**Modeling decision:** Use `class_weight='balanced'` for all tree models. Evaluate SMOTE on the training fold only (never on validation). Primary metric is ROC-AUC, not accuracy.

---

## Insight 2: Feature skewness — log transforms needed for MLP
| Feature | Skewness | Needs log transform? |
|---|---:|---|
{skew_lines}

**Modeling decision:** Apply `np.log1p()` to features with skewness > 2 before feeding into the Tabular MLP. Tree-based models (RF, XGBoost) are robust to skewness and do not need this transformation.

---

## Insight 3: Feature separability — which features actually help
| Feature | Mean (fail) | Mean (success) | Cohen's d | Strength |
|---|---:|---:|---:|---|
{sep_lines}

**Modeling decision:** Features with Cohen's d > 0.2 are strong candidates. Features with d < 0.05 may be dropped if they add noise. Developer track record and price should be among the most important features.

---

## Insight 4: Multicollinearity — redundant feature pairs
| Feature pair | Correlation | Action |
|---|---:|---|
{corr_lines}

**Modeling decision:** Drop one feature from any pair with correlation > 0.85. For pairs in the 0.7-0.85 range, keep both but monitor feature importance — if both rank low, drop the weaker one. Tree models handle this naturally; MLP is sensitive.

---

## Insight 5: Outlier analysis — how many, do they carry signal, and what to do
| Feature | Outliers | % of data | Skewness | Signal bias | Treatment |
|---|---:|---:|---:|---|---|
{outlier_lines}

**Signal bias** indicates whether outliers are disproportionately in one class. If biased toward "successful," removing outliers would destroy predictive signal. Capping preserves the direction while limiting magnitude.

**Treatment strategy by model type:**
- **RF / XGBoost / LightGBM:** No outlier treatment needed. Tree models split on rank order, not magnitude. A game priced at $999 vs $60 makes no difference if the split threshold is at $30.
- **MLP (Neural Network):** Treatment required because the network does arithmetic with raw values. Large outliers cause gradient explosion and dominate weight updates.
  - Features with skewness > 2: Apply `np.log1p()` to compress the right tail naturally.
  - Features with skewness 1-2 and >5% outliers: Cap at the 99th percentile.
  - After treatment: Apply `StandardScaler` to normalize all features to mean=0, std=1.
- **Important:** Never remove outlier rows. Capping or transforming values preserves all training examples while limiting their numerical influence.

---

## Insight 6: Temporal concept drift
{drift_text}

**Modeling decision:** If success rates have shifted significantly over time, consider using a time-based train/test split (train on pre-2022, test on 2022+) instead of random splitting. This simulates real-world deployment where the model predicts future games.

---

## Insight 7: Leakage proof — why post-launch columns are excluded
| Column | Correlation with success | Risk level |
|---|---:|---|
{leak_lines}

**Modeling decision:** These correlations confirm that including post-launch metrics would inflate model performance artificially. The exclusion decision from Week 1 is validated.

---

## Insight 8: Edge cases to handle before modeling
| Condition | Count |
|---|---:|
{edge_lines}

**Modeling decision:** Games with 0 genres/categories/tags should be imputed as a separate "Unknown" category or dropped if count is small. Games with no release date need imputation or removal.

---

## Hypothesis validation summary

| Hypothesis | Result | Key evidence |
|---|---|---|
| Paid games succeed more than free games | To validate | Compare success rates by is_free |
| Multi-platform games succeed more | To validate | Success rate by platform_count |
| Developer track record predicts success | To validate | Cohen's d and correlation analysis |
| Genre affects success probability | To validate | Success rate varies significantly by genre |
| Price tier matters more than raw price | To validate | Compare model performance with/without tiers |
| Post-launch metrics would cause leakage | Validated | High correlations in leakage proof chart |

---

## Modeling recommendations
1. **Train/test split:** Consider time-based split if concept drift is significant; otherwise 80/20 stratified random split.
2. **Class balancing:** `class_weight='balanced'` for RF/XGBoost. SMOTE only on training folds.
3. **Feature preprocessing:** Log-transform skewed features for MLP only. Standardize all features for MLP.
4. **Feature selection:** After initial RF, use SHAP importance to identify and potentially drop low-signal features.
5. **Evaluation:** ROC-AUC primary, F1 secondary. Report confusion matrix, precision, recall for both classes.

## What is applied in the CSV vs deferred to modeling

| EDA recommendation | Applied in CSV? | Why |
|---|---|---|
| Drop `years_since_release` (redundant with `release_year`, corr=-1.0) | Yes | Redundancy is always wrong, regardless of model type |
| Fill NaN values in numeric features with median | Yes | All models need complete data |
| Add `has_tags` flag (18.8% of games have 0 tags) | Yes | Lets models distinguish "no tag data" from "tags present but not matching top 20" |
| Add `has_achievements` flag (45.8% of games have 0) | Yes | Lets models separate "no achievements" signal from "how many achievements" |
| Add `has_genres`, `has_categories` flags | Yes | Same pattern for sparse categorical coverage |
| Cap price at $200 (8 extreme outliers) | Yes | These are joke/asset-flip listings, not real games. Safe for all models. |
| Log-transform skewed features (achievements, developer_game_count) | No — deferred | Tree models perform worse with log transforms. Applied per-model in training. |
| Cap outliers at 99th percentile | No — deferred | Only needed for MLP. Trees are robust to outliers. |
| StandardScaler normalization | No — deferred | Only needed for MLP. Trees are invariant to scaling. |
| Drop low-separability features (tag_count, description_length) | No — deferred | Let SHAP confirm after training rather than pre-judging. |

## Issues and challenges

### 1. Class imbalance ({balance['pct_success']:.1f}% positive class)
Only about 1 in 6 games meets our success threshold. If a model simply predicted "not successful" for every game, it would achieve ~84% accuracy while being completely useless. This is the most significant technical challenge in the project because it affects every model we train.

**How we address it:** We use `class_weight='balanced'` in tree-based models, which penalizes misclassifying the minority class more heavily. We evaluate SMOTE oversampling on training folds only (never on validation data, which would leak synthetic information). Most importantly, we use ROC-AUC as our primary metric instead of accuracy, because ROC-AUC measures how well the model ranks games regardless of the class distribution.

### 2. Success metric captures sentiment, not revenue
Our definition of success is based on review sentiment (80% positive ratio with at least 50 reviews). This means a game can be commercially profitable but labeled "unsuccessful" if it has polarized reviews — for example, PUBG has over 50 million estimated owners but only 59% positive reviews due to its competitive community. Conversely, a niche indie game with 60 reviews that are 95% positive is labeled "successful" despite minimal commercial impact.

**How we address it:** We validate our sentiment-based label against commercial metrics (estimated owners, peak concurrent users) in the EDA using `eda_` prefixed columns. The correlation analysis shows whether our label aligns with commercial outcomes. We document this as a known scope limitation: our model predicts community reception, which is one meaningful dimension of success but not the only one.

### 3. Survivorship bias
The dataset only contains games currently listed on Steam. Games that were delisted, removed, or shut down — which are disproportionately failures — are absent. This means our dataset underrepresents the "unsuccessful" category in ways we cannot measure, and our model may overestimate success rates for the broader population of games.

**How we address it:** This is an inherent limitation of the data source that cannot be fixed without access to historical Steam records. We document it transparently and note that model predictions apply to "games that remain on Steam" rather than all games ever released.

### 4. Review bombing
Competitive multiplayer games (PUBG, Helldivers 2, Destiny 2) and games involved in controversies receive coordinated negative review campaigns that depress their positive ratio independently of game quality. These campaigns can shift a game from "successful" to "unsuccessful" under our metric based on community politics rather than the game itself.

**How we address it:** We do not filter or adjust for review bombing because any detection method would introduce subjective judgment about which negative reviews are "legitimate." Instead, we acknowledge that our model learns from community sentiment as-is, including its distortions. The `pct_pos_recent` field (available in the raw data but excluded from modeling) could be used in future work to detect sentiment shifts.

### 5. Feature sparsity in one-hot encoded columns
Our top-15 genre, top-12 category, and top-20 tag binary features cover the majority of games, but rare combinations create sparse rows where most binary columns are zero. Games with 0 tags (16,872 games, 18.8%) have all 20 tag columns set to zero, which is indistinguishable from "has tags but none match our top 20" without the `has_tags` flag we added.

**How we address it:** We added `has_tags`, `has_achievements`, `has_genres`, and `has_categories` binary flags to let models distinguish between "data not present" and "data present but not matching encoded categories." For the MLP, sparse binary features may add noise — we will monitor SHAP importance and consider dropping low-signal one-hot columns if they hurt performance.

## Tasks remaining
1. Model training: RF → XGBoost/LightGBM → MLP → Stacking Ensemble
2. Hyperparameter tuning with Optuna (Bayesian optimization)
3. SHAP feature importance analysis
4. K-Means clustering for market archetypes
5. Interactive prediction tool
"""


# ===================================================================
# ASSEMBLY & OUTPUT
# ===================================================================
def assemble_output(df):
    """
    Build the final CSV with model features + eda reference.

    IMPORTANT: This outputs RAW engineered features without log transforms,
    outlier capping, or scaling. Those treatments are MODEL-SPECIFIC:
      - RF / XGBoost / LightGBM: use raw features (trees are robust)
      - MLP: apply log1p on skewed cols → cap at p99 → StandardScaler
    The modeling script handles this, not the feature CSV.

    What IS applied here:
      - Drop years_since_release (perfectly correlated with release_year, -1.0)
      - Fill NaN in release_year with median (games with missing dates)
      - Fill NaN in developer/publisher track records with base rate
    """
    id_cols = ["appid", "name"]
    target = "success"

    # Fix: drop years_since_release (redundant with release_year, corr = -1.0)
    # Keeping release_year because it's more interpretable
    model_base = [
        "price","is_free","price_tier","required_age","achievements","has_achievements",
        "platform_count","language_count","audio_language_count",
        "release_year","release_month","release_day_of_week","release_quarter",
        "released_in_sale_month",
        "description_length","has_website",
        "developer_historical_success","publisher_historical_success",
        "developer_game_count",
        "genre_count","has_genres","category_count","has_categories",
        "tag_count","has_tags",
    ]
    dynamic = sorted(c for c in df.columns
                     if c.startswith(("genre_","cat_","tag_")) and c not in model_base)
    eda_cols = sorted(c for c in df.columns if c.startswith("eda_"))

    model_features = [c for c in model_base + dynamic if c in df.columns]
    all_cols = id_cols + [target] + model_features + eda_cols
    output = df[all_cols].copy()

    # Fix edge cases: fill NaNs in numeric model features
    for col in model_features:
        if output[col].dtype in ["float64", "int64", "Float64", "Int64"]:
            null_count = output[col].isnull().sum()
            if null_count > 0:
                median_val = output[col].median()
                output[col] = output[col].fillna(median_val)
                print(f"    Filled {null_count:,} nulls in {col} with median ({median_val:.2f})")

    return output, model_features, eda_cols


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("STEAM GAMES — EDA & FEATURE ENGINEERING")
    print("=" * 60)

    print("\n[1/8] Loading dataset ...")
    df = pd.read_csv(SOURCE_FILE, low_memory=False)
    print(f"  {len(df):,} games, {len(df.columns)} columns")

    print("\n[2/8] Feature engineering ...")
    df = engineer_features(df)

    print("\n[3/8] Assembling output CSV ...")
    output, model_features, eda_cols = assemble_output(df)
    output.to_csv(OUTPUT_CSV, index=False)
    print(f"  Model features: {len(model_features)} | EDA reference: {len(eda_cols)}")
    print(f"  Saved → {OUTPUT_CSV}")

    print("\n[4/8] Running EDA analyses ...")
    CHARTS_DIR.mkdir(exist_ok=True)

    print("    Section 2: Target analysis")
    balance = eda_target(df, CHARTS_DIR)

    print("    Section 3: Feature distributions")
    skewness = eda_distributions(df, CHARTS_DIR)

    print("    Section 4: Class separability")
    separability = eda_separability(df, CHARTS_DIR)

    print("    Section 5: Multicollinearity")
    high_corr = eda_multicollinearity(df, CHARTS_DIR)

    print("    Section 6: Category coverage & genre analysis")
    cardinality = eda_cardinality(df, CHARTS_DIR)

    print("    Section 7: Outlier analysis")
    outliers = eda_outliers(df, CHARTS_DIR)

    print("    Section 8: Temporal patterns")
    temporal = eda_temporal(df, CHARTS_DIR)

    print("    Section 9: Leakage proof")
    leakage = eda_leakage_proof(df, CHARTS_DIR)

    print("    Section 10: Edge cases")
    edge_cases = eda_edge_cases(df)

    print("    Section 11: Success breakdowns")
    eda_success_breakdowns(df, CHARTS_DIR)

    print("\n[5/8] Generating insights report ...")
    report = generate_report(df, balance, skewness, separability, high_corr,
                             cardinality, outliers, temporal, leakage, edge_cases,
                             model_features, eda_cols)
    OUTPUT_REPORT.write_text(report)
    print(f"  Saved → {OUTPUT_REPORT}")

    print("\n[6/8] Sanity check ...")
    nulls = output[model_features].isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(nulls) == 0:
        print("  Model features: no nulls")
    else:
        for col, n in nulls.head(5).items():
            print(f"  {col}: {n:,} nulls")

    print("\n[7/8] Summary")
    print(f"  Output CSV: {output.shape[1]} columns, {output.shape[0]} rows")
    print(f"  Charts: {len(list(CHARTS_DIR.glob('*.png')))} saved to EDA/charts/")

    print("\n[8/8] Done.")


if __name__ == "__main__":
    main()