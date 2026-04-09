from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 42
N_SPLITS = 5
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def parse_listish(value: object) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def safe_text_len(value: object) -> int:
    if pd.isna(value):
        return 0
    return len(str(value))


def add_multi_hot_features(
    feature_df: pd.DataFrame,
    source_series: pd.Series,
    prefix: str,
    top_n: int,
) -> pd.DataFrame:
    parsed = source_series.apply(parse_listish)
    top_labels = (
        parsed.explode()
        .dropna()
        .astype(str)
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    for label in top_labels:
        column = f"{prefix}_{label.lower().replace(' ', '_').replace('-', '_').replace('&', 'and')}"
        feature_df[column] = parsed.apply(lambda items: int(label in items))
    return feature_df


def build_engineered_dataset(raw_path: Path, engineered_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, low_memory=False)

    engineered = pd.DataFrame(index=df.index)
    engineered["appid"] = df["appid"]
    engineered["name"] = df["name"]
    engineered["success"] = df["success"].astype(int)

    release_dates = pd.to_datetime(df["release_date"], errors="coerce")
    reference_date = pd.Timestamp("2025-03-31")

    engineered["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    engineered["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0)
    engineered["dlc_count"] = pd.to_numeric(df["dlc_count"], errors="coerce").fillna(0)
    engineered["achievements"] = pd.to_numeric(df["achievements"], errors="coerce").fillna(0)
    engineered["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0)
    engineered["windows"] = df["windows"].astype(int)
    engineered["mac"] = df["mac"].astype(int)
    engineered["linux"] = df["linux"].astype(int)
    engineered["is_free"] = (engineered["price"] == 0).astype(int)
    engineered["has_dlc"] = (engineered["dlc_count"] > 0).astype(int)
    engineered["has_achievements"] = (engineered["achievements"] > 0).astype(int)
    engineered["release_year"] = release_dates.dt.year.fillna(release_dates.dt.year.median()).astype(int)
    engineered["release_month"] = release_dates.dt.month.fillna(0).astype(int)
    engineered["game_age_days"] = (
        reference_date - release_dates
    ).dt.days.fillna((reference_date - release_dates.dropna().median()).days).astype(int)
    engineered["log_price"] = np.log1p(engineered["price"])
    engineered["description_length"] = df["detailed_description"].apply(safe_text_len)
    engineered["about_length"] = df["about_the_game"].apply(safe_text_len)
    engineered["short_description_length"] = df["short_description"].apply(safe_text_len)

    for source_col, target_col in [
        ("supported_languages", "supported_language_count"),
        ("full_audio_languages", "full_audio_language_count"),
        ("developers", "developer_count"),
        ("publishers", "publisher_count"),
        ("categories", "category_count"),
        ("genres", "genre_count"),
    ]:
        engineered[target_col] = df[source_col].apply(lambda value: len(parse_listish(value)))

    price_bins = [-0.01, 0, 4.99, 9.99, 19.99, 29.99, 59.99, np.inf]
    price_labels = ["free", "0_01_4_99", "5_00_9_99", "10_00_19_99", "20_00_29_99", "30_00_59_99", "60_plus"]
    price_tiers = pd.cut(engineered["price"], bins=price_bins, labels=price_labels)
    price_dummies = pd.get_dummies(price_tiers, prefix="price_tier", dtype=int)
    engineered = pd.concat([engineered, price_dummies], axis=1)

    engineered = add_multi_hot_features(engineered, df["genres"], "genre", top_n=15)
    engineered = add_multi_hot_features(engineered, df["categories"], "category", top_n=15)

    numeric_cols = engineered.select_dtypes(include=["number", "bool"]).columns
    engineered[numeric_cols] = engineered[numeric_cols].apply(pd.to_numeric, errors="coerce")
    engineered[numeric_cols] = engineered[numeric_cols].fillna(0)

    engineered_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(engineered_path, index=False)
    return engineered


def ensure_engineered_dataset(
    engineered_path: Path,
    raw_fallback_path: Path,
    downloads_source_path: Path | None = None,
) -> pd.DataFrame:
    if engineered_path.exists():
        return pd.read_csv(engineered_path, low_memory=False)

    if downloads_source_path is not None and downloads_source_path.exists():
        engineered_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloads_source_path, engineered_path)
        print(f"Copied engineered dataset from {downloads_source_path} to {engineered_path}")
        return pd.read_csv(engineered_path, low_memory=False)

    if raw_fallback_path.exists():
        print(f"Input file missing at {engineered_path}. Building it from {raw_fallback_path.name}.")
        return build_engineered_dataset(raw_fallback_path, engineered_path)

    raise FileNotFoundError(
        f"Could not find engineered dataset at {engineered_path}, download source at "
        f"{downloads_source_path}, or fallback raw dataset at {raw_fallback_path}."
    )


def load_modeling_data(
    engineered_path: Path,
    raw_fallback_path: Path,
    downloads_source_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = ensure_engineered_dataset(engineered_path, raw_fallback_path, downloads_source_path)

    if not engineered_path.exists():
        print(f"Saved engineered dataset to {engineered_path}")

    if "success" not in df.columns:
        raise ValueError("Expected a 'success' column in the engineered dataset.")

    drop_cols = ["success", "appid", "name"]
    eda_cols = [col for col in df.columns if col.startswith("eda_")]
    X = df.drop(columns=drop_cols + eda_cols, errors="ignore").copy()
    y = df["success"].astype(int).copy()

    raw_dtype_counts = X.dtypes.astype(str).value_counts()
    object_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if object_cols:
        X = pd.get_dummies(X, columns=object_cols, dummy_na=False, dtype=int)

    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    print("Dataset validation summary")
    print(f"Source shape: {df.shape}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Null values in X: {int(X.isna().sum().sum())}")
    print("Class balance:")
    print(y.value_counts().sort_index().to_string())
    print("Raw X dtype counts before encoding:")
    print(raw_dtype_counts.to_string())
    print("Model-ready X dtype counts after encoding:")
    print(X.dtypes.astype(str).value_counts().to_string())

    return X, y, df


def evaluate_random_forest(X: pd.DataFrame, y: pd.Series, modeling_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics_by_fold: list[dict[str, float]] = []
    roc_curves: list[tuple[np.ndarray, np.ndarray, float]] = []
    confusion_total = np.zeros((2, 2), dtype=int)
    last_model = None

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves.append((fpr, tpr, auc(fpr, tpr)))
        confusion_total += confusion_matrix(y_test, y_pred, labels=[0, 1])

        metrics_by_fold.append(
            {
                "fold": fold,
                "roc_auc": roc_auc_score(y_test, y_prob),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_test, y_pred),
            }
        )
        last_model = model

    metrics_df = pd.DataFrame(metrics_by_fold)
    metrics_df.to_csv(modeling_dir / "rf_baseline_cv_metrics.csv", index=False)

    summary = {
        metric: metrics_df[metric].mean()
        for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
    }
    summary.update(
        {
            f"{metric}_std": metrics_df[metric].std(ddof=1)
            for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
        }
    )

    plot_roc_curves(roc_curves, modeling_dir / "rf_baseline_roc_curve.png")
    plot_confusion_matrix(confusion_total, modeling_dir / "rf_baseline_confusion_matrix.png")
    if last_model is not None:
        plot_feature_importance(
            last_model.feature_importances_,
            X.columns.tolist(),
            modeling_dir / "rf_baseline_feature_importance.png",
        )

    return metrics_df, summary


def plot_roc_curves(roc_curves: list[tuple[np.ndarray, np.ndarray, float]], output_path: Path) -> None:
    mean_fpr = np.linspace(0, 1, 200)
    interpolated_tprs = []

    plt.figure(figsize=(9, 7))
    for idx, (fpr, tpr, fold_auc) in enumerate(roc_curves, start=1):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)
        plt.plot(fpr, tpr, alpha=0.35, label=f"Fold {idx} (AUC={fold_auc:.3f})")

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    plt.plot(mean_fpr, mean_tpr, color="black", linewidth=2.5, label=f"Mean ROC (AUC={mean_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest Baseline ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(confusion_total: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_total, display_labels=["Fail", "Success"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Random Forest Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], output_path: Path) -> None:
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    top_features = importance_df.sort_values("importance", ascending=False).head(20).sort_values("importance")

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#2c7fb8")
    plt.xlabel("Impurity Importance")
    plt.title("Random Forest Baseline Top 20 Features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    top_features.sort_values("importance", ascending=False).to_csv(
        output_path.with_suffix(".csv"), index=False
    )


def write_results_report(
    modeling_dir: Path,
    engineered_path: Path,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_df: pd.DataFrame,
    summary: dict[str, float],
) -> None:
    top_features = pd.read_csv(modeling_dir / "rf_baseline_feature_importance.csv").head(10)
    top_feature_text = ", ".join(top_features["feature"].head(5).tolist())

    report = f"""# Random Forest Baseline Results

The baseline model used `RandomForestClassifier` with `n_estimators=300`, `max_depth=None`, `class_weight='balanced'`, and `random_state=42`. Evaluation used stratified 5-fold cross-validation on the engineered dataset at `data/output/steam_features_engineered.csv`, with ID columns (`appid`, `name`) and any `eda_` prefixed columns removed before training. The final modeling matrix contained {X.shape[0]:,} rows and {X.shape[1]:,} usable features, with class balance of {int((y == 0).sum()):,} unsuccessful games versus {int((y == 1).sum()):,} successful games.

Across folds, the baseline achieved ROC-AUC of {summary['roc_auc']:.3f} +/- {summary['roc_auc_std']:.3f}, F1 of {summary['f1']:.3f} +/- {summary['f1_std']:.3f}, precision of {summary['precision']:.3f} +/- {summary['precision_std']:.3f}, recall of {summary['recall']:.3f} +/- {summary['recall_std']:.3f}, and accuracy of {summary['accuracy']:.3f} +/- {summary['accuracy_std']:.3f}. Because only about 16% of games are labeled successful, using `class_weight='balanced'` was important for keeping the model from collapsing into an almost-all-negative classifier. The resulting recall and F1 are more meaningful than raw accuracy alone for this project.

The strongest impurity-based feature importance signals came from {top_feature_text}. These are useful baseline signals, but they should still be interpreted cautiously because impurity importance can favor higher-cardinality or more frequently splitting variables. This run establishes the floor for later comparisons, while Optuna tuning and stronger boosted models can be evaluated afterward against the same cross-validation setup.
"""

    (modeling_dir / "rf_baseline_results.md").write_text(report)

    payload = {
        "engineered_dataset": str(engineered_path),
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "metrics_mean_std": {
            metric: {
                "mean": round(float(summary[metric]), 6),
                "std": round(float(summary[f"{metric}_std"]), 6),
            }
            for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]
        },
    }
    (modeling_dir / "rf_baseline_summary.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    modeling_dir = repo_root / "Modeling"
    data_output_dir = repo_root / "data" / "output"
    engineered_path = data_output_dir / "steam_features_engineered.csv"
    raw_fallback_path = repo_root / "games_march2025_cleaned_binary_success.csv"
    downloads_source_path = Path("/Users/laasyavenugopal/Downloads/steam_features_engineered.csv")

    modeling_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_modeling_data(engineered_path, raw_fallback_path, downloads_source_path)
    metrics_df, summary = evaluate_random_forest(X, y, modeling_dir)
    write_results_report(modeling_dir, engineered_path, X, y, metrics_df, summary)

    print("")
    print("Cross-validation results")
    for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        print(f"{metric}: {summary[metric]:.4f} +/- {summary[f'{metric}_std']:.4f}")
    print("")
    print(f"Saved metrics to {modeling_dir / 'rf_baseline_cv_metrics.csv'}")
    print(f"Saved report to {modeling_dir / 'rf_baseline_results.md'}")
    print(f"Saved plots to {modeling_dir}")


if __name__ == "__main__":
    main()
