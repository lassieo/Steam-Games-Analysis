"""
Optuna Hyperparameter Tuning
=============================

Tunes XGBoost hyperparameters using Bayesian optimization.

Why XGBoost: From Kanniese's results, XGBoost was the strongest tree model
(ROC-AUC 0.8748 vs LightGBM 0.8799 vs RF baseline 0.879). We tune the strongest
performer to push it further.

Why Optuna over GridSearchCV: GridSearchCV does exhaustive search, which is slow
and dumb. Optuna uses Tree-Structured Parzen Estimator (TPE) — it learns from
previous trials to focus on promising regions of the hyperparameter space.

Reads:  data/output/steam_features_engineered.csv
Writes: Modeling/optuna_best_params.json
        Modeling/optuna_trials.csv
        Modeling/optuna_results.md
        Modeling/charts/optuna_optimization_history.png
        Modeling/charts/optuna_param_importance.png

Requires: pip install optuna xgboost
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )
except ImportError:
    raise ImportError(
        "Optuna is required. Install with: pip install optuna"
    )

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "XGBoost is required. Install with: pip install xgboost"
    )

from _modeling_utils import (
    CHARTS_DIR,
    MODELING_DIR,
    RANDOM_STATE,
    ensure_dirs,
    get_cv_splitter,
    load_modeling_data,
)

# Number of Optuna trials. Each trial trains a 5-fold CV XGBoost.
# 50 trials is a reasonable balance — gets you most of the benefit without
# burning too much time. Bump to 100 if you have time to spare.
N_TRIALS = 50


# ---------------------------------------------------------------------------
# Optuna objective function
# ---------------------------------------------------------------------------
def make_objective(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float):
    """
    Creates the objective function for Optuna.

    Each trial samples a hyperparameter combination, runs 5-fold CV,
    and returns the mean ROC-AUC for Optuna to maximize.
    """

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        # These ranges are based on common XGBoost tuning advice for tabular data
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "tree_method": "hist",  # faster training
            "n_jobs": -1,
        }

        cv = get_cv_splitter()
        fold_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            fold_scores.append(roc_auc_score(y_val, y_prob))

        mean_auc = float(np.mean(fold_scores))
        return mean_auc

    return objective


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_optimization_history_chart(study: optuna.Study, output_path: Path) -> None:
    fig = plot_optimization_history(study).figure
    fig.set_size_inches(9, 5)
    fig.axes[0].set_title("Optuna Optimization History (XGBoost)")
    fig.axes[0].set_ylabel("ROC-AUC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_param_importance_chart(study: optuna.Study, output_path: Path) -> None:
    try:
        fig = plot_param_importances(study).figure
        fig.set_size_inches(9, 6)
        fig.axes[0].set_title("Hyperparameter Importance (Optuna)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"  Skipping param importance plot: {exc}")



def evaluate_best_params_full_metrics(study: optuna.Study, X: pd.DataFrame, y: pd.Series, scale_pos_weight: float) -> dict:
    """
    Re-run 5-fold CV with the best params and compute all metrics
    (Optuna only optimized ROC-AUC, so F1/precision/recall weren't tracked).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    best_params = study.best_params
    full_params = {
        **best_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
    }

    cv = get_cv_splitter()
    fold_metrics = {"roc_auc": [], "f1": [], "precision": [], "recall": [], "accuracy": []}

    print("\n  Re-evaluating best params to compute full metrics ...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**full_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
        fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        print(f"  Fold {fold}/5: ROC-AUC={fold_metrics['roc_auc'][-1]:.4f}, F1={fold_metrics['f1'][-1]:.4f}")

    summary = {}
    for metric, values in fold_metrics.items():
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1))

    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_results(study: optuna.Study, baseline_auc: float | None, full_metrics: dict) -> None:
    best_params = study.best_params
    best_score = study.best_value

    # Save best params + full metrics as JSON
    payload = {
        "model": "XGBClassifier",
        "n_trials": len(study.trials),
        "best_roc_auc": float(best_score),
        "best_params": best_params,
        "full_metrics": full_metrics,  # NEW: F1, precision, recall, accuracy
    }
    (MODELING_DIR / "optuna_best_params.json").write_text(json.dumps(payload, indent=2))

    # Save all trials as CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(MODELING_DIR / "optuna_trials.csv", index=False)

    # Markdown report
    params_text = "\n".join(f"- `{k}` = {v}" for k, v in best_params.items())
    improvement_text = ""
    if baseline_auc is not None:
        delta = best_score - baseline_auc
        pct = (delta / baseline_auc) * 100
        improvement_text = (
            f" Compared against the untuned XGBoost baseline ROC-AUC of {baseline_auc:.4f}, "
            f"this represents an improvement of {delta:+.4f} ({pct:+.2f}%)."
        )

    report = f"""# Optuna Hyperparameter Tuning Results

XGBoost was selected as the model to tune based on Kanniese's earlier comparison showing it as the strongest single tree model. Optuna's TPE sampler ran {len(study.trials)} trials, with each trial training a 5-fold stratified cross-validation using `random_state=42` (the same splits as every other model in this project). The objective was to maximize the mean ROC-AUC across folds.

The best trial achieved a ROC-AUC of **{best_score:.4f}**.{improvement_text}

The best hyperparameters found were:

{params_text}

These parameters are saved to `optuna_best_params.json` and will be used by the stacking ensemble script. The full trial history is saved to `optuna_trials.csv` for reference, and the optimization history and parameter importance plots are saved in the `charts/` folder. Note that even with extensive tuning, the improvement over the untuned baseline is typically modest for tabular data — most of the predictive power comes from the features and the model class, not from hyperparameter optimization.
"""
    (MODELING_DIR / "optuna_results.md").write_text(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("OPTUNA HYPERPARAMETER TUNING (XGBoost)")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/3] Loading data ...")
    X, y = load_modeling_data()

    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    print(f"  scale_pos_weight = {scale_pos_weight:.3f}")

    # Try to read Kanniese's untuned baseline for comparison
    baseline_auc = None
    xgb_metrics_path = MODELING_DIR / "xgb_lgbm_cv_metrics.csv"
    if xgb_metrics_path.exists():
        try:
            xgb_df = pd.read_csv(xgb_metrics_path)
            if "xgb_roc_auc" in xgb_df.columns:
                baseline_auc = float(xgb_df["xgb_roc_auc"].mean())
                print(f"  Found untuned XGBoost baseline ROC-AUC: {baseline_auc:.4f}")
        except Exception:
            pass

    print(f"\n[2/3] Running Optuna with {N_TRIALS} trials ...")
    print("  (Each trial = 5-fold CV. This will take a while.)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="xgboost_steam_success",
    )

    objective = make_objective(X, y, scale_pos_weight)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number + 1:3d}/{N_TRIALS}: ROC-AUC={trial.value:.4f} "
              f"(best so far: {study.best_value:.4f})")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback], show_progress_bar=False)

    print("\n[3/3] Computing full metrics for the best params and saving results ...")
    full_metrics = evaluate_best_params_full_metrics(study, X, y, scale_pos_weight)
    plot_optimization_history_chart(study, CHARTS_DIR / "optuna_optimization_history.png")
    plot_param_importance_chart(study, CHARTS_DIR / "optuna_param_importance.png")
    write_results(study, baseline_auc, full_metrics)

    print("\nFull metrics for tuned XGBoost:")
    for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
        print(f"  {metric:12s}: {full_metrics[f'{metric}_mean']:.4f} +/- {full_metrics[f'{metric}_std']:.4f}")

    print(f"\nBest ROC-AUC: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nSaved best params to {MODELING_DIR / 'optuna_best_params.json'}")
    print(f"Saved trials to {MODELING_DIR / 'optuna_trials.csv'}")
    print(f"Saved report to {MODELING_DIR / 'optuna_results.md'}")
    print("\nDone.")


if __name__ == "__main__":
    main()