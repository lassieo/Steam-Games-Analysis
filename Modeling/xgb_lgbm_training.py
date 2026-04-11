# =========================
# XGBoost & LightGBM Training
# =========================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -------------------------
# Load Data
# -------------------------
# REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = "steam_features_engineered.csv"

df = pd.read_csv(DATA_PATH)

# -------------------------
# Clean Data
# -------------------------
X = df.drop(columns=[c for c in df.columns if c.startswith('eda_')] + ['appid', 'name', 'success'])
y = df['success']

print("Shape:", X.shape)
print("Class balance:\n", y.value_counts())

# -------------------------
# Compute imbalance weight
# -------------------------
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# -------------------------
# Cross Validation Setup
# -------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y):
    roc_list, f1_list, prec_list, rec_list = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        roc_list.append(roc_auc_score(y_val, y_prob))
        f1_list.append(f1_score(y_val, y_pred))
        prec_list.append(precision_score(y_val, y_pred))
        rec_list.append(recall_score(y_val, y_pred))

    return {
        "ROC-AUC": (np.mean(roc_list), np.std(roc_list)),
        "F1": (np.mean(f1_list), np.std(f1_list)),
        "Precision": (np.mean(prec_list), np.std(prec_list)),
        "Recall": (np.mean(rec_list), np.std(rec_list))
    }

# -------------------------
# Train XGBoost
# -------------------------
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_results = evaluate_model(xgb_model, X, y)

# -------------------------
# Train LightGBM
# -------------------------
lgbm_model = LGBMClassifier(
    is_unbalance=True,
    random_state=42
)

lgbm_results = evaluate_model(lgbm_model, X, y)

# -------------------------
# Print Results
# -------------------------
print("\nXGBoost Results:")
for k, v in xgb_results.items():
    print(f"{k}: {v[0]:.4f} ± {v[1]:.4f}")

print("\nLightGBM Results:")
for k, v in lgbm_results.items():
    print(f"{k}: {v[0]:.4f} ± {v[1]:.4f}")