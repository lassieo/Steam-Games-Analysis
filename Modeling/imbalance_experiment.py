# =========================
# Imbalance Experiment
# =========================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -------------------------
# Load Data
# -------------------------
DATA_PATH = "steam_features_engineered.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[c for c in df.columns if c.startswith('eda_')] + ['appid', 'name', 'success'])
y = df['success']

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# Evaluation Function
# -------------------------
def run_experiment(mode):
    roc_list, f1_list = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply SMOTE only to training
        if mode == "smote":
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if mode == "weighted":
            model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
        else:
            model = XGBClassifier(random_state=42, eval_metric='logloss')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        roc_list.append(roc_auc_score(y_val, y_prob))
        f1_list.append(f1_score(y_val, y_pred))

    return np.mean(roc_list), np.mean(f1_list)

# -------------------------
# Run Experiments
# -------------------------
results = {}

for mode in ["none", "weighted", "smote"]:
    roc, f1 = run_experiment(mode)
    results[mode] = (roc, f1)

# -------------------------
# Print Table
# -------------------------
print("\nImbalance Comparison:")
for k, v in results.items():
    print(f"{k}: ROC-AUC={v[0]:.4f}, F1={v[1]:.4f}")