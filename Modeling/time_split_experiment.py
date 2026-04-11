# =========================
# Time-Based Split Experiment
# =========================

import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# -------------------------
# Load Data
# -------------------------
DATA_PATH = "steam_features_engineered.csv"

df = pd.read_csv(DATA_PATH)

# -------------------------
# Split Data
# -------------------------
train_df = df[df['release_year'] < 2022]
test_df = df[df['release_year'] >= 2022]

X_train = train_df.drop(columns=[c for c in df.columns if c.startswith('eda_')] + ['appid', 'name', 'success'])
y_train = train_df['success']

X_test = test_df.drop(columns=[c for c in df.columns if c.startswith('eda_')] + ['appid', 'name', 'success'])
y_test = test_df['success']

# -------------------------
# Train Model
# -------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_prob = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_prob)

print(f"Time-based ROC-AUC: {roc:.4f}")