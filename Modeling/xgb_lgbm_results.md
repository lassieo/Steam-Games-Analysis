# XGBoost and LightGBM Results

The two gradient boosting models were trained on the same engineered dataset (`data/output/steam_features_engineered.csv`) as the Random Forest baseline, using identical stratified 5-fold cross-validation splits with `random_state=42`. Both models received imbalance handling: XGBoost via `scale_pos_weight = (n_neg / n_pos)`, and LightGBM via `is_unbalance=True`. No hyperparameter tuning was applied at this stage — the goal here is to establish the strongest untuned tree model before passing it to Optuna.

The final modeling matrix contained 89,618 rows and 72 usable features, with class balance of 75,001 unsuccessful games versus 14,617 successful games.

**XGBoost results:** ROC-AUC of 0.875 +/- 0.003, F1 of 0.564 +/- 0.008, precision of 0.447 +/- 0.007, recall of 0.762 +/- 0.009, accuracy of 0.808 +/- 0.004.

**LightGBM results:** ROC-AUC of 0.880 +/- 0.003, F1 of 0.556 +/- 0.006, precision of 0.425 +/- 0.005, recall of 0.805 +/- 0.008, accuracy of 0.791 +/- 0.003.

LightGBM performed slightly better and will be the model passed to Optuna hyperparameter tuning in the next stage. Both models substantially outperform what would be expected from random guessing (ROC-AUC of 0.5) and demonstrate that the engineered features carry meaningful predictive signal. The next step is to tune the stronger model with Bayesian optimization to extract additional performance, then combine all base models in a stacking ensemble.
