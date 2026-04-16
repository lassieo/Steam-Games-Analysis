# Final Model Comparison

This is the complete model comparison across all five tiers of the model ladder. Every model was trained on the same engineered dataset (`data/output/steam_features_engineered.csv`) using identical stratified 5-fold cross-validation splits with `random_state=42`. This guarantees that the metrics are directly comparable — any difference in scores reflects real differences in model capability, not differences in data preprocessing or evaluation methodology.

## Results

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Stacking Ensemble | 0.8844 +/- 0.0025 | 0.5683 | 0.4436 | 0.7904 | 0.8041 |
| XGBoost (tuned) | 0.8841 +/- 0.0024 | 0.5756 | 0.4592 | 0.7712 | 0.8145 |
| LightGBM | 0.8799 +/- 0.0029 | 0.5564 | 0.4250 | 0.8052 | 0.7906 |
| Random Forest | 0.8792 +/- 0.0028 | 0.4337 | 0.7119 | 0.3120 | 0.8672 |
| XGBoost | 0.8748 +/- 0.0027 | 0.5638 | 0.4474 | 0.7620 | 0.8077 |
| Tabular MLP | 0.8724 +/- 0.0032 | 0.4879 | 0.6514 | 0.3909 | 0.8664 |

The best-performing model by ROC-AUC was **Stacking Ensemble** with a score of **0.8844**.

## Interpretation

The progression from Random Forest baseline through tuned XGBoost to the stacking ensemble shows the typical pattern for tabular classification: most of the gain comes from moving to gradient boosting (XGBoost or LightGBM), with smaller gains from hyperparameter tuning, and incremental gains from ensembling. The Tabular MLP is included to address the professor's feedback about exhausting model options — neural networks rarely beat gradient boosting on tabular data, but trying it shows we did not stop at the first reasonable result.

The metrics in this table are the basis for the SHAP analysis (which uses the best-performing single model) and for the final hypothesis validation. Note that even the highest-performing model has a ceiling determined by the inherent predictability of the task: predicting which Steam games will be successful is genuinely hard, because much of what makes a game succeed depends on factors that cannot be measured before launch (community reception, marketing impact, timing relative to other releases). The model can only learn from features that exist in the data, and a ROC-AUC in the high 0.80s is a strong result for this kind of problem.
