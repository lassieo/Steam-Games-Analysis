# Optuna Hyperparameter Tuning Results

XGBoost was selected as the model to tune based on Kanniese's earlier comparison showing it as the strongest single tree model. Optuna's TPE sampler ran 50 trials, with each trial training a 5-fold stratified cross-validation using `random_state=42` (the same splits as every other model in this project). The objective was to maximize the mean ROC-AUC across folds.

The best trial achieved a ROC-AUC of **0.8841**. Compared against the untuned XGBoost baseline ROC-AUC of 0.8748, this represents an improvement of +0.0093 (+1.06%).

The best hyperparameters found were:

- `n_estimators` = 600
- `max_depth` = 10
- `learning_rate` = 0.013722315372926544
- `subsample` = 0.7417033228523667
- `colsample_bytree` = 0.6330518651571728
- `min_child_weight` = 10
- `reg_alpha` = 0.004293015645912328
- `reg_lambda` = 7.933219499760244e-08
- `gamma` = 8.275580538169897e-07

These parameters are saved to `optuna_best_params.json` and will be used by the stacking ensemble script. The full trial history is saved to `optuna_trials.csv` for reference, and the optimization history and parameter importance plots are saved in the `charts/` folder. Note that even with extensive tuning, the improvement over the untuned baseline is typically modest for tabular data — most of the predictive power comes from the features and the model class, not from hyperparameter optimization.
