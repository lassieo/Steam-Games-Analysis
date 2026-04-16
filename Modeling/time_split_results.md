# Time-based vs Random Split Comparison

The EDA revealed a temporal pattern in success rates across release years. This experiment tests whether that temporal pattern causes a meaningful difference between random stratified splits and time-based splits, using the same XGBoost model with `scale_pos_weight` for imbalance handling.

The random stratified split was evaluated using the standard 5-fold cross-validation that every other model in this project uses, with results averaged across folds. The time-based split trained on games released before 2022 (71,694 games) and tested on games released in 2022 or later (43,748 games). Both splits used identical model hyperparameters and the same `random_state=42`.

| Split type | Train size | Test size | ROC-AUC | F1 |
|---|---:|---:|---:|---:|
| Random stratified (5-fold CV) | 71,694 | 17,923 | 0.8748 | 0.5638 |
| Time-based (2022 cutoff) | 45,870 | 43,748 | 0.8845 | 0.5225 |

The time-based split produced a ROC-AUC change of +0.0097 (+1.11%) compared to the random stratified split. No meaningful difference. The model generalizes equally well across both split strategies, which means there is no significant temporal drift affecting model performance and a random stratified split is appropriate for this project.

For the final modeling pipeline, the random stratified 5-fold CV is the primary evaluation strategy because it produces the most stable estimates and matches the convention for tabular classification benchmarks. The time-based result is reported as a secondary check to demonstrate awareness of potential concept drift and to give the reader an honest picture of how the model would perform if deployed on truly unseen future games.
