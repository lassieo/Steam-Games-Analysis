# Class Imbalance Experiment Results

The Steam dataset has a 16.3% positive class (14,617 successful games out of 89,618 total). This experiment compares three strategies for handling that imbalance, all using XGBoost as the base model and the same stratified 5-fold cross-validation as every other model in the project.

The three strategies tested were: no balancing (let XGBoost train on the raw distribution), weighted (use `scale_pos_weight` to penalize minority class errors more heavily), and SMOTE (synthetic minority oversampling applied only to the training fold within each CV iteration). The SMOTE application is critical: synthetic samples must never appear in the validation fold, or the metrics become inflated by leakage. This implementation applies SMOTE inside the CV loop, on the training data only, generating new training data fresh for each fold.

| Strategy | ROC-AUC | F1 |
|---|---|---|
| None | 0.8776 +/- 0.0027 | 0.5160 +/- 0.0067 |
| Weighted | 0.8748 +/- 0.0027 | 0.5638 +/- 0.0082 |
| Smote | 0.8642 +/- 0.0008 | 0.5264 +/- 0.0061 |

The best ROC-AUC came from the **none** strategy and the best F1 came from the **weighted** strategy. ROC-AUC is the primary metric for this project because it is invariant to the prediction threshold, but F1 is also informative because it directly reflects the precision-recall tradeoff at the default 0.5 threshold.

The general pattern in tabular classification with moderate imbalance (10-20% minority class) is that `scale_pos_weight` typically matches or beats SMOTE while being far simpler and faster. SMOTE can help in extreme imbalance situations (under 5% minority class) but tends to introduce noise when there are already plenty of minority examples to learn from. The recommendation for this project is to use `scale_pos_weight` in tree-based models because it is computationally cheap, mathematically clean, and produces results competitive with more expensive resampling methods.
