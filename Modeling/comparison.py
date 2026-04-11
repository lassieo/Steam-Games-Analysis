import matplotlib.pyplot as plt
import numpy as np

models = ['RF', 'XGB', 'LGBM']

roc_scores = [0.83, 0.8748, 0.8799]
f1_scores  = [0.52, 0.5638, 0.5564]

x = np.arange(len(models))
width = 0.35

plt.figure()

plt.bar(x - width/2, roc_scores, width, label='ROC-AUC')
plt.bar(x + width/2, f1_scores, width, label='F1')

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Model Comparison (RF vs XGB vs LGBM)")
plt.legend()

plt.savefig("Modeling/model_comparison_bar.png")
plt.show()