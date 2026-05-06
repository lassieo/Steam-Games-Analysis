# Steam Games Success Prediction

**Predicting Commercial Success of Games on Steam: A Data-Driven Machine Learning Approach**

Applied Machine Learning Group Project · DSBA 6156
Team: Laasya Venugopal, Pujitha Attuluri, Kanniese Chen

**[Live App](https://steam-games-analysis-dsba6156.streamlit.app)**·  **[Dataset](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)**

---

## Overview

This project analyzes 89,618 Steam games to predict commercial success using only pre-launch features. We define a game as **successful** if it has at least 80% positive reviews AND at least 50 total reviews which is a definition that captures both quality (sentiment) and reach (engagement), yielding a 16.3% positive class rate.

We engineered 72 features from the raw dataset and trained six models in a progressive complexity ladder: Random Forest → XGBoost → LightGBM → Tabular MLP → Optuna-tuned XGBoost → Stacking Ensemble. The best model (stacking ensemble) achieved **ROC-AUC 0.8844**, with the tuned XGBoost at 0.8841 , effectively identical. The deployed app uses tuned XGBoost for cleaner SHAP interpretability, revealing that developer track record, community engagement (tags), and pricing are the strongest predictors of success.

## Key Results

| Model | ROC-AUC | F1 |
|---|---|---|
| Stacking Ensemble | 0.8844 | 0.568 |
| XGBoost (tuned) | 0.8841 | 0.576 |
| LightGBM | 0.8799 | 0.556 |
| Random Forest | 0.8792 | 0.434 |
| XGBoost (untuned) | 0.8748 | 0.564 |
| Tabular MLP | 0.8724 | 0.488 |

## Repository Structure

```
Steam-Games-Analysis/
├── data/
│   ├── input/                              # Raw dataset (download from Kaggle)
│   └── output/
│       └── steam_features_engineered.csv   # Engineered features (Git LFS)
├── Success Definition/
│   └── steam_success_analysis.py           # Success metric definition
├── Strategy Validation/
│   └── strategy_validation_analysis.py     # Feature audit and risk analysis
├── EDA/
│   ├── steam_eda_feature_engineering.py    # EDA + feature engineering pipeline
│   └── charts/                             # EDA visualizations
├── Hypotheses_EDA_ModelPlan/
│   └── week2_eda_analysis.py               # Hypothesis-driven EDA
├── Modeling/
│   ├── _modeling_utils.py                  # Shared CV splits and data loading
│   ├── rf_baseline.py                      # Random Forest baseline
│   ├── xgb_lgbm_training.py               # XGBoost and LightGBM
│   ├── mlp_training.py                     # Tabular MLP with preprocessing
│   ├── optuna_tuning.py                    # Bayesian hyperparameter tuning
│   ├── stacking_ensemble.py                # Stacking ensemble with LR meta-learner
│   ├── shap_analysis.py                    # SHAP interpretation + hypothesis validation
│   ├── model_comparison.py                 # Final comparison table generator
│   ├── imbalance_experiment.py             # Class imbalance strategy comparison
│   ├── time_split_experiment.py            # Temporal drift validation
│   ├── final_model_comparison.csv          # All model metrics
│   └── charts/                             # Model performance visualizations
├── streamlit_app/
│   ├── app.py                              # Main Streamlit entry point
│   ├── app_utils.py                        # Shared data/model loaders
│   ├── predictor.py                        # Game success predictor with SHAP
│   ├── chatbot.py                          # Data-grounded LLM chatbot (Groq)
│   ├── dashboard.py                        # Interactive EDA dashboard
│   ├── export_model.py                     # Model export for deployment
│   ├── requirements.txt                    # Python dependencies
│   ├── models/                             # Trained model artifacts
│   └── .streamlit/
│       └── secrets.toml.template           # API key template (never commit real keys)
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- pip or conda

### 1. Clone the repository

```bash
git clone https://github.com/lassieo/Steam-Games-Analysis.git
cd Steam-Games-Analysis
```

### 2. Install Git LFS and pull data

The engineered CSV (~22MB compressed) is stored via Git LFS:

```bash
git lfs install
git lfs pull
```

If LFS is not available, download the raw dataset from [Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset), place it in `data/input/`, and run the pipeline from Step 4.

### 3. Install dependencies

```bash
pip install -r streamlit_app/requirements.txt
pip install lightgbm optuna imbalanced-learn
```

### 4. Run the pipeline (optional — regenerate everything from scratch)

```bash
# Step 1: Generate success labels
python "Success Definition/steam_success_analysis.py"

# Step 2: Validate features and risks
python "Strategy Validation/strategy_validation_analysis.py"

# Step 3: EDA + feature engineering (generates steam_features_engineered.csv)
python "EDA/steam_eda_feature_engineering.py"

# Step 4: Run models (in order)
python Modeling/rf_baseline.py
python Modeling/xgb_lgbm_training.py
python Modeling/imbalance_experiment.py
python Modeling/time_split_experiment.py
python Modeling/mlp_training.py
python Modeling/optuna_tuning.py
python Modeling/stacking_ensemble.py
python Modeling/shap_analysis.py
python Modeling/model_comparison.py

# Step 5: Export model for the Streamlit app
python streamlit_app/export_model.py
```

### 5. Run the Streamlit app locally

```bash
cd streamlit_app
streamlit run app.py
```

For the chatbot to work, create `.streamlit/secrets.toml` with your Groq API key:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com).

## Streamlit App Features

The deployed app has three tabs:

**Predictor** — Enter game attributes (or use preset scenarios like AAA Blockbuster, Indie Gem, Free-to-Play, First-Time Developer) and get an instant success probability with a SHAP waterfall chart explaining which features drove the prediction. Uses the 16.3% base rate as the decision threshold for statistically honest classification.

**Ask the Data** — Natural language Q&A powered by Groq's Llama 3.3 70B. Every answer is grounded in real dataset queries — the chatbot never invents numbers. Supports aggregate statistics, individual game lookups, top-N recommendations with genre/year/price filters, and project context questions.

**Dashboard** — Interactive EDA with sidebar filters for year range, price tier, genre, and success status. Includes the actual SHAP summary plot, model comparison chart reading from real CV metrics, and hypothesis validation cards with expandable evidence.

## Technical Decisions

- **ROC-AUC as primary metric** — threshold-invariant and robust to the 16.3% class imbalance
- **Base rate (16.3%) as prediction threshold** — more honest than the default 0.5 for imbalanced data
- **scale_pos_weight for tree models** — outperformed SMOTE in our imbalance experiment
- **MLP preprocessing per fold** — log transforms, outlier capping, and StandardScaler fit only on training data
- **Tuned XGBoost in the app** — stacking ensemble beats it by only 0.0003 ROC-AUC; single model gives cleaner SHAP
- **Price tiers** — Budget (<$15), Mid ($15–50), Premium ($50–85), AAA ($85+)

## Hypotheses Validated by SHAP

| Hypothesis | Verdict | Key Evidence |
|---|---|---|
| H1: Paid > Free | Strongly supported | price ranked #6 in SHAP |
| H2: Price tier > raw price |  Partially reversed | Raw price (#6) beat price_tier (#19) |
| H3: Developer track record | Strongest finding | 3 features in SHAP top 15 |
| H4: Genre matters | Supported | genre_action ranked #13 |
| H5: Multi-platform helps | Supported | platform_count ranked #11 |

## Dataset

Source: [Steam Games Dataset 2025](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset) by artermiloff on Kaggle  
Size: 89,618 games · 51 raw columns · 72 engineered features  
Time range: 1997–2025 (meaningful volume from 2013 onward)
