# Steam Games Analysis

This project explores what makes a game “successful” on Steam using real-world data like user reviews and engagement patterns.

The goal is to take a messy dataset, clean it, define a meaningful success metric, and build a strong foundation for future machine learning models.

---

## Project Goal

**What actually makes a Steam game successful?**

Instead of guessing, we used data to define success in a way that reflects both:
- how much players like a game  
- and how many people have actually played/reviewed it  

---

## How We Define Success

After testing multiple approaches, we defined a game as **successful** if:

- at least **80% of its reviews are positive**, and  
- it has at least **50 total reviews**

In code:

```python
success = 1 if positive_ratio_calc >= 0.80 and review_total_calc >= 50 else 0
```

---


## Setup — Getting the Data

The CSV files are too large for GitHub (~450MB each). Download them from Google Drive and place them in the repo:

**Google Drive link:** *https://drive.google.com/drive/folders/1EzG--wNHYlTRENYVGS7OOsHMOcTxxiUB?usp=sharing*

After downloading, place the files as follows:
```
data/
├── input/
│   └── games_march2025_cleaned.csv          ← download this
└── output/
    └── games_march2025_with_success.csv     ← download this (or generate with Step 1)
```

All scripts use **relative paths** from the repo root — no hardcoded desktop paths. Once the data files are in place, everything runs on any machine without editing paths.

---

## How to Run

Run from the repo root, in order:
```bash
# Step 1: Generate success labels (reads data/input/, writes data/output/)
python "Success Definition/steam_success_analysis.py"

# Step 2: Validate features and risks (reads data/output/, writes reports)
python "Strategy Validation/strategy_validation_analysis.py"

# Step 3: EDA + feature engineering (reads data/output/, writes data/output/ + charts)
python "EDA/steam_eda_feature_engineering.py"
```

**Requirements:** Python 3.9+, pandas, numpy, matplotlib, seaborn
