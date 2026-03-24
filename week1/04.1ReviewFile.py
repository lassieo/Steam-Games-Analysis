import pandas as pd

# =========================
# 1. LOAD SOURCE DATA
# =========================
games = pd.read_csv("dataset2cleaned/march2025_cleaned_v2.csv")

# =========================
# 2. STANDARDIZE COLUMNS
# =========================
games.columns = (
    games.columns
    .str.strip()
    .str.lower()
)

# =========================
# 3. SELECT REVIEW COLUMNS
# =========================
review_cols = [
    "appid",
    "num_reviews_total",
    "num_reviews_recent",
    "pct_pos_total",
    "pct_pos_recent",
    "positive",
    "negative"
]

reviews = games[review_cols].copy()

# =========================
# 4. CLEAN DATA TYPES
# =========================
reviews["appid"] = reviews["appid"].astype(str)

num_cols = [
    "num_reviews_total",
    "num_reviews_recent",
    "pct_pos_total",
    "pct_pos_recent",
    "positive",
    "negative"
]

for col in num_cols:
    reviews[col] = pd.to_numeric(reviews[col], errors="coerce")

# =========================
# 5. REMOVE DUPLICATES
# =========================
reviews = reviews.drop_duplicates(subset=["appid"])

# =========================
# 6. FINAL CHECK
# =========================
print("Shape:", reviews.shape)
print("Missing:\n", reviews.isnull().sum())

# =========================
# 7. SAVE FILE
# =========================
reviews.to_csv("reviews/march2025_steam_reviews.csv", index=False)

print("\n✅ steam_reviews.csv created successfully!")