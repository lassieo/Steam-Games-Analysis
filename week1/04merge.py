import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
games_2025 = pd.read_csv("dataset2/games_march2025_cleaned.csv")
reviews_2025 = pd.read_csv("reviews/march2025_steam_reviews.csv")

games_2024 = pd.read_csv("dataset2/games_may2024_cleaned.csv")
reviews_2024 = pd.read_csv("reviews/may2024_steam_reviews.csv")

# =========================
# 2. STANDARDIZE COLUMNS
# =========================
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )
    return df

games_2025 = clean_columns(games_2025)
reviews_2025 = clean_columns(reviews_2025)

games_2024 = clean_columns(games_2024)
reviews_2024 = clean_columns(reviews_2024)

# =========================
# 3. FIX KEY COLUMN
# =========================
# Some datasets use AppID instead of appid

games_2024.columns = games_2024.columns.str.lower()
reviews_2024.columns = reviews_2024.columns.str.lower()

# Ensure string type
for df in [games_2025, reviews_2025, games_2024, reviews_2024]:
    df["appid"] = df["appid"].astype(str)

# =========================
# 4. CHECK OVERLAP
# =========================
def check_overlap(games, reviews, label):
    overlap = set(games["appid"]) & set(reviews["appid"])
    print(f"{label} overlap:", len(overlap))

check_overlap(games_2025, reviews_2025, "2025")
check_overlap(games_2024, reviews_2024, "2024")

# =========================
# 5. MERGE (CORRECT JOIN)
# =========================
# LEFT JOIN = keep all games

merged_2025 = games_2025.merge(
    reviews_2025,
    on="appid",
    how="left"
)

merged_2024 = games_2024.merge(
    reviews_2024,
    on="appid",
    how="left"
)

# =========================
# 6. ADD SNAPSHOT LABEL
# =========================
merged_2025["snapshot"] = "2025-03"
merged_2024["snapshot"] = "2024-05"

# =========================
# 7. CHECK DUPLICATES
# =========================
print("2025 duplicates:", merged_2025.duplicated(subset=["appid"]).sum())
print("2024 duplicates:", merged_2024.duplicated(subset=["appid"]).sum())

# =========================
# 8. COMBINE DATASETS
# =========================
merged_all = pd.concat([merged_2024, merged_2025], ignore_index=True)

# =========================
# 9. FINAL CHECK
# =========================
print("\nFinal shape:", merged_all.shape)
print("Missing values:\n", merged_all.isnull().sum().head())

# =========================
# 10. SAVE OUTPUT
# =========================
merged_all.to_csv("merged_raw.csv", index=False)

print("\n✅ merged_raw.csv saved successfully!")