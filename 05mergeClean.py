import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("merged_raw.csv")

# =========================
# 2. STANDARDIZE COLUMNS
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# =========================
# 3. DROP IRRELEVANT COLUMNS
# =========================

# --- High missing / useless ---
drop_cols = [
    "score_rank",
    "metacritic_url",
    "notes"
]

# --- Raw text / JSON (not useful for analysis) ---
drop_cols += [
    "reviews",              # raw text (already extracted metrics)
    "detailed_description",
    "about_the_game",
    "short_description",
    "screenshots",
    "movies",
    "packages"
]

# Drop safely
df = df.drop(columns=drop_cols, errors="ignore")

# =========================
# 4. HANDLE MISSING VALUES
# =========================

# --- URL fields → fill empty ---
url_cols = ["website", "support_url", "support_email"]

for col in url_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# --- Numeric columns → fill with 0 ---
num_cols = df.select_dtypes(include=[np.number]).columns

for col in num_cols:
    df[col] = df[col].fillna(0)

# =========================
# 5. FIX DATA TYPES
# =========================

# --- ID ---
df["appid"] = df["appid"].astype(str)

# --- Date ---
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

# --- Boolean ---
bool_cols = ["windows", "mac", "linux"]

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(bool)

# =========================
# 6. FIX INCONSISTENCIES
# =========================

# --- Ensure price is non-negative ---
df["price"] = df["price"].clip(lower=0)

# --- Fix percentage columns ---
pct_cols = ["pct_pos_total", "pct_pos_recent"]

for col in pct_cols:
    if col in df.columns:
        df[col] = df[col].clip(0, 100)

# --- Remove impossible values ---
df.loc[df["user_score"] < 0, "user_score"] = 0

# =========================
# 7. REMOVE DUPLICATES
# =========================
df = df.drop_duplicates(subset=["appid", "snapshot"])

# =========================
# 8. FINAL CHECK
# =========================
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().head())
print("\nDuplicates:", df.duplicated().sum())

# =========================
# 9. SAVE CLEAN DATA
# =========================
df.to_csv("cleaned_v1.csv", index=False)

print("\n✅ cleaned_v1.csv saved successfully!")