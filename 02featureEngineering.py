import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_dataset.csv")

print("Initial shape:", df.shape)

# =========================
# 2. LOG TRANSFORM FEATURES
# =========================
# Log helps reduce skew (VERY important)

log_cols = [
    "positive",
    "negative",
    "total_reviews",
    "average_playtime_forever",
    "peak_ccu"
]

for col in log_cols:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])  # log(1 + x) avoids log(0)

# =========================
# 3. HANDLE ESTIMATED OWNERS
# =========================
# Convert range like "10000-20000" → midpoint

def parse_owners(val):
    try:
        low, high = val.split("-")
        return (int(low) + int(high)) / 2
    except:
        return np.nan

df["owners_mid"] = df["estimated_owners"].apply(parse_owners)

# Log transform owners
df["log_owners"] = np.log1p(df["owners_mid"])

# =========================
# 4. ENCODING (CATEGORICAL)
# =========================

# --- OPTION A: SIMPLE (RECOMMENDED) ---
# Extract first genre only (reduces dimensionality)

df["primary_genre"] = df["genres"].astype(str).str.split(";").str[0]

# One-hot encoding
genre_dummies = pd.get_dummies(df["primary_genre"], prefix="genre")

df = pd.concat([df, genre_dummies], axis=1)

# =========================
# 5. TAG FEATURES (LIGHT VERSION)
# =========================

# Count number of tags (simple but useful feature)
df["num_tags"] = df["tags"].astype(str).apply(lambda x: len(x.split(",")))

# =========================
# 6. NORMALIZE PERCENTAGES
# =========================

pct_cols = ["pct_pos_total", "pct_pos_recent"]

for col in pct_cols:
    if col in df.columns:
        df[col] = df[col] / 100  # convert to 0–1 scale

# =========================
# 7. FINAL CHECK
# =========================
print("\nFinal shape:", df.shape)
print("New columns added:",
      [col for col in df.columns if "log_" in col or "genre_" in col])

# =========================
# 8. SAVE OUTPUT
# =========================
df.to_csv("final_features.csv", index=False)

print("\n✅ final_features.csv saved successfully!")