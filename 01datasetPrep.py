import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("cleaned_v1.csv")

print("Initial shape:", df.shape)

# =========================
# 2. DROP NON-ESSENTIAL COLUMNS
# =========================
# Optional but recommended (these are noisy / not useful)

drop_cols = ["website", "support_url", "support_email"]

df = df.drop(columns=drop_cols, errors="ignore")

# =========================
# 3. ENSURE CONSISTENT TYPES
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

# --- Numeric ---
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 4. FINAL MISSING HANDLING
# =========================

# Fill any remaining numeric NaNs with 0
for col in num_cols:
    df[col] = df[col].fillna(0)

# Fill remaining text NaNs
text_cols = df.select_dtypes(include=["object"]).columns
for col in text_cols:
    df[col] = df[col].fillna("unknown")
# Combine duplicate columns after merge

if "positive_x" in df.columns and "positive_y" in df.columns:
    df["positive"] = df["positive_y"]  # use reviews version
    df = df.drop(columns=["positive_x", "positive_y"])

if "negative_x" in df.columns and "negative_y" in df.columns:
    df["negative"] = df["negative_y"]
    df = df.drop(columns=["negative_x", "negative_y"])
# =========================
# 5. FEATURE ENGINEERING (OPTIONAL BUT STRONG)
# =========================

# --- Positive ratio ---
df["positive_ratio"] = df["positive"] / (
    df["positive"] + df["negative"]
)

# Avoid division by zero
df["positive_ratio"] = df["positive_ratio"].fillna(0)

# --- Total engagement ---
df["total_reviews"] = df["positive"] + df["negative"]

# =========================
# 6. SORT & ORGANIZE
# =========================

df = df.sort_values(by=["appid", "snapshot"])

# =========================
# 7. FINAL CHECK
# =========================
print("\nFinal shape:", df.shape)
print("Remaining missing:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# =========================
# 8. SAVE FINAL DATASET
# =========================
df.to_csv("final_dataset.csv", index=False)

print("\n✅ final_dataset.csv saved successfully!")