import pandas as pd
import numpy as np

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("final_features.csv")

print("Initial shape:", df.shape)

# =========================
# 2. CHECK MISSING VALUES
# =========================
missing = df.isnull().sum()
missing_total = missing.sum()

print("\n--- MISSING VALUES ---")
print("Total missing values:", missing_total)

# Show columns with missing
missing_cols = missing[missing > 0].sort_values(ascending=False)
print("\nColumns with missing values:\n", missing_cols)

# =========================
# 3. FIX MISSING VALUES (FINAL PASS)
# =========================

# Numeric → fill 0
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

# Categorical → fill "unknown"
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna("unknown")

# Re-check
missing_after = df.isnull().sum().sum()

print("\nMissing after fix:", missing_after)

# =========================
# 4. CHECK DATA TYPES (MODEL READY)
# =========================
print("\n--- DATA TYPES ---")
print(df.dtypes.value_counts())

# =========================
# 5. OPTIONAL: REMOVE NON-NUMERIC (FOR ML)
# =========================

# Keep numeric + encoded features only
model_df = df.select_dtypes(include=[np.number])

print("\nModel dataset shape:", model_df.shape)

# =========================
# 6. FINAL VERDICT
# =========================
if missing_after == 0:
    print("\n✅ Dataset is clean and ready for modeling!")
else:
    print("\n⚠️ Still has missing values.")

# =========================
# 7. SAVE CLEAN VERSION
# =========================
df.to_csv("final_features_clean.csv", index=False)

print("\n✅ final_features_clean.csv saved!")