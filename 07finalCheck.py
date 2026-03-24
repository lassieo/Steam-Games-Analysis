import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("cleaned_v1.csv")

print("Initial shape:", df.shape)

# =========================
# 2. CHECK DUPLICATES
# =========================
total_duplicates = df.duplicated().sum()
appid_duplicates = df.duplicated(subset=["appid", "snapshot"]).sum()

print("\n--- DUPLICATE CHECK ---")
print("Total duplicate rows:", total_duplicates)
print("Duplicate (appid, snapshot):", appid_duplicates)

# =========================
# 3. CHECK MISSING VALUES
# =========================
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_pct
}).sort_values(by="missing_percent", ascending=False)

print("\n--- MISSING VALUES (TOP 10) ---")
print(missing_df.head(10))

# =========================
# 4. FLAG MAJOR ISSUES
# =========================
# Columns we don't care about missing
ignore_cols = ["website", "support_url", "support_email"]

major_missing = missing_df[
    (missing_df["missing_percent"] > 20) &
    (~missing_df.index.isin(ignore_cols))
]

print("\n--- MAJOR MISSING (>20%) ---")
print(major_missing)

# =========================
# 5. FINAL VERDICT
# =========================
if total_duplicates == 0 and appid_duplicates == 0 and major_missing.empty:
    print("\n✅ Dataset PASSED final check!")
else:
    print("\n⚠️ Dataset has issues — review above.")