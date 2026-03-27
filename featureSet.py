import pandas as pd

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("steam_features_engineered.csv")

# =========================
# 2. ALL FEATURES
# =========================
all_features = df.columns.tolist()

print("\n--- ALL FEATURES ---")
for col in all_features:
    print(col)

print("\nTotal features:", len(all_features))

# =========================
# 3. GROUP FEATURES (FOR REPORT)
# =========================

feature_groups = {
    "Target": [],
    "Identifiers": [],
    "Pricing": [],
    "Temporal": [],
    "Platform / Language": [],
    "Engagement": [],
    "Developer / Publisher": [],
    "Categories (cat_)": [],
    "Genres (genre_)": [],
    "Tags (tag_)": [],
    "EDA Features (eda_)": []
}

for col in all_features:
    if col == "success":
        feature_groups["Target"].append(col)
    elif col in ["appid", "name"]:
        feature_groups["Identifiers"].append(col)
    elif "price" in col or "free" in col:
        feature_groups["Pricing"].append(col)
    elif "release" in col:
        feature_groups["Temporal"].append(col)
    elif "platform" in col or "language" in col:
        feature_groups["Platform / Language"].append(col)
    elif col in ["achievements", "has_achievements"]:
        feature_groups["Engagement"].append(col)
    elif "developer" in col or "publisher" in col:
        feature_groups["Developer / Publisher"].append(col)
    elif col.startswith("cat_"):
        feature_groups["Categories (cat_)"].append(col)
    elif col.startswith("genre_"):
        feature_groups["Genres (genre_)"].append(col)
    elif col.startswith("tag_"):
        feature_groups["Tags (tag_)"].append(col)
    elif col.startswith("eda_"):
        feature_groups["EDA Features (eda_)"].append(col)

# =========================
# 4. PRINT GROUPED FEATURES
# =========================
print("\n--- FEATURE GROUPS ---")

for group, cols in feature_groups.items():
    print(f"\n{group} ({len(cols)} features):")
    for col in cols:
        print("  -", col)

# =========================
# 5. SAVE TO FILE (TXT)
# =========================
with open("feature_set.txt", "w") as f:
    f.write("FEATURE SET\n\n")

    for group, cols in feature_groups.items():
        f.write(f"{group} ({len(cols)} features):\n")
        for col in cols:
            f.write(f"  - {col}\n")
        f.write("\n")

print("\n✅ feature_set.txt saved!")