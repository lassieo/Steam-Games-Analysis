import pandas as pd
import os

folder_path = "dataset2"  # change this

def check_data_quality(file_path):
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print("Error reading file:", e)
        return
    
    # ---------------------------
    # 1. Missing Values
    # ---------------------------
    print("\n[Missing Values]")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if missing.empty:
        print("No missing values")
    else:
        for col, val in missing.items():
            print(f"{col}: {val} missing ({val/len(df)*100:.2f}%)")
    
    # ---------------------------
    # 2. Duplicate Rows
    # ---------------------------
    print("\n[Duplicate Rows]")
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    
    # Check duplicates based on appid if exists
    for key in ["appid", "steam_appid", "AppID"]:
        if key in df.columns:
            dup_key = df.duplicated(subset=[key]).sum()
            print(f"Duplicate {key}: {dup_key}")
    
    # ---------------------------
    # 3. Weird Formats
    # ---------------------------
    print("\n[Weird Format Checks]")
    
    for col in df.columns:
        col_data = df[col]
        
        # Mixed types
        types = col_data.map(type).value_counts()
        if len(types) > 1:
            print(f"{col}: Mixed data types -> {types.to_dict()}")
        
        # Check numeric columns stored as text
        if col_data.dtype == object:
            try:
                pd.to_numeric(col_data.dropna())
                print(f"{col}: Might be numeric but stored as string")
            except:
                pass
        
        # Check long text (possible JSON or messy data)
        if col_data.dtype == object:
            avg_len = col_data.dropna().astype(str).map(len).mean()
            if avg_len > 100:
                print(f"{col}: Long text / possible JSON")
    
    print("\n[Basic Info]")
    print(df.info())


# Run for all CSVs
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        check_data_quality(os.path.join(folder_path, file))