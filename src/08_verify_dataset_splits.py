from pathlib import Path
import pandas as pd
file_path = Path("data/processed/subject_splits.csv")
if not file_path.exists():
    print("Split file not found")
    raise SystemExit
df = pd.read_csv(file_path)
print("\nDATASET SPLIT VERIFICATION")
required_columns = ["Subject", "Split", "Total_Epochs", "Left_Hand", "Right_Hand", "Both_Hands", "Both_Feet"]
if not all(column in df.columns for column in required_columns):
    print("Required columns are missing")
    raise SystemExit
if df.isnull().any().any():
    print("Missing values found")
    raise SystemExit
if df["Subject"].duplicated().any():
    print("Duplicate subjects found")
    raise SystemExit
valid_splits = ["train", "validation", "test"]
if not df["Split"].isin(valid_splits).all():
    print("Invalid split names found")
    raise SystemExit
print("\nSUBJECT DISTRIBUTION")
for split in valid_splits:
    subset = df[df["Split"] == split]
    print(split, ":", len(subset), "subjects")
print("\nCLASS DISTRIBUTION")
class_columns = ["Left_Hand", "Right_Hand", "Both_Hands", "Both_Feet"]
for split in valid_splits:
    subset = df[df["Split"] == split]
    print("\n", split.upper())
    for column in class_columns:
        count = subset[column].sum()
        print(column, ":", count)
        if count == 0:
            print("WARNING: Class missing in", split)
print("\nEPOCH DISTRIBUTION")
for split in valid_splits:
    subset = df[df["Split"] == split]
    print(split, ":", subset["Total_Epochs"].sum())
print("\nCHECKING SUBJECT OVERLAP")
train_subjects = set(df[df["Split"] == "train"]["Subject"])
val_subjects = set(df[df["Split"] == "validation"]["Subject"])
test_subjects = set(df[df["Split"] == "test"]["Subject"])
train_val = train_subjects.intersection(val_subjects)
train_test = train_subjects.intersection(test_subjects)
val_test = val_subjects.intersection(test_subjects)
print("Train-Validation overlap:", len(train_val))
print("Train-Test overlap:", len(train_test))
print("Validation-Test overlap:", len(val_test))
if len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0:
    print("\nNO SUBJECT OVERLAP FOUND")
else:
    print("\nDATASET SPLIT NEEDS CORRECTION")
print("\nSPLIT VERIFICATION COMPLETED")
