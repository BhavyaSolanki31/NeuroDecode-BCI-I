from pathlib import Path
import numpy as np
import csv
data_path = Path("data/processed/subjects")
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42
classes = ["left_hand", "right_hand", "both_hands", "both_feet"]
print("\nEEG DATASET SPLITTING")
if not data_path.exists():
    print("Dataset folder not found:", data_path)
    raise SystemExit
if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.0001:
    print("Split ratios must add up to 1.")
    raise SystemExit
subject_files = []
for file_path in sorted(data_path.glob("S*.npz")):
    try:
        with np.load(file_path, allow_pickle=False) as data:
            if "X" not in data or "y" not in data:
                print("Missing X or y:", file_path.name)
                continue
            X = data["X"]
            y = data["y"]
            if X.ndim != 3:
                print("Invalid dimensions:", file_path.name)
                continue
            if len(X) == 0 or len(X) != len(y):
                print("Invalid data length:", file_path.name)
                continue
            if X.shape[1] != 64 or X.shape[2] != 513:
                print("Incorrect EEG shape:", file_path.name)
                continue
            if not np.isfinite(X).all():
                print("Invalid EEG values:", file_path.name)
                continue
            if not np.isin(y, classes).all():
                print("Invalid labels:", file_path.name)
                continue
            subject_files.append(file_path)
            print("Valid:", file_path.name)
    except Exception as error:
        print("Error reading:", file_path.name)
        print(error)
total_subjects = len(subject_files)
print("\nTotal valid subjects:", total_subjects)
if total_subjects < 3:
    print("Not enough subjects for splitting.")
    raise SystemExit
rng = np.random.default_rng(random_seed)
rng.shuffle(subject_files)
train_count = int(total_subjects * train_ratio)
val_count = int(total_subjects * val_ratio)
test_count = total_subjects - train_count - val_count
train_files = subject_files[:train_count]
val_files = subject_files[train_count:train_count + val_count]
test_files = subject_files[train_count + val_count:]
print("\nSPLIT RESULTS")
print("Training subjects:", len(train_files))
print("Validation subjects:", len(val_files))
print("Testing subjects:", len(test_files))
split_data = {
    "train": train_files,
    "validation": val_files,
    "test": test_files
}
split_counts = {}
for split_name in split_data:
    split_counts[split_name] = {}
    for label in classes:
        split_counts[split_name][label] = 0
split_rows = []
for split_name in split_data:
    for file_path in split_data[split_name]:
        with np.load(file_path, allow_pickle=False) as data:
            y = data["y"]
            class_count = {}
            for label in classes:
                count = int(np.sum(y == label))
                class_count[label] = count
                split_counts[split_name][label] += count
            row = {
                "Subject": file_path.stem,
                "Split": split_name,
                "Total_Epochs": len(y),
                "Left_Hand": class_count["left_hand"],
                "Right_Hand": class_count["right_hand"],
                "Both_Hands": class_count["both_hands"],
                "Both_Feet": class_count["both_feet"]
            }
            split_rows.append(row)
split_file = output_path / "subject_splits.csv"
with open(split_file, "w", newline="") as file:
    field_names = [
        "Subject",
        "Split",
        "Total_Epochs",
        "Left_Hand",
        "Right_Hand",
        "Both_Hands",
        "Both_Feet"
    ]
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(split_rows)
print("\nCLASS DISTRIBUTION")
for split_name in split_data:
    print("\n", split_name.upper())
    for label in classes:
        print(label, ":", split_counts[split_name][label])
print("\nFINAL DATASET SUMMARY")
print("Total subjects:", total_subjects)
print("Training subjects:", train_count)
print("Validation subjects:", val_count)
print("Testing subjects:", test_count)
print("\nSplit file saved:", split_file)
print("\nDATASET SPLITTING COMPLETED")

