from pathlib import Path
import numpy as np
import pandas as pd
base_path = Path("data/processed/experiments")
output_path = Path("data/processed")
classes = ["left_hand", "right_hand", "both_hands", "both_feet"]
experiments = [
    "EXP1_1_40Hz_0_4s",
    "EXP2_8_30Hz_0_4s",
    "EXP3_1_40Hz_05_25s",
    "EXP4_8_30Hz_05_25s"
]
results = []
print("\nPREPROCESSING EXPERIMENT COMPARISON")
for experiment in experiments:
    folder = base_path / experiment / "subjects"
    if not folder.exists():
        print("Folder not found:", experiment)
        continue
    files = sorted(folder.glob("S*.npz"))
    total_epochs = 0
    class_counts = {label: 0 for label in classes}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    shapes = set()
    valid_subjects = 0
    for file_path in files:
        try:
            data = np.load(file_path, allow_pickle=False)
            X = data["X"]
            y = data["y"]
            split = str(data["split"].item())
            if len(X) != len(y):
                print("Invalid data:", file_path.name)
                data.close()
                continue
            if not np.isfinite(X).all():
                print("Invalid EEG values:", file_path.name)
                data.close()
                continue
            if split not in split_counts:
                print("Invalid split:", file_path.name)
                data.close()
                continue
            valid_subjects += 1
            total_epochs += len(y)
            split_counts[split] += len(y)
            shapes.add(X.shape[1:])
            for label in classes:
                class_counts[label] += int(np.sum(y == label))
            data.close()
        except Exception as error:
            print("Error:", file_path.name, error)
    print("\n", experiment)
    print("Subjects:", valid_subjects)
    print("Epochs:", total_epochs)
    print("Shapes:", shapes)
    print("Class distribution:", class_counts)
    print("Split distribution:", split_counts)
    results.append({
        "Experiment": experiment,
        "Subjects": valid_subjects,
        "Total_Epochs": total_epochs,
        "Data_Shape": str(shapes),
        "Left_Hand": class_counts["left_hand"],
        "Right_Hand": class_counts["right_hand"],
        "Both_Hands": class_counts["both_hands"],
        "Both_Feet": class_counts["both_feet"],
        "Train_Epochs": split_counts["train"],
        "Validation_Epochs": split_counts["validation"],
        "Test_Epochs": split_counts["test"]
    })
if len(results) == 0:
    print("No experiment results found")
    raise SystemExit
df = pd.DataFrame(results)
report_file = output_path / "preprocessing_comparison.csv"
df.to_csv(report_file, index=False)
print("\nFINAL COMPARISON")
print(df.to_string(index=False))
print("\nReport saved:", report_file)
print("\nPREPROCESSING COMPARISON COMPLETED")
