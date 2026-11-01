from pathlib import Path
import numpy as np
import csv
data_path = Path("data/processed/subjects")
subject_numbers = range(1, 110)
classes = [
    "left_hand",
    "right_hand",
    "both_hands",
    "both_feet"
]
total_subjects = 0
total_epochs = 0
class_counts = {
    "left_hand": 0,
    "right_hand": 0,
    "both_hands": 0,
    "both_feet": 0
}
missing_subjects = []
invalid_subjects = []
print("PROCESSED DATASET VALIDATION")
for number in subject_numbers:
    subject = "S" + str(number).zfill(3)
    file_path = data_path / (subject + ".npz")
    print("\nChecking:", subject)
    if not file_path.exists():
        print("File missing!")
        missing_subjects.append(subject)
        continue
    try:
        data = np.load(file_path, allow_pickle=False)
        X = data["X"]
        y = data["y"]
        subject_ids = data["subject_ids"]
        run_ids = data["run_ids"]
        channel_names = data["channel_names"]
        sampling_rate = data["sampling_rate"]
        if X.ndim != 3:
            print("Invalid EEG dimensions")
            invalid_subjects.append(subject)
            data.close()
            continue
        if len(X) != len(y):
            print("Data and labels do not match")
            invalid_subjects.append(subject)
            data.close()
            continue
        if len(X) != len(subject_ids):
            print("Subject ID mismatch")
            invalid_subjects.append(subject)
            data.close()
            continue
        if len(X) != len(run_ids):
            print("Run ID mismatch")
            invalid_subjects.append(subject)
            data.close()
            continue
        if X.shape[1] != len(channel_names):
            print("Channel count mismatch")
            invalid_subjects.append(subject)
            data.close()
            continue
        if X.shape[1] != 64 or X.shape[2] != 513:
            print("Unexpected EEG shape:", X.shape)
            invalid_subjects.append(subject)
            data.close()
            continue
        if float(sampling_rate) != 128.0:
            print("Incorrect sampling rate")
            invalid_subjects.append(subject)
            data.close()
            continue
        if not np.isfinite(X).all():
            print("NaN or infinite values found")
            invalid_subjects.append(subject)
            data.close()
            continue
        if not np.all(subject_ids == subject):
            print("Incorrect subject IDs")
            invalid_subjects.append(subject)
            data.close()
            continue
        valid_runs = [4, 6, 8, 10, 12, 14]
        if not np.isin(run_ids, valid_runs).all():
            print("Invalid run IDs")
            invalid_subjects.append(subject)
            data.close()
            continue
        valid_labels = True
        for label in y:
            if label not in classes:
                valid_labels = False
        if not valid_labels:
            print("Unknown class label found")
            invalid_subjects.append(subject)
            data.close()
            continue
        for label in y:
            class_counts[label] += 1
        total_epochs += len(X)
        total_subjects += 1
        print("Validation successful")
        print("Epochs:", len(X))
        print("Data shape:", X.shape)
        print("Labels:", len(y))
        data.close()
    except Exception as error:
        print("Error:", error)
        invalid_subjects.append(subject)
print("FINAL VALIDATION RESULTS")
print("Expected subjects:", 109)
print("Valid subjects:", total_subjects)
print("Missing subjects:", len(missing_subjects))
print("Invalid subjects:", len(invalid_subjects))
print("Total EEG epochs:", total_epochs)
print("\nFOUR-CLASS DISTRIBUTION")
for label in classes:
    print(label, ":", class_counts[label])
print("\nCHECKING ALL FOUR CLASSES")
for label in classes:
    if class_counts[label] > 0:
        print(label, ": Available")
    else:
        print(label, ": Missing")
print("\nMISSING SUBJECT FILES")
if len(missing_subjects) == 0:
    print("No missing subject files")
else:
    for subject in missing_subjects:
        print(subject)
print("\nINVALID SUBJECT FILES")
if len(invalid_subjects) == 0:
    print("No invalid subject files")
else:
    for subject in invalid_subjects:
        print(subject)
report_path = Path("data/processed/validation_report.csv")
with open(report_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Expected subjects", 109])
    writer.writerow(["Valid subjects", total_subjects])
    writer.writerow(["Missing subjects", len(missing_subjects)])
    writer.writerow(["Invalid subjects", len(invalid_subjects)])
    writer.writerow(["Total epochs", total_epochs])
    for label in classes:
        writer.writerow([label, class_counts[label]])
    writer.writerow(["Missing subject IDs", ", ".join(missing_subjects)])
    writer.writerow(["Invalid subject IDs", ", ".join(invalid_subjects)])
if total_subjects > 0 and len(invalid_subjects) == 0:
    print("DATASET VALIDATION COMPLETED")
else:
    print("DATASET NEEDS ATTENTION")
print("Report saved:", report_path)