from pathlib import Path
import mne
import numpy as np
import csv
data_path = Path("data/raw/files")
output_path = Path("data/processed/subjects")
output_path.mkdir(parents=True, exist_ok=True)
subject_numbers = range(1, 110)
run_numbers = [4, 6, 8, 10, 12, 14]
RUN_MAPPING = {
    4: {"T1": "left_hand", "T2": "right_hand"},
    8: {"T1": "left_hand", "T2": "right_hand"},
    12: {"T1": "left_hand", "T2": "right_hand"},
    6: {"T1": "both_hands", "T2": "both_feet"},
    10: {"T1": "both_hands", "T2": "both_feet"},
    14: {"T1": "both_hands", "T2": "both_feet"}
}
total_subjects = 0
total_runs = 0
total_missing_runs = 0
total_epochs = 0
total_dropped = 0
total_left = 0
total_right = 0
total_both_hands = 0
total_both_feet = 0
reference_channels = None
summary_rows = []
print("ALL SUBJECTS EEG PREPROCESSING")
print("Total subjects:", len(subject_numbers))
print("Runs per subject:", len(run_numbers))
print("Expected recordings:", len(subject_numbers) * len(run_numbers))
for subject_number in subject_numbers:
    subject = "S" + str(subject_number).zfill(3)
    subject_path = data_path / subject
    print("PROCESSING SUBJECT:", subject)
    if not subject_path.exists():
        print("Subject folder not found:", subject)
        continue
    subject_X = []
    subject_y = []
    subject_ids = []
    subject_run_ids = []
    subject_left = 0
    subject_right = 0
    subject_both_hands = 0
    subject_both_feet = 0
    subject_successful_runs = 0
    subject_dropped = 0
    subject_missing_runs = 0
    for run_number in run_numbers:
        file_name = subject + "R" + str(run_number).zfill(2) + ".edf"
        file_path = subject_path / file_name
        print("\nProcessing:", file_name)
        if not file_path.exists():
            print("File not found:", file_name)
            subject_missing_runs += 1
            total_missing_runs += 1
            continue
        mapping = RUN_MAPPING[run_number]
        try:
            print("Loading EEG data")
            raw = mne.io.read_raw_edf(
                file_path,
                preload=True,
                verbose="ERROR"
            )
            print("Original sampling rate:", raw.info["sfreq"])
            raw.pick("eeg")
            if len(raw.ch_names) == 0:
                print("No EEG channels found")
                continue
            if reference_channels is None:
                reference_channels = raw.ch_names.copy()
            else:
                if set(raw.ch_names) == set(reference_channels):
                    raw.reorder_channels(reference_channels)
                else:
                    print("EEG channel mismatch. Skipping run.")
                    continue
            print("EEG channels:", len(raw.ch_names))
            print("Applying 1-40 Hz filter")
            raw.filter(
                l_freq=1.0,
                h_freq=40.0,
                method="iir",
                verbose="ERROR"
            )
            print("Applying average reference")
            raw.set_eeg_reference(
                ref_channels="average",
                projection=False,
                verbose="ERROR"
            )
            print("Resampling to 128 Hz")
            raw.resample(
                128,
                npad="auto",
                verbose="ERROR"
            )
            print("Extracting events")
            events, event_id = mne.events_from_annotations(
                raw,
                event_id={"T1": 1, "T2": 2},
                verbose="ERROR"
            )
            if len(events) == 0:
                print("No events found")
                continue
            if "T1" not in event_id or "T2" not in event_id:
                print("T1 or T2 event missing")
                continue
            print("Creating EEG epochs")
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=0.0,
                tmax=4.0,
                baseline=None,
                preload=True,
                reject_by_annotation=True,
                verbose="ERROR"
            )
            X = epochs.get_data().astype(np.float32)
            event_codes = epochs.events[:, 2]
            class_names = []
            for code in event_codes:
                if code == event_id["T1"]:
                    class_names.append(mapping["T1"])
                elif code == event_id["T2"]:
                    class_names.append(mapping["T2"])
            y = np.array(class_names)
            if len(X) != len(y) or len(X) == 0:
                print("Invalid data. Skipping run.")
                continue
            if not np.isfinite(X).all():
                print("NaN or infinite values found. Skipping run.")
                continue
            dropped = len(events) - len(epochs)
            subject_dropped += dropped
            total_dropped += dropped
            for label in y:
                if label == "left_hand":
                    subject_left += 1
                elif label == "right_hand":
                    subject_right += 1
                elif label == "both_hands":
                    subject_both_hands += 1
                elif label == "both_feet":
                    subject_both_feet += 1
            subject_X.append(X)
            subject_y.append(y)
            subject_ids.append(np.array([subject] * len(y)))
            subject_run_ids.append(np.full(len(y), run_number))
            subject_successful_runs += 1
            total_runs += 1
            print("Run completed successfully")
            print("Retained epochs:", len(X))
        except Exception as error:
            print("Error processing", file_name)
            print("Error:", error)
            continue
    if len(subject_X) == 0:
        print("\nNo valid data for:", subject)
        continue
    X_subject = np.concatenate(subject_X, axis=0)
    y_subject = np.concatenate(subject_y, axis=0)
    subject_id_array = np.concatenate(subject_ids)
    run_id_array = np.concatenate(subject_run_ids)
    output_file = output_path / (subject + ".npz")
    np.savez_compressed(
        output_file,
        X=X_subject,
        y=y_subject,
        subject_ids=subject_id_array,
        run_ids=run_id_array,
        channel_names=np.array(reference_channels),
        sampling_rate=np.array(128.0)
    )
    print("\nSUBJECT DATA SAVED:", output_file)
    print("Data shape:", X_subject.shape)
    print("Label shape:", y_subject.shape)
    total_left += subject_left
    total_right += subject_right
    total_both_hands += subject_both_hands
    total_both_feet += subject_both_feet
    subject_epoch_count = len(y_subject)
    total_epochs += subject_epoch_count
    total_subjects += 1
    summary_rows.append({
        "Subject": subject,
        "Successful_Runs": subject_successful_runs,
        "Missing_Runs": subject_missing_runs,
        "Retained_Epochs": subject_epoch_count,
        "Dropped_Epochs": subject_dropped,
        "Left_Hand": subject_left,
        "Right_Hand": subject_right,
        "Both_Hands": subject_both_hands,
        "Both_Feet": subject_both_feet
    })
    print("\nSUBJECT COMPLETED:", subject)
    print("Successful runs:", subject_successful_runs)
    print("Total epochs:", subject_epoch_count)
    print("Left Hand:", subject_left)
    print("Right Hand:", subject_right)
    print("Both Hands:", subject_both_hands)
    print("Both Feet:", subject_both_feet)
print("SAVING DATASET SUMMARY")
summary_file = Path("data/processed/preprocessing_summary.csv")
summary_file.parent.mkdir(parents=True, exist_ok=True)
with open(summary_file, "w", newline="") as csv_file:
    field_names = [
        "Subject",
        "Successful_Runs",
        "Missing_Runs",
        "Retained_Epochs",
        "Dropped_Epochs",
        "Left_Hand",
        "Right_Hand",
        "Both_Hands",
        "Both_Feet"
    ]
    writer = csv.DictWriter(csv_file,fieldnames=field_names)
    writer.writeheader()
    writer.writerows(summary_rows)
print("Summary saved:", summary_file)
print("FINAL EEG DATASET RESULTS")
print("Total subjects processed:", total_subjects)
print("Total successful runs:", total_runs)
print("Total missing recordings:", total_missing_runs)
print("Total retained epochs:", total_epochs)
print("Total dropped epochs:", total_dropped)
print("\nFINAL CLASS DISTRIBUTION")
print("Left Hand:", total_left)
print("Right Hand:", total_right)
print("Both Hands:", total_both_hands)
print("Both Feet:", total_both_feet)
print("\nCLASS TOTAL CHECK")
print("Total class labels:", total_left + total_right + total_both_hands + total_both_feet)
print("PREPROCESSING COMPLETED")
