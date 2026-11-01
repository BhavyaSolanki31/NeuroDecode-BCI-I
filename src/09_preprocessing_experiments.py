from pathlib import Path
import numpy as np
import pandas as pd
import mne
data_path = Path("data/raw/files")
split_file = Path("data/processed/subject_splits.csv")
output_path = Path("data/processed/experiments")
output_path.mkdir(parents=True, exist_ok=True)
run_mapping = {
    4: {"T1": "left_hand", "T2": "right_hand"},
    8: {"T1": "left_hand", "T2": "right_hand"},
    12: {"T1": "left_hand", "T2": "right_hand"},
    6: {"T1": "both_hands", "T2": "both_feet"},
    10: {"T1": "both_hands", "T2": "both_feet"},
    14: {"T1": "both_hands", "T2": "both_feet"}
}
experiments = {
    "EXP1_1_40Hz_0_4s": (1.0, 40.0, 0.0, 4.0),
    "EXP2_8_30Hz_0_4s": (8.0, 30.0, 0.0, 4.0),
    "EXP3_1_40Hz_05_25s": (1.0, 40.0, 0.5, 2.5),
    "EXP4_8_30Hz_05_25s": (8.0, 30.0, 0.5, 2.5)
}
if not split_file.exists():
    print("Subject split file not found")
    raise SystemExit
df = pd.read_csv(split_file)
print("\nPREPROCESSING EXPERIMENTS")
for experiment_name, settings in experiments.items():
    low_freq, high_freq, tmin, tmax = settings
    print("EXPERIMENT:", experiment_name)
    experiment_folder = output_path / experiment_name / "subjects"
    experiment_folder.mkdir(parents=True, exist_ok=True)
    total_epochs = 0
    successful_subjects = 0
    failed_subjects = []
    for _, row in df.iterrows():
        subject = row["Subject"]
        split = row["Split"]
        subject_folder = data_path / subject
        X_list = []
        y_list = []
        run_ids = []
        channel_names = []
        for run_number in run_mapping:
            file_path = subject_folder / (
                subject + "R" + str(run_number).zfill(2) + ".edf"
            )
            if not file_path.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(
                    file_path,
                    preload=True,
                    verbose="ERROR"
                )
                raw.pick("eeg")
                raw.filter(
                    l_freq=low_freq,
                    h_freq=high_freq,
                    method="iir",
                    verbose="ERROR"
                )
                raw.set_eeg_reference(
                    ref_channels="average",
                    projection=False,
                    verbose="ERROR"
                )
                raw.resample(128, npad="auto", verbose="ERROR")
                events, event_id = mne.events_from_annotations(
                    raw,
                    event_id={"T1": 1, "T2": 2},
                    verbose="ERROR"
                )
                if len(events) == 0:
                    continue
                if "T1" not in event_id or "T2" not in event_id:
                    continue
                epochs = mne.Epochs(
                    raw,
                    events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    preload=True,
                    reject_by_annotation=True,
                    verbose="ERROR"
                )
                X = epochs.get_data()
                codes = epochs.events[:, -1]
                labels = []
                for code in codes:
                    if code == event_id["T1"]:
                        labels.append(run_mapping[run_number]["T1"])
                    elif code == event_id["T2"]:
                        labels.append(run_mapping[run_number]["T2"])
                if len(X) > 0 and len(X) == len(labels):
                    X_list.append(X)
                    y_list.extend(labels)
                    run_ids.extend([run_number] * len(X))
                    channel_names = raw.ch_names
                del raw
                del epochs
            except Exception as error:
                print("Error:", file_path.name, error)
        if len(X_list) == 0:
            failed_subjects.append(subject)
            continue
        X_final = np.concatenate(X_list, axis=0)
        y_final = np.array(y_list)
        subject_ids = np.array([subject] * len(y_final))
        run_ids = np.array(run_ids)
        if not np.isfinite(X_final).all():
            failed_subjects.append(subject)
            continue
        save_file = experiment_folder / (subject + ".npz")
        np.savez_compressed(
            save_file,
            X=X_final,
            y=y_final,
            subject_ids=subject_ids,
            run_ids=run_ids,
            channel_names=np.array(channel_names),
            sampling_rate=np.array(128.0),
            split=np.array(split)
        )
        total_epochs += len(X_final)
        successful_subjects += 1
        print(
            subject,
            "Saved:",
            len(X_final),
            "epochs"
        )
    print("\nEXPERIMENT RESULTS")
    print("Successful subjects:", successful_subjects)
    print("Failed subjects:", len(failed_subjects))
    print("Total epochs:", total_epochs)
    print("Failed subject IDs:", failed_subjects)
    print("Saved in:", experiment_folder)
print("\nALL PREPROCESSING EXPERIMENTS COMPLETED")
