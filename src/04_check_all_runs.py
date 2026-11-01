from pathlib import Path
import mne
import numpy as np
subject = "S001"
data_path = Path("data/raw/files") / subject
run_numbers = [4, 6, 8, 10, 12, 14]
RUN_MAPPING = {
    4: {"T1": "left_hand", "T2": "right_hand"},
    8: {"T1": "left_hand", "T2": "right_hand"},
    12: {"T1": "left_hand", "T2": "right_hand"},
    6: {"T1": "both_hands", "T2": "both_feet"},
    10: {"T1": "both_hands", "T2": "both_feet"},
    14: {"T1": "both_hands", "T2": "both_feet"}
}
all_X = []
all_y = []
total_epochs = 0
total_dropped = 0
total_events = 0
left_hand = 0
right_hand = 0
both_hands = 0
both_feet = 0
successful_runs = 0
print("EEG ALL-RUNS PREPROCESSING TEST")
print("Subject:", subject)
for run_number in run_numbers:
    print("PROCESSING RUN:", run_number)
    file_path = data_path / (subject + "R" + str(run_number).zfill(2) + ".edf")
    if file_path.exists():
        print("File found:", file_path.name)
    else:
        print("File not found:", file_path.name)
        continue
    mapping = RUN_MAPPING[run_number]
    print("\nLOADING EEG DATA")
    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose="ERROR"
    )
    print("Original sampling rate:", raw.info["sfreq"])
    print("Original channels:", len(raw.ch_names))
    raw.pick("eeg")
    if len(raw.ch_names) == 0:
        print("No EEG channels found. Skipping run.")
        continue
    print("EEG channels:", len(raw.ch_names))
    print("\nAPPLYING BAND-PASS FILTER")
    raw.filter(
        l_freq=1.0,
        h_freq=40.0,
        method="iir",
        verbose="ERROR"
    )
    print("Filter applied: 1-40 Hz")
    print("\nAPPLYING AVERAGE REFERENCE")
    raw.set_eeg_reference(ref_channels="average",projection=False,verbose="ERROR")
    print("Average reference applied")
    print("\nRESAMPLING")
    raw.resample(
        128,
        npad="auto",
        verbose="ERROR"
    )
    print("New sampling rate:", raw.info["sfreq"])
    print("\nEXTRACTING EVENTS")
    events, event_id = mne.events_from_annotations(
        raw,
        event_id={"T1": 1, "T2": 2},
        verbose="ERROR"
    )
    print("Event IDs:", event_id)
    print("Total events:", len(events))
    if len(events) == 0:
        print("No events found. Skipping run.")
        continue
    if "T1" not in event_id or "T2" not in event_id:
        print("T1 or T2 missing. Skipping run.")
        continue
    print("\nCREATING EPOCHS")
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
    X = epochs.get_data()
    print("EEG data shape:", X.shape)
    print("Retained epochs:", len(X))
    event_codes = epochs.events[:, 2]
    class_names = []
    for code in event_codes:
        if code == event_id["T1"]:
            class_names.append(mapping["T1"])
        elif code == event_id["T2"]:
            class_names.append(mapping["T2"])
    y = np.array(class_names)
    if len(X) != len(y) or len(X) == 0:
        print("Data validation failed. Skipping run.")
        continue
    if not np.isfinite(X).all():
        print("Invalid EEG values found. Skipping run.")
        continue
    dropped = 0
    for reasons in epochs.drop_log:
        if len(reasons) > 0:
            dropped = dropped + 1
    print("\nRUN CLASS DISTRIBUTION")
    run_left = 0
    run_right = 0
    run_both_hands = 0
    run_both_feet = 0
    for label in y:
        if label == "left_hand":
            run_left = run_left + 1
        elif label == "right_hand":
            run_right = run_right + 1
        elif label == "both_hands":
            run_both_hands = run_both_hands + 1
        elif label == "both_feet":
            run_both_feet = run_both_feet + 1
    print("Left Hand:", run_left)
    print("Right Hand:", run_right)
    print("Both Hands:", run_both_hands)
    print("Both Feet:", run_both_feet)
    print("Dropped epochs:", dropped)
    all_X.append(X)
    all_y.append(y)
    total_epochs = total_epochs + len(X)
    total_events = total_events + len(events)
    total_dropped = total_dropped + dropped
    left_hand = left_hand + run_left
    right_hand = right_hand + run_right
    both_hands = both_hands + run_both_hands
    both_feet = both_feet + run_both_feet
    successful_runs = successful_runs + 1
    print("\nRun processing completed!")
print("COMBINING EEG DATA")
if successful_runs == 0:
    print("No runs were processed successfully.")
    raise SystemExit
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
print("Combined EEG shape:", X_all.shape)
print("Combined label shape:", y_all.shape)
print("FINAL CLASS DISTRIBUTION")
print("Left Hand:", left_hand)
print("Right Hand:", right_hand)
print("Both Hands:", both_hands)
print("Both Feet:", both_feet)
print("FINAL DATASET SUMMARY")
print("Subject:", subject)
print("Successful runs:", successful_runs)
print("Total events:", total_events)
print("Total retained epochs:", total_epochs)
print("Total dropped epochs:", total_dropped)
print("EEG channels:", X_all.shape[1])
print("Time samples:", X_all.shape[2])
print("Combined data shape:", X_all.shape)
print("Combined label shape:", y_all.shape)
print("Missing values:", np.isnan(X_all).sum())
print("Infinite values:", np.isinf(X_all).sum())
print("\nALL-RUNS PREPROCESSING COMPLETED")
