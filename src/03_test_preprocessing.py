from pathlib import Path
import mne
import numpy as np
file_path = Path("data/raw/files/S001/S001R04.edf")
RUN_MAPPING = {
    4: {"T1": "left_hand", "T2": "right_hand"},
    8: {"T1": "left_hand", "T2": "right_hand"},
    12: {"T1": "left_hand", "T2": "right_hand"},
    6: {"T1": "both_hands", "T2": "both_feet"},
    10: {"T1": "both_hands", "T2": "both_feet"},
    14: {"T1": "both_hands", "T2": "both_feet"}
}
print("\nEEG PREPROCESSING STARTED")
if file_path.exists():
    print("File found:", file_path.name)
else:
    print("File not found!")
    exit()
run_number = int(file_path.stem[-2:])
if run_number in RUN_MAPPING:
    mapping = RUN_MAPPING[run_number]
    print("Run number:", run_number)
else:
    print("Invalid run number!")
    exit()
print("\nLOADING EEG DATA")
raw = mne.io.read_raw_edf(
    file_path,
    preload=True,
    verbose="ERROR"
)
print("Original sampling rate:", raw.info["sfreq"])
print("Original channels:", len(raw.ch_names))
print("\nSELECTING EEG CHANNELS")
raw.pick("eeg")
print("Selected EEG channels:", len(raw.ch_names))
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
print("\nRESAMPLING EEG DATA")
raw.resample(128,npad="auto",verbose="ERROR")
print("New sampling rate:", raw.info["sfreq"])
print("\nEXTRACTING EVENTS")
events, event_id = mne.events_from_annotations(
    raw,
    event_id={"T1": 1, "T2": 2},
    verbose="ERROR"
)
print("Total events:", len(events))
print("Event IDs:", event_id)
if "T1" in event_id and "T2" in event_id:
    print("Both event types found")
else:
    print("T1 or T2 event missing!")
    exit()
print("\nCREATING EEG EPOCHS")
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
print("Epochs created successfully")
X = epochs.get_data()
print("\nEXTRACTING EEG DATA")
print("Number of epochs:", len(X))
print("EEG data shape:", X.shape)
print("\nGENERATING CLASS LABELS")
event_codes = epochs.events[:, 2]
class_names = []
for code in event_codes:
    if code == event_id["T1"]:
        class_names.append(mapping["T1"])
    else:
        class_names.append(mapping["T2"])
y = np.array(class_names)
print("Labels generated:", len(y))
print("\nCHECKING EEG DATA")
if len(X) == len(y) and len(X) > 0:
    print("Data and labels are correct")
else:
    print("Data validation failed!")
    exit()
if np.isnan(X).any():
    print("Warning: NaN values found")
else:
    print("No NaN values found")
if np.isinf(X).any():
    print("Warning: Infinite values found")
else:
    print("No infinite values found")
print("\nCHECKING DROPPED EPOCHS")
dropped_epochs = 0
for reasons in epochs.drop_log:
    if len(reasons) > 0:
        dropped_epochs = dropped_epochs + 1
print("Dropped epochs:", dropped_epochs)
print("Retained epochs:", len(epochs))
print("\nCLASS DISTRIBUTION")
left_hand = 0
right_hand = 0
both_hands = 0
both_feet = 0
for label in y:
    if label == "left_hand":
        left_hand = left_hand + 1
    elif label == "right_hand":
        right_hand = right_hand + 1
    elif label == "both_hands":
        both_hands = both_hands + 1
    elif label == "both_feet":
        both_feet = both_feet + 1
print("Left Hand:", left_hand)
print("Right Hand:", right_hand)
print("Both Hands:", both_hands)
print("Both Feet:", both_feet)
print("FINAL PREPROCESSING RESULTS")
print("File:", file_path.name)
print("Run number:", run_number)
print("Sampling rate:", raw.info["sfreq"], "Hz")
print("EEG channels:", len(raw.ch_names))
print("Total epochs:", len(X))
print("Data shape:", X.shape)
print("Label shape:", y.shape)
print("\nPREPROCESSING COMPLETED")


