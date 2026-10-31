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
    14: {"T1": "both_hands", "T2": "both_feet"}}
print("\nLOADING EEG DATA")
raw = mne.io.read_raw_edf(file_path,preload=True,verbose="ERROR")
print("Original sampling rate:", raw.info["sfreq"])
print("Original channels:", len(raw.ch_names))
raw.pick("eeg")
print("EEG channels selected:", len(raw.ch_names))
print("\nAPPLYING BAND-PASS FILTER")
raw.filter(l_freq=1.0,h_freq=40.0,method="iir",verbose="ERROR")
print("APPLYING AVERAGE REFERENCE")
raw.set_eeg_reference(ref_channels="average",projection=False,verbose="ERROR")
print("RESAMPLING TO 128 Hz")
raw.resample(128,npad="auto",verbose="ERROR")
run_number = int(file_path.stem[-2:])
mapping = RUN_MAPPING[run_number]
events, event_id = mne.events_from_annotations(raw,event_id={"T1": 1, "T2": 2},verbose="ERROR")
epochs = mne.Epochs(raw,events,event_id=event_id,tmin=0.0,tmax=4.0,baseline=None,preload=True,reject_by_annotation=True,verbose="ERROR")
X = epochs.get_data()
event_codes = epochs.events[:, 2]
class_names = []
for code in event_codes:
    if code == event_id["T1"]:
        class_names.append(mapping["T1"])
    elif code == event_id["T2"]:
        class_names.append(mapping["T2"])
y = np.array(class_names)
print("PREPROCESSING TEST RESULTS")
print("Run number:", run_number)
print("Final sampling rate:", raw.info["sfreq"])
print("Number of EEG channels:", len(raw.ch_names))
print("Number of epochs:", len(X))
print("EEG data shape:", X.shape)
print("Label shape:", y.shape)
print("\nCLASS DISTRIBUTION:")
for label in sorted(set(y)):
    print(label, ":", np.sum(y == label))
print("\nPREPROCESSING TEST COMPLETED")
