from pathlib import Path
from collections import Counter
import mne
file_path = Path("data/raw/files/S001/S001R04.edf")
print("EEG PREPROCESSING - INITIAL INSPECTION")
if file_path.exists():
    raw = mne.io.read_raw_edf(file_path,preload=False,verbose="ERROR")
    print("\nFILE:", file_path.name)
    print("Sampling frequency:", raw.info["sfreq"], "Hz")
    print("Total channels:", len(raw.ch_names))
    print("Recording duration:", round(raw.times[-1], 2), "seconds")
    print("\nCHANNEL NAMES:")
    print(raw.ch_names)
    print("\nCHANNEL TYPES:")
    channel_types = Counter(raw.get_channel_types())
    print(channel_types)
    print("\nEVENT COUNTS:")
    event_counts = Counter(raw.annotations.description)
    print(event_counts)
    print("\nFIRST 10 ANNOTATIONS:")
    for i in range(10):
        if i < len(raw.annotations):
            onset = raw.annotations.onset[i]
            duration = raw.annotations.duration[i]
            label = raw.annotations.description[i]
            print(
                label,
                "| Onset:", round(float(onset), 2),
                "| Duration:", round(float(duration), 2)
            )
    print("\nEEG INSPECTION COMPLETED")
else:
    print("ERROR: EDF file not found:", file_path)

