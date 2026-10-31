import re
from pathlib import Path
from collections import Counter
import mne
DATA_ROOT = Path("data/raw/files")
RUN_MAPPING = {
    4: {"T1": "left_hand", "T2": "right_hand"},
    8: {"T1": "left_hand", "T2": "right_hand"},
    12: {"T1": "left_hand", "T2": "right_hand"},
    6: {"T1": "both_hands", "T2": "both_feet"},
    10: {"T1": "both_hands", "T2": "both_feet"},
    14: {"T1": "both_hands", "T2": "both_feet"}
}
class_totals = Counter()
run_totals = Counter()
checked_files = 0
missing_files = []
errors = []
print("=" * 55)
print("FOUR-CLASS MOTOR IMAGERY LABEL AUDIT")
print("=" * 55)
if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")
subjects = sorted(
    folder for folder in DATA_ROOT.iterdir()
    if folder.is_dir()
    and re.fullmatch(r"S\d{3}", folder.name))
print("\nTOTAL SUBJECTS FOUND:", len(subjects))
for subject in subjects:
    for run, mapping in RUN_MAPPING.items():
        filename = subject / f"{subject.name}R{run:02d}.edf"
        if not filename.exists():
            missing_files.append(str(filename))
            continue
        try:
            raw = mne.io.read_raw_edf(filename,preload=False,verbose="ERROR")
            events = Counter(raw.annotations.description)
            for event_code, class_name in mapping.items():
                count = events.get(event_code, 0)
                class_totals[class_name] += count
                run_totals[(run, class_name)] += count
            checked_files += 1
        except Exception as e:
            errors.append((filename.name, str(e)))
print("\nFILES SUCCESSFULLY CHECKED:", checked_files)
print("\nCLASS-WISE EVENT COUNTS")
for name in ["left_hand","right_hand","both_hands","both_feet"]:
    print(f"{name}: {class_totals[name]}")
print("\nRUN-WISE EVENT COUNTS")
for (run, name), count in sorted(run_totals.items()):
    print(f"Run {run} | {name}: {count}")
print("\nMISSING FILES:", len(missing_files))
print("READING ERRORS:", len(errors))
if missing_files:
    print("\nMISSING FILE DETAILS:")
    for file in missing_files:
        print(file)
if errors:
    print("\nREADING ERROR DETAILS:")
    for file, message in errors:
        print(file, message)
print("\n" + "=" * 55)
if (
    len(subjects) == 109
    and checked_files == 654
    and len(missing_files) == 0
    and len(errors) == 0
):
    print("LABEL AUDIT PASSED")
else:
    print("LABEL AUDIT NEEDS ATTENTION")
print("=" * 55)



