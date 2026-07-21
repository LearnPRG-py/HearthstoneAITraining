import os
import pandas as pd

# CONFIGURATION
CSV_PATH = "ai_data.csv"
TARGET_DIR = "./mediapipe_landmarks"  # Path to the folder containing the files
FILE_COLUMN = "videos"  # Change to the column name in your CSV holding file names
DRY_RUN = False  # Set to False to actually delete files!

# 1. Load the list of files to keep from the CSV
df = pd.read_csv(CSV_PATH)

# Extract file names and convert to a set for fast lookup
# os.path.basename ensures we match filenames even if full paths are in the CSV
files_to_keep = set(df[FILE_COLUMN].apply(os.path.basename))

# Also explicitly keep the CSV itself so it doesn't get deleted
files_to_keep.add(os.path.basename(CSV_PATH))

print(f"Loaded {len(files_to_keep)} unique file name(s) to keep.\n")

# 2. Iterate through the target directory and remove non-matching files
deleted_count = 0
kept_count = 0

for filename in os.listdir(TARGET_DIR):
    file_path = os.path.join(TARGET_DIR, filename)

    # Only process actual files (skip directories)
    if os.path.isfile(file_path):
        if filename not in files_to_keep:
            if DRY_RUN:
                print(f"[DRY RUN - WOULD DELETE]: {file_path}")
            else:
                os.remove(file_path)
                print(f"[DELETED]: {file_path}")
            deleted_count += 1
        else:
            kept_count += 1

print("\n--- Summary ---")
print(f"Files kept: {kept_count}")
print(f"Files {'that would be deleted' if DRY_RUN else 'deleted'}: {deleted_count}")

if DRY_RUN:
    print("\n⚠️  This was a DRY RUN. Set DRY_RUN = False in the script to execute deletions.")