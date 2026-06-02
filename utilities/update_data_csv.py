# DO NOT EDIT - UTILITY TO UPDATE data.csv

# How this works:
# 1. go through each entry for column "videos" in data.csv
# An equivalent exists in folder mediapipe_landmarks as .npz files
# However these files go part_x_[video_name].npz, update videos column
# to accurately reflect the new names.

import pandas as pd

csv_path = "data.csv"
df = pd.read_csv(csv_path)

# Get one video name, lets say its "abc.mp4"
# the equivalent of this file is either part_1_abc.npz or part_2_abc.npz or part_3_abc.npz ... part_11_abc.npz
# We need to find the correct part number and update the videos column to be "part_x'

# use os.path.exists for each of the part numbers to find the correct one
import os


def find_part_number(video_name):
    for i in range(1, 12):  # Check part_1 to part_11
        part_file = f"mediapipe_landmarks/part_{i}_{video_name.replace('.mp4', '.npz')}"
        if os.path.exists(part_file):
            return part_file
    return None  # If no part file is found

df["videos"] = df["videos"].apply(find_part_number)

# Save the updated DataFrame back to CSV
df.to_csv(csv_path, index=False)
