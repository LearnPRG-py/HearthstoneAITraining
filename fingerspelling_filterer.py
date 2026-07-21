# Gemini generated script for rivial filtering.

import pandas as pd
df = pd.read_csv("ai_data.csv")

fs_df = df[df['word'] == 'fs']
other_df = df[df['word'] != 'fs']
sampled_fs_df = fs_df.sample(n=min(256, len(fs_df)), random_state=42)
final_df = pd.concat([other_df, sampled_fs_df]).sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv("ai_data.csv", index=False)

print(f"Original 'fs' rows: {len(fs_df)}")
print(f"Sampled 'fs' rows: {len(sampled_fs_df)}")
print(f"Total rows in new dataset: {len(final_df)}")