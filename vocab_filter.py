import pandas as pd

should_be_there = [
    'help', 'stop', 'go', 'wait', 'now', 'careful', 'call', 'need', 'safe', 'danger', 'rescue', 'escape',
    'doctor', 'nurse', 'hospital', 'interpreter', 'deaf', 'hardofhearing', 'tty', 'callattention', 'calltty',
    'hurt', 'pain', 'bleed', 'blood', 'sick', 'breathe', 'accident',
    'fire', 'smoke', 'gun', 'shoot', 'shooting', 'siren', 'alarm', 'drown', 'water',
    'above', 'under', 'over', 'between', 'behind', 'front', 'back', 'near', 'outside', 'inside', 'up', 'down', 'left', 'right',
    'name', 'address', 'phone', 'write', 'read', 'understand', 'know', 'sign', 'fingerspell', 'fingerspelling',
    'where', 'when', 'who', 'what', 'why', 'how',
    'yes', 'no', 'ok',
    'anyone'
]

fingerspelling = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
    'u', 'v', 'w', 'x', 'y', 'z'
]

df = pd.read_csv("data.csv")
condition = df['word'].isin(should_be_there) | df['word'].isin(fingerspelling)
export = df[condition].copy()
export.loc[export['word'].isin(fingerspelling), 'word'] = "fs"
export.to_csv("ai_data.csv", index=False)
