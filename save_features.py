import os
import pandas as pd
from tqdm import tqdm
from feature_extractor import extract_feature
from config import FIXED_LEN

feature_type = 'chroma'

input_root = '/home/sangheon/Desktop/GTZAN_Data/Data/genres_original'
output_root = '/home/sangheon/Desktop/GTZAN_Data/Data/Chromagram'

os.makedirs(output_root, exist_ok = True)
genres = os.listdir(input_root)
fixed_length = FIXED_LEN[feature_type]

for genre in genres:
    genre_input = os.path.join(input_root, genre)
    genre_output = os.path.join(output_root, genre)
    os.makedirs(genre_output, exist_ok=True)

    for fname in tqdm(os.listdir(genre_input), desc=f"Processing {genre}"):
        if not fname.endswith('.wav'):
            continue
        try:
            path = os.path.join(genre_input, fname)
            feat = extract_feature(path, feature_type)
            feat_df = pd.DataFrame(feat)
            csv_path = os.path.join(genre_output, fname.replace('.wav', '.csv'))
            feat_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")