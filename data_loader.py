import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import *

def load_dataset(mel_root, fixed_length = FIXED_LEN):
    genres = os.listdir(mel_root)
    X, y_true, file_names = [], [], []

    for genre in genres:
        genre_dir = os.path.join(mel_root, genre)
        for csv_file in tqdm(glob.glob(f'{genre_dir}/*.csv'), desc = f"Loading {genre}"):
            try:
                mel = pd.read_csv(csv_file).values
                mel_flat = mel.flatten()
                if len(mel_flat) >= fixed_length:
                    mel_flat = mel_flat[:fixed_length]
                else:
                    mel_flat = np.pad(mel_flat, (0, fixed_length - len(mel_flat)))
                X.append(mel_flat)
                y_true.append(genre)
                file_names.append(os.path.basename(csv_file))
            except Exception as e:
                print(f'Error loading {csv_file}: {e}')
    return np.array(X), y_true, file_names
