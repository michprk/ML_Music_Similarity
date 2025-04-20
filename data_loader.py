import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import *

def load_dataset(feature_root, feature_type = 'mel'):
    genres = os.listdir(feature_root)
    X, y_true, file_names = [], [], []
    fixed_length = FIXED_LEN[feature_type]

    for genre in genres:
        genre_dir = os.path.join(feature_root, genre)

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

def load_dataset(feature_root, feature_type = 'mel'):
    genres = os.listdir(feature_root)
    X, y_true, file_names = [], [], []
    fixed_length = FIXED_LEN[feature_type]

    for genre in genres:
        genre_dir = os.path.join(feature_root, genre)

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

def load_combined_dataset(mel_root, chroma_root):

    genres = os.listdir(mel_root)
    X, y_true, file_names = [], [], []
    fixed_length = FIXED_LEN['concatenated']

    for genre in genres:
        mel_genre_dir = os.path.join(mel_root, genre)
        chroma_genre_dir = os.path.join(chroma_root, genre)

        for csv_file in tqdm(glob.glob(f'{mel_genre_dir}/*.csv'), desc = f"Loading {genre}"):
            fname = os.path.basename(csv_file)
            mel_path = os.path.join(mel_genre_dir, fname)
            chroma_path = os.path.join(chroma_genre_dir, fname)

            mel = pd.read_csv(mel_path).values
            chroma = pd.read_csv(chroma_path).values

            T = min(mel.shape[1], chroma.shape[1])
            mel = mel[:, :T]
            chroma = chroma[:, :T]
            feat = np.concatenate([mel, chroma], axis = 0)
            feat_flat = feat.flatten()
            if len(feat_flat) >= fixed_length:
                feat_flat = feat_flat[:fixed_length]
            else:
                feat_flat = np.pad(feat_flat, (0, fixed_length - len(feat_flat)))

            X.append(feat_flat)
            y_true.append(genre)
            file_names.append(fname)

    return np.array(X), y_true, file_names


