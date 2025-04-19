import librosa
import numpy as np
from config import *

def extract_feature(audio_path, feature_type = 'mel'):
    y, sr = librosa.load(audio_path, sr = SR)
    if feature_type == 'mel':
        feature = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = N_FFT, hop_length = HOP_LENGTH, n_mels = N_MELS)
        feature_db = librosa.power_to_db(feature)

    elif feature_type == 'chroma':
        feature_db = librosa.feature.chroma_cqt(y = y, sr = sr, hop_length = HOP_LENGTH)

    return feature_db

def flatten_feature(feat, fixed_length):
    feat_flat = feat.flatten()
    if len(feat_flat) >= fixed_length:
        return feat_flat[:fixed_length]
    return np.pad(feat_flat, (0, fixed_length - len(feat_flat)))
