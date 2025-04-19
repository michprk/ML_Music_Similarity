import librosa
import numpy as np
from config import *

def extract_mel(audio_path):
    y, sr = librosa.load(audio_path, sr = SR)
    mel = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = N_FFT, hop_length = HOP_LENGTH, n_mels = N_MELS)
    mel_db = librosa.power_to_db(mel)
    return mel_db

def flatten_mel(mel_db, fixed_length = FIXED_LEN):
    mel_flat = mel_db.flatten()
    if len(mel_flat) >= fixed_length:
        return mel_flat[:fixed_length]
    return np.pad(mel_flat, (0, fixed_length - len(mel_flat)))