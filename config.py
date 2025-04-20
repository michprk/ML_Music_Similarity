SR = 16000
N_MELS = 129
N_FFT = 2048
HOP_LENGTH = 512
FIXED_LEN = {
    "mel": 129 * 1293,
    "chroma": 12 * 1293,
    "concatenated" : 141 * 1293
}
CLUSTER_TYPE = {
    "kmeans": "kmeans",
    "spectral": "spectral"
}


