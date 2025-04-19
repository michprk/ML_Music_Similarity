from data_loader import load_dataset
from clustering import ClusterModel
from recommender import Recommender
from feature_extractor import extract_mel, flatten_mel
from visualize import plot_clusters
from config import *

mel_root = '/home/sangheon/Desktop/GTZAN_Data/Data/MelSpec'
test_audio = '/home/sangheon/Desktop/ML_Music_Similarity/test_data/Last Dinosaurs - Sense.wav'

X, y_true, file_names = load_dataset(mel_root)

model = ClusterModel(n_clusters = 10)
model.fit(X)
plot_clusters(model.X_pca, model.clusters)

mel_db = extract_mel(test_audio)
mel_flat = flatten_mel(mel_db)
input_scaled, input_pca  = model.transform_input([mel_flat])
input_cluster = model.predict_cluster(input_scaled)

recommender = Recommender(model.X_scaled, file_names, model.clusters)
recommendations = recommender.recommend(input_scaled, input_cluster, top_k=5)

print("\nTop-5 Recommended Songs:")
for fname, score in recommendations:
    print(f"{fname} (Similarity: {score:.4f})")