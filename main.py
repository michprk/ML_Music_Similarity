from data_loader import load_dataset
from clustering import ClusterModel
from recommender import Recommender
from feature_extractor import extract_feature, flatten_feature
from visualize import plot_clusters
from config import *
import os

feature_type = 'mel'
fixed_lengths = FIXED_LEN[feature_type]
cluster_choice = CLUSTER_TYPE['spectral']

root_base_path = '/home/sangheon/Desktop/GTZAN_Data/Data'
feature_root = os.path.join(root_base_path, {
    'mel': 'MelSpec',
    'chroma': 'Chromagram'
}[feature_type])

test_audio = '/home/sangheon/Desktop/ML_Music_Similarity/test_data/Last Dinosaurs - Sense.wav'

X, y_true, file_names = load_dataset(feature_root, feature_type)

model = ClusterModel(n_clusters = 10, cluster_type = cluster_choice)
model.fit(X)
plot_clusters(model.X_pca, model.clusters)

mel_db = extract_feature(test_audio, feature_type)
mel_flat = flatten_feature(mel_db, fixed_lengths)
input_scaled, input_pca  = model.transform_input([mel_flat])
input_cluster = model.predict_cluster(input_scaled)

recommender = Recommender(model.X_scaled, file_names, model.clusters)
recommendations = recommender.recommend(input_scaled, input_cluster, top_k=5)

print("\n Top-5 Recommended Songs:")
for fname, score in recommendations:
    print(f"{fname} (Similarity: {score:.4f})")