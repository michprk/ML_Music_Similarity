from data_loader import load_dataset, load_combined_dataset
from clustering import ClusterModel
from recommender import Recommender
from feature_extractor import extract_feature, flatten_feature
from visualize import plot_clusters
from config import *
import os

feature_type = 'concatenated'
fixed_lengths = FIXED_LEN[feature_type]
cluster_choice = CLUSTER_TYPE['kmeans']

root_base_path = '/home/sangheon/Desktop/GTZAN_Data/Data'

if feature_type == 'concatenated':
    mel_root = os.path.join(root_base_path, 'MelSpec')
    chroma_root = os.path.join(root_base_path, 'Chromagram')
else:
    feature_root = os.path.join(root_base_path, {
        'mel': 'MelSpec',
        'chroma': 'Chromagram'
    }[feature_type])

test_audio = '/home/sangheon/Desktop/ML_Music_Similarity/test_data/Last Dinosaurs - Sense.wav'

if feature_type == 'concatenated':
    X, y_true, file_names = load_combined_dataset(mel_root, chroma_root)
else:
    X, y_true, file_names = load_dataset(feature_root, feature_type)

model = ClusterModel(n_clusters = 10, cluster_type = cluster_choice)
model.fit(X)
plot_clusters(
    model.X_pca_vis,
    model.clusters,
    centroids=model.pca_for_vis.transform(model.kmeans_fallback.cluster_centers_),
    save_path='cluster_plot.png',
    show=True
)

feat_db = extract_feature(test_audio, feature_type)
feat_flat = flatten_feature(feat_db, fixed_lengths)

input_scaled, input_pca  = model.transform_input([feat_flat])
input_cluster = model.predict_cluster([feat_flat])

recommender = Recommender(model.X_reduced, file_names, model.clusters)
recommendations = recommender.recommend(input_scaled, input_cluster, top_k=5)

print("\n Top-5 Recommended Songs:")
for fname, score in recommendations:
    print(f"{fname} (Similarity: {score:.4f})")