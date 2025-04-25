from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommender:
    def __init__(self, X_reduced, file_names, clusters):
        self.X_reduced = X_reduced
        self.file_names = file_names
        self.clusters = clusters

    def recommend(self, input_vector, input_cluster, top_k=5):
        candidate_indices = np.where(self.clusters == input_cluster)[0]
        similarities = cosine_similarity(input_vector, self.X_reduced[candidate_indices])[0]
        top_k_indices = candidate_indices[np.argsort(similarities)[::-1][:top_k]]
        return [(self.file_names[i], similarities[np.argsort(similarities)[::-1][j]]) for j, i in enumerate(top_k_indices)]