from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusterModel:
    def __init__(self, n_clusters = 10, cluster_type = "kmeans", pca_enabled = True, pca_dim = 300):
        self.scaler = StandardScaler()
        self.cluster_type = cluster_type
        self.kmeans_fallback = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.n_clusters = n_clusters
        self.pca_enabled = pca_enabled
        self.pca_dim = pca_dim
        self.pca = PCA(n_components = pca_dim) if pca_enabled else None
        self.pca_for_vis = PCA(n_components = 5)

        if cluster_type == "kmeans":
            self.clusterer = self.kmeans_fallback

        elif cluster_type == "spectral":
            self.clusterer = SpectralClustering(n_clusters = n_clusters, random_state = 42, affinity='nearest_neighbors', n_neighbors= 5)

    def fit(self, X):
        self.X_scaled = self.scaler.fit_transform(X)

        if self.pca_enabled:
            self.X_reduced = self.pca.fit_transform(self.X_scaled)
        else:
            self.X_reduced = self.X_scaled

        self.clusters = self.clusterer.fit_predict(self.X_reduced)

        self.kmeans_fallback.fit(self.X_reduced)

        self.X_pca_vis = self.pca_for_vis.fit_transform(self.X_reduced)

    def predict_cluster(self, x_new):
        x_scaled = self.scaler.transform(x_new)
        x_reduced = self.pca.transform(x_scaled) if self.pca_enabled else x_scaled
        return self.kmeans_fallback.predict(x_reduced)[0]

    def transform_input(self, x_new):
        x_scaled = self.scaler.transform(x_new)
        x_reduced = self.pca.transform(x_scaled) if self.pca_enabled else x_scaled
        x_pca_2d = self.pca_for_vis.transform(x_reduced)
        return x_reduced, x_pca_2d