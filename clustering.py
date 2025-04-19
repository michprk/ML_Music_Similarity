from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusterModel:
    def __init__(self, n_clusters = 10, cluster_type = "kmeans"):
        self.scaler = StandardScaler()
        self.cluster_type = cluster_type
        self.pca = PCA(n_components = 2)
        self.kmeans_fallback = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.n_clusters = n_clusters
        if cluster_type == "kmeans":
            self.clusterer = self.kmeans_fallback

        elif cluster_type == "spectral":
            self.clusterer = SpectralClustering(n_clusters = n_clusters, random_state = 42, affinity='nearest_neighbors')

    def fit(self, X):
        self.X_scaled = self.scaler.fit_transform(X)
        self.clusters = self.clusterer.fit_predict(self.X_scaled)
        if self.cluster_type == "spectral":
            self.kmeans_fallback.fit(self.X_scaled)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def predict_cluster(self, x_new):
        return self.kmeans_fallback.predict(x_new)[0]

    def transform_input(self, x_new):
        x_scaled = self.scaler.transform(x_new)
        x_pca = self.pca.transform(x_scaled)
        return x_scaled, x_pca


