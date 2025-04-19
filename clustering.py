from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusterModel:
    def __init__(self, n_clusters = 10):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 10)
        self.pca = PCA(n_components = 2)

    def fit(self, X):
        self.X_scaled = self.scaler.fit_transform(X)
        self.clusters = self.kmeans.fit_predict(self.X_scaled)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def predict_cluster(self, x_new):
        return self.kmeans.predict(x_new)[0]

    def transform_input(self, x_new):
        x_scaled = self.scaler.transform(x_new)
        x_pca = self.pca.transform(x_scaled)
        return x_scaled, x_pca

