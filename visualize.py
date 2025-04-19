import matplotlib.pyplot as plt

def plot_clusters(X_pca, clusters):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.title("K-Means Clustering of Mel Spectrograms (PCA-Reduced)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("clusters.png")
    plt.close()
