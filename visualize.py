import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(X_pca, clusters, centroids=None, new_point=None, new_point_label="Input", save_path="clusters.png", show=False):
    """
    Parameters:
    - X_pca: 2D PCA-reduced features (n_samples, 2)
    - clusters: cluster assignment (n_samples,)
    - centroids: optional (n_clusters, 2), PCA-reduced cluster centers
    - new_point: optional (1, 2) array-like, new input point in PCA space
    - new_point_label: str, label for the new point
    - save_path: str, where to save the figure
    - show: bool, whether to show plot interactively
    """
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6, edgecolors='w', linewidths=0.5)
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Clustering Result (PCA-Reduced)")
    plt.grid(True)

    # ✅ 클러스터 중심 표시
    if centroids is not None:
        centroids = np.array(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X', label='Centroids')

    # ✅ 새 입력 포인트 강조
    if new_point is not None:
        plt.scatter(new_point[0, 0], new_point[0, 1], c='red', s=120, edgecolors='black', label=new_point_label, marker='*')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()
