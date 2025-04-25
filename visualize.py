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

    if centroids is not None:
        centroids = np.array(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, marker='X', label='Centroids')

    if new_point is not None:
        plt.scatter(new_point[0, 0], new_point[0, 1], c='red', s=120, edgecolors='black', label=new_point_label, marker='*')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()


def plot_by_genre(X_pca, y_true, save_path="pca_by_genre.png", show=False):
    """
    Plot PCA-reduced data colored by ground-truth genre labels.

    Parameters:
    - X_pca: PCA-reduced features (n_samples, 2)
    - y_true: list of genre labels (n_samples,)
    - save_path: path to save the image
    - show: whether to display the plot
    """
    plt.figure(figsize=(10, 7))

    # 유일한 장르 추출 및 컬러 매핑
    genres = sorted(list(set(y_true)))
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    genre_colors = [genre_to_idx[label] for label in y_true]

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=genre_colors, cmap='tab10', alpha=0.7, edgecolors='w', linewidths=0.5)

    # 범례 수동 설정
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               label=genre, markersize=8, markerfacecolor=plt.cm.tab10(i / 10))
               for i, genre in enumerate(genres)]

    plt.legend(handles=handles, title="Genres", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection Colored by Ground Truth Genres")
    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

