import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def run_pca(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("[PCA] Explained variance:", pca.explained_variance_ratio_)

 
    save_path = "../results/plots/pca.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

   
    labels = np.array(labels)

    # Colors for each genre
    colors = {"literature": "red", "news": "blue", "social": "green"}


    plt.figure()

    for genre in set(labels):
        idx = (labels == genre)
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=genre,
            color=colors.get(genre, "black"),
            alpha=0.6
        )

    plt.legend()
    plt.title("PCA of Genres")

    plt.savefig(save_path)
    plt.close()

    print(f" PCA plot saved at {save_path}")

    return X_pca