"""
pca_analysis.py
PCA visualization of the structural feature space, with per-genre
cluster dispersion (Syntactic Rigidity Metric visual).
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PLOTS_DIR = "results/plots"
GENRE_COLORS = {
    "literature": "#534AB7",
    "news":       "#0F6E56",
    "social":     "#993C1D",
}


def run_pca(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    n_components: int = 5,
    save_dir: str = PLOTS_DIR,
):
    """
    1. Fit PCA, report explained variance.
    2. 2D scatter plot coloured by genre (with confidence ellipses).
    3. Scree plot.
    4. Feature loading plot (PC1 vs PC2).
    5. Return projected coordinates.
    """
    os.makedirs(save_dir, exist_ok=True)

    labels = np.array(labels)
    genres = sorted(set(labels))

    # Scale before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(n_components, X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\n" + "=" * 50)
    print("PCA EXPLAINED VARIANCE")
    print("=" * 50)
    for i, (var, cum) in enumerate(zip(explained, cumulative), 1):
        print(f"  PC{i}: {var:.4f}  (cumulative: {cum:.4f})")

    # ---------------------------------------------------------------
    # Figure 1: 2D scatter with confidence ellipses
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for genre in genres:
        mask = labels == genre
        pts  = X_pca[mask, :2]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            label=genre,
            color=GENRE_COLORS.get(genre, "gray"),
            alpha=0.4, s=18, linewidths=0,
        )
        # Confidence ellipse (1 SD)
        _confidence_ellipse(pts, ax, color=GENRE_COLORS.get(genre, "gray"), n_std=1.5)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    ax.set_title("PCA of structural features by genre")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_scatter.png"), dpi=150)
    plt.close()
    logger.info("PCA scatter plot saved")

    # ---------------------------------------------------------------
    # Figure 2: Scree plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n_comp + 1), explained * 100, color="#534AB7", alpha=0.7, label="Individual")
    ax.plot(range(1, n_comp + 1), cumulative * 100, "o-", color="#993C1D", label="Cumulative")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, label="80% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_scree.png"), dpi=150)
    plt.close()
    logger.info("PCA scree plot saved")

    # ---------------------------------------------------------------
    # Figure 3: Feature loadings (biplot for PC1 vs PC2)
    # ---------------------------------------------------------------
    if feature_names and len(feature_names) == X.shape[1]:
        loadings = pca.components_[:2].T  # shape: (n_features, 2)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(loadings[:, 0], loadings[:, 1], s=30, color="#0F6E56", alpha=0.7)
        for i, name in enumerate(feature_names):
            ax.annotate(name, (loadings[i, 0], loadings[i, 1]),
                        fontsize=7, ha="center", va="bottom")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("PC1 loading")
        ax.set_ylabel("PC2 loading")
        ax.set_title("Feature loadings on PC1 and PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "pca_loadings.png"), dpi=150)
        plt.close()
        logger.info("PCA loadings plot saved")

    return X_pca


def _confidence_ellipse(pts: np.ndarray, ax, color: str, n_std: float = 1.5):
    """Draw a covariance ellipse for a set of 2D points."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if len(pts) < 3:
        return

    cov = np.cov(pts[:, 0], pts[:, 1])
    if np.any(np.isnan(cov)):
        return

    pearson = cov[0, 1] / max(np.sqrt(cov[0, 0] * cov[1, 1]), 1e-9)
    pearson = np.clip(pearson, -1, 1)

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        edgecolor=color,
        facecolor="none",
        linewidth=1.5,
        linestyle="--",
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x  = np.mean(pts[:, 0])
    mean_y  = np.mean(pts[:, 1])

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
