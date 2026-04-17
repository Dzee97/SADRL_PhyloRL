import numpy as np
import matplotlib.pyplot as plt


def plot_umap_heatmap(embedding: np.ndarray, values: np.ndarray, save_path: str, title: str,
                      point_size: float = 4.0, alpha: float = 0.5, cmap: str | None = "YlGn",
                      colorbar_label: str = "Log_likelihood", figsize: tuple[float, float] = (10, 8)):
    # --- compute indicators ---
    max_ll = np.max(values)
    max_idx = np.argmax(values)

    near_mask = values >= (max_ll - 0.1)
    near_indices = np.where(near_mask)[0]

    # --- base plot ---
    plt.figure(figsize=figsize)

    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        # gridsize=100,
        s=point_size,
        alpha=alpha,
        cmap=cmap,
        edgecolors="none",
        # linewidths=0.2
    )

    # --- highlight near-optimal trees ---
    # plt.scatter(
    #    embedding[near_indices, 0],
    #    embedding[near_indices, 1],
    #    s=point_size * 6,
    #    facecolors="none",
    #    edgecolors="orange",
    #    linewidths=1.0,
    #    label="Within 0.1 of max",
    # )

    # --- highlight global maximum ---
    plt.scatter(
        embedding[max_idx, 0],
        embedding[max_idx, 1],
        s=point_size * 6,
        c="red",
        linewidths=1.0,
        label="Maximum",
    )

    # --- colorbar and labels ---
    plt.colorbar(sc, label=colorbar_label)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # --- legend ---
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_umap_categories(
    embedding: np.ndarray,
    categories: np.ndarray,
    save_path: str,
    title: str,
    figsize: tuple[float, float] = (10, 8),
    point_size: float = 4.0,
    alpha: float = 0.7,
    cmap: str = "tab10",
):
    if embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2).")
    if len(categories) != embedding.shape[0]:
        raise ValueError("categories must have same length as number of points.")

    plt.figure(figsize=figsize)

    unique_cats = np.unique(categories)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(unique_cats)))

    for color, cat in zip(colors, unique_cats):
        mask = categories == cat
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=alpha,
            color=color,
            edgecolors="none",
            label=f"Cluster {cat}",
        )

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(markerscale=3, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_umap_rgb(
    embedding: np.ndarray,
    rgb: np.ndarray,
    save_path: str,
    title: str,
    point_size: float = 4.0,
    alpha: float = 0.8,
    figsize: tuple[float, float] = (10, 8),
):
    if embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n_samples, 2).")
    if rgb.shape != (embedding.shape[0], 3):
        raise ValueError("rgb must have shape (n_samples, 3).")

    plt.figure(figsize=figsize)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=rgb,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
