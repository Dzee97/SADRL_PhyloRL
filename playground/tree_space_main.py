import numpy as np
from scipy.special import factorial2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm
from umap import UMAP
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import warnings
from tree_space_spr import generate_all_internal_splits, calculate_split_compatibility, generate_all_topologies, \
    full_clade_set, partition_clade_set, rooted_remainder_parent_map, spr_move_from_clades, \
    canonical_internal_mask_from_clades, canonical_tree_from_clade_set, newick_from_labeled_tree, \
    calculate_topology_quartet_matrix, enumerate_quartet_resolutions, split_displays_quartet_resolution
from tree_space_eval import prepare_alignment, evaluate_or_reuse_likelihoods
from tree_space_umap import evaluate_or_reuse_umap, embedding_to_rgb
from tree_space_plot import plot_umap_heatmap, plot_umap_rgb, plot_umap_categories
from tree_space_helper import print_memory_stats
from typing import Any
from multiprocessing import Pool
import math

eval_dir = "sample_001/eval"
umap_cache_dir = "sample_001/umap_cache"
alignment_path = "sample_001/sample_aln.fasta"
model_path = "sample_001/raxml_eval_pars.raxml.bestModel"
raxmlng_path = "../dependencies/raxmlng/raxml-ng"
threads = "8"

# --------------------------
# - Reading alignment file -
# --------------------------

print(f"\n#1 Reading alignment file: {alignment_path}")
taxa_names, records = prepare_alignment(alignment_path)
num_taxa = len(taxa_names)
print(f"• Alignment file contains {num_taxa} taxa")

# -------------------------------------
# - Generating topologies as bitmasks -
# -------------------------------------

print(f"\n#2 Generating topologies for {num_taxa} taxa")

print("• Generating all possible internal splits")
all_internal, split_sizes = generate_all_internal_splits(num_taxa)

print("• Calculating pairwise compatibility between all internal split")
compat_matrix = calculate_split_compatibility(all_internal, num_taxa)

print("• Generating all possible tree topologies")
topologies, topology_splits = generate_all_topologies(all_internal, compat_matrix, num_taxa)
topology_split_sizes = split_sizes * topology_splits

num_topologies = len(topologies)
print(f"• Number of generated topologies: {num_topologies}")
assert num_topologies == factorial2(2 * num_taxa - 5, exact=True)

# -------------------------------------------------------
# - Calculating balance measures from topology bitmasks -
# -------------------------------------------------------

print("\n#3 Calculating balance measures from topology bitmasks")

topology_cherry_count = np.sum(topology_split_sizes == 2, axis=1)
masked_sizes = np.ma.masked_equal(topology_split_sizes, 0)
topology_split_mean = masked_sizes.mean(axis=1)

# -------------------------------------------------------
# - Calculating quartet matrix from topology bitmasks -
# -------------------------------------------------------

print("\n#4 Calculating quartet matrix from topology bitmasks")

quartets, resolutions, quartet_col_ranges = enumerate_quartet_resolutions(num_taxa)
num_cols = len(resolutions)


def caclulate_topology_quartet_resolutions(i: int):
    topo = topologies[i]
    M = np.zeros(num_cols, dtype=bool)

    for q_idx, (start, end) in enumerate(quartet_col_ranges):
        found = False

        for col in range(start, end):
            Pmask, Qmask = resolutions[col]

            if any(split_displays_quartet_resolution(m, Pmask, Qmask) for m in topo):
                M[col] = True
                found = True
                break

        if not found:
            raise ValueError(
                f"Topology {i} did not resolve quartet {q_idx}; "
                "unexpected for a fully resolved binary tree."
            )

    return M


topology_quartet_matrix = np.empty((num_topologies, num_cols), dtype=bool)
with Pool(int(threads)) as p:
    for i, row_mask in enumerate(tqdm(p.imap(caclulate_topology_quartet_resolutions, range(num_topologies)),
                                      total=num_topologies)):
        topology_quartet_matrix[i, :] = row_mask

assert np.all(topology_quartet_matrix.sum(axis=1) == math.comb(num_taxa, 4))

# -----------------------------------------
# - Deriving trees from topology bitmasks -
# -----------------------------------------

print("\n#4 Deriving labeled trees, unlabeled shapes and newick representations from topologies")


def derive_tree_from_topology(i):
    topo = topologies[i]
    topo_clades = full_clade_set(topo, num_taxa)
    unlabeled_shape, labeled_tree = canonical_tree_from_clade_set(topo_clades, num_taxa)
    newick = newick_from_labeled_tree(labeled_tree, taxa_names)
    return unlabeled_shape, labeled_tree, newick


topology_labeled_trees = [None] * num_topologies
topology_newicks = [""] * num_topologies
topology_shape_ids = np.empty(num_topologies, dtype=np.int32)
shape_to_id = {}

with Pool(int(threads)) as p:
    for i, (shape, tree, newick) in enumerate(tqdm(p.imap(derive_tree_from_topology, range(num_topologies)),
                                                   total=num_topologies)):
        topology_labeled_trees[i] = tree
        topology_newicks[i] = newick

        if shape not in shape_to_id:
            shape_to_id[shape] = len(shape_to_id)
        topology_shape_ids[i] = shape_to_id[shape]

num_shapes = len(shape_to_id)
print(f"• Number of unique unlabeled shapes: {num_shapes}")

# ---------------------------------------------
# - Evaluating likelihood from topology trees -
# ---------------------------------------------

print("\n#4 Evaluating likelihood of all newick representations from topologies")
likelihoods, reusable = evaluate_or_reuse_likelihoods(
    eval_dir, raxmlng_path, records, model_path, topology_newicks, threads)
if reusable:
    print("• Found existing likelihood calculations to reuse")

# -------------------------------------------------------
# - Calculating RF and Quartet distances to ML topology -
# -------------------------------------------------------

print("\n#5 Calculating Robinson–Foulds and Quartet distances for each topology to ML topology")
ml_idx = np.argmax(likelihoods)
ml_topology_splits = topology_splits[ml_idx]
ml_topology_quartets = topology_quartet_matrix[ml_idx]

topology_rf_to_ml = np.sum(ml_topology_splits != topology_splits, axis=1)
topology_qd_to_ml = np.sum(ml_topology_quartets != topology_quartet_matrix, axis=1) // 2

# -----------------------------------------------
# - Calculating SPR neighborhood for topologies -
# -----------------------------------------------

topology_index = {tuple(topo): i for i, topo in enumerate(topologies)}


def calculate_spr_neighborhood(i: int):
    topo = topologies[i]

    topo_clades = full_clade_set(topo, num_taxa)

    total_moves = 0
    neighbor_set = set()

    for prune in topo_clades:
        prune_clades, remainder, remainder_clades = partition_clade_set(topo_clades, prune, num_taxa)
        parent_map, nontrivial_regrafts = rooted_remainder_parent_map(remainder, remainder_clades)
        total_moves += len(nontrivial_regrafts)

        for regraft in nontrivial_regrafts:
            new_clades = spr_move_from_clades(parent_map, prune, prune_clades, remainder, remainder_clades, regraft)
            new_top = canonical_internal_mask_from_clades(new_clades, num_taxa)

            j = topology_index[tuple(new_top)]

            neighbor_set.add((i, j))

    total_neighbors = len(neighbor_set)

    assert total_moves == 4 * (num_taxa - 3) * (num_taxa - 2)
    assert total_neighbors == 2 * (num_taxa - 3) * (2 * num_taxa - 7)

    return neighbor_set


def build_sparse_spr_adj_matrix(topology_neighbor_sets):
    num_topologies = len(topology_neighbor_sets)
    rows, cols = [], []

    for neighbor_set in topology_neighbor_sets:
        for i, j in neighbor_set:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(num_topologies, num_topologies))


print("\n#5 Calculating SPR neighborhood for each topology")
with Pool(int(threads)) as p:
    topology_neighbor_sets = list(tqdm(p.imap(calculate_spr_neighborhood, range(num_topologies)), total=num_topologies))

n_neighbors = len(topology_neighbor_sets[0])
topology_adj_matrix = build_sparse_spr_adj_matrix(topology_neighbor_sets)

topology_dist_to_ml = shortest_path(topology_adj_matrix, indices=ml_idx)


# --------------------------------------------
# - Creating SPR graph embeddings using UMAP -
# --------------------------------------------

print("\n#6 Creating 2D SPR graph embeddings using UMAP")
random_state = 42
min_dist = 0.0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    print(f"• n_neighbors={n_neighbors} - Creating/reusing embedding for adjacency matrix")
    embedding_adj, reusable = evaluate_or_reuse_umap(
        cache_dir=umap_cache_dir,
        matrix=topology_adj_matrix,
        n_neighbors=n_neighbors,
        metric="precomputed",
        random_state=random_state,
        min_dist=min_dist
    )
    if reusable:
        print("  ↳ Reused cached adjacency embedding")

# --------------------
# - Plotting results -
# --------------------

print("\n#7 Plotting results")
figsize = (10, 8)
point_size = 4.0
alpha = 0.7
colorbar_label = "Log-likelihood",

save_path = f"umap_adj_nn{n_neighbors}.png"
title = "Likelihood landscape (Uniform distance)"
plot_umap_heatmap(embedding_adj, likelihoods, save_path, title)


plot_umap_heatmap(
    embedding_adj,
    topology_cherry_count,
    save_path=f"umap_adj_cherries_nn{n_neighbors}.png",
    title="SPR graph UMAP colored by cherry count",
    colorbar_label="Cherry count"
)

plot_umap_heatmap(
    embedding_adj,
    topology_split_mean,
    save_path=f"umap_adj_split_mean_nn{n_neighbors}.png",
    title="SPR graph UMAP colored by mean split size",
    colorbar_label="Mean split size"
)

plot_umap_heatmap(
    embedding_adj,
    -topology_rf_to_ml,
    save_path=f"umap_adj_rf_to_ml_nn{n_neighbors}.png",
    title="SPR graph UMAP colored by RF distance to ML topology",
    colorbar_label="RF distance to ML tree",
)

plot_umap_heatmap(
    embedding_adj,
    -topology_qd_to_ml,
    save_path=f"umap_adj_qd_to_ml_nn{n_neighbors}.png",
    title="SPR graph UMAP colored by split size Quartet distance to ML topology",
    colorbar_label="Quartet distance to ML tree",
)

plot_umap_heatmap(
    embedding_adj,
    -topology_dist_to_ml,
    save_path=f"umap_adj_dist_to_ml_nn{n_neighbors}.png",
    title="SPR graph UMAP colored by SPR distance to ML topology",
    colorbar_label="SPR distance to ML tree",
)

print("\n#8 HDBSCAN clustering on SPR UMAP embedding")
hdbscan = HDBSCAN(min_cluster_size=5000, min_samples=5)
topology_cluster_labels = hdbscan.fit_predict(embedding_adj)
plot_umap_categories(
    embedding_adj,
    topology_cluster_labels,
    save_path="umap_adj_hdbscan.png",
    title="SPR graph UMAP colored by HDBSCAN clusters",
    cmap="tab20"
)

top_k = 50
n_clusters = topology_cluster_labels.max()


def get_top_k_medoid_indices(data, labels, k):
    cluster_top_indices = []
    for i in range(n_clusters):
        indices_in_cluster = np.where(labels == i)[0]
        if len(indices_in_cluster) == 0:
            continue

        cluster_points = data[indices_in_cluster]
        dist_matrix = pairwise_distances(cluster_points, metric='euclidean')

        # Sum distances for each point and find indices of the smallest k
        dist_sums = dist_matrix.sum(axis=1)
        # Use argsort to get indices of points from most central to least central
        # clip k in case cluster is smaller than k
        actual_k = min(k, len(indices_in_cluster))
        local_top_k_idxs = np.argsort(dist_sums)[:actual_k]

        global_top_k_idxs = indices_in_cluster[local_top_k_idxs]
        cluster_top_indices.append(global_top_k_idxs)

    return cluster_top_indices  # Returns a list of arrays


# 1. Get list of top k indices for every cluster
all_clusters_top_k = get_top_k_medoid_indices(embedding_adj, topology_cluster_labels, k=top_k)

# 2. Loop through each cluster and calculate the average QD
all_avg_topology_qd = []
for i, top_k_idxs in enumerate(all_clusters_top_k):
    # Initialize an array to accumulate distances
    # We use float64 to prevent precision issues during averaging
    total_qd_sum = np.zeros(topology_quartet_matrix.shape[0], dtype=np.float64)

    for med_idx in top_k_idxs:
        med_topology_quartets = topology_quartet_matrix[med_idx]
        # Calculate QD for this specific medoid
        qd = np.sum(med_topology_quartets != topology_quartet_matrix, axis=1) // 2
        total_qd_sum += qd

    # Calculate the average distance across all k medoids
    avg_topology_qd = total_qd_sum / len(top_k_idxs)
    all_avg_topology_qd.append(avg_topology_qd)

    plot_umap_heatmap(
        embedding_adj,
        -avg_topology_qd,  # Negative for heatmap styling if desired
        save_path=f"umap_adj_avg_qd_cluster{i}_k{top_k}.png",
        title=f"Avg QD to top {len(top_k_idxs)} medoids in Cluster {i}",
        colorbar_label="Average Quartet Distance",
    )

    avg_topology_dist = shortest_path(topology_adj_matrix, indices=top_k_idxs).mean(axis=0)

    plot_umap_heatmap(
        embedding_adj,
        -avg_topology_dist,  # Negative for heatmap styling if desired
        save_path=f"umap_adj_avg_dist_cluster{i}_k{top_k}.png",
        title=f"Avg SPR distance to top {len(top_k_idxs)} medoids in Cluster {i}",
        colorbar_label="Average SPR Distance",
    )
