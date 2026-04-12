import numpy as np
from scipy.special import factorial2
from scipy.sparse import csr_matrix
from tqdm import tqdm
from umap import UMAP
import warnings
from tree_space_spr import generate_all_internal_splits, calculate_split_compatibility, generate_all_topologies, \
    full_clade_set, partition_clade_set, rooted_remainder_parent_map, spr_move_from_clades, \
    canonical_internal_mask_from_clades, canonical_tree_from_clade_set, newick_from_labeled_tree, \
    topology_quartet_matrix
from tree_space_eval import prepare_alignment, evaluate_or_reuse_likelihoods
from tree_space_umap import evaluate_or_reuse_umap
from tree_space_plot import plot_umap_heatmap
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
topology_split_var = masked_sizes.var(axis=1)

# topology_umap1d: np.ndarray | Any = UMAP(n_neighbors=50, n_components=3).fit_transform(topology_split_sizes)
# col_min = topology_umap1d.min(axis=0)
# col_max = topology_umap1d.max(axis=0)
# topology_umap1d_scaled = (topology_umap1d - col_min) / (col_max - col_min)

quartet_matrix = topology_quartet_matrix(topologies, num_taxa)

assert np.all(quartet_matrix.sum(axis=1) == math.comb(num_taxa, 4))

topology_quartet_umap: np.ndarray | Any = UMAP(
    n_neighbors=50, n_components=3, metric="hamming").fit_transform(quartet_matrix)
col_min = topology_quartet_umap.min(axis=0)
col_max = topology_quartet_umap.max(axis=0)
topology_quartet_rgb = (topology_quartet_umap - col_min) / (col_max - col_min)

# -----------------------------------------
# - Deriving trees from topology bitmasks -
# -----------------------------------------

print("\n#4 Deriving labeled trees, unlabeled shapes and newick representations from topologies")
topologies_unlabeled_shapes = []
topologies_labeled_trees = []
topologies_newicks: list[str] = []

for topo in tqdm(topologies, total=num_topologies):
    topo_clades = full_clade_set(topo, num_taxa)
    unlabeled_shape, labeled_tree = canonical_tree_from_clade_set(topo_clades, num_taxa)

    topologies_unlabeled_shapes.append(unlabeled_shape)
    topologies_labeled_trees.append(labeled_tree)

    newick = newick_from_labeled_tree(labeled_tree, taxa_names)

    topologies_newicks.append(newick)

shape_to_id = {}
topology_shape_ids = np.empty(num_topologies, dtype=np.int32)

for i, shape in enumerate(topologies_unlabeled_shapes):
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
    eval_dir, raxmlng_path, records, model_path, topologies_newicks, threads)
if reusable:
    print("• Found existing likelihood calculations to reuse")

# ------------------------------------------------------
# - Calculating RF distances to ML topology -
# ------------------------------------------------------
print("\n#5 Calculating Robinson–Foulds distances for each topology to ML topology")
ml_idx = np.argmax(likelihoods)
ml_topology_splits = topology_splits[ml_idx]
ml_topology_split_sizes = topology_split_sizes[ml_idx]

topology_rf_to_ml = np.sum(ml_topology_splits != topology_splits, axis=1)
topology_ed_to_ml = np.sqrt(np.sum((ml_topology_split_sizes - topology_split_sizes)**2, axis=1))


# -----------------------------------------------
# - Calculating SPR neighborhood for topologies -
# -----------------------------------------------

topology_index = {tuple(topo): i for i, topo in enumerate(topologies)}


def calculate_spr_neighborhood(i: int):
    topo = topologies[i]
    ll_i = likelihoods[i]

    topo_clades = full_clade_set(topo, num_taxa)

    total_moves = 0
    neighbor_scores: dict[int, float] = {}

    adj_entries = {}
    delta_ll_entries = {}

    for prune in topo_clades:
        prune_clades, remainder, remainder_clades = partition_clade_set(topo_clades, prune, num_taxa)
        parent_map, nontrivial_regrafts = rooted_remainder_parent_map(remainder, remainder_clades)
        total_moves += len(nontrivial_regrafts)

        for regraft in nontrivial_regrafts:
            new_clades = spr_move_from_clades(parent_map, prune, prune_clades, remainder, remainder_clades, regraft)
            new_top = canonical_internal_mask_from_clades(new_clades, num_taxa)

            j = topology_index[tuple(new_top)]

            adj_entries[(i, j)] = 1.0
            adj_entries[(j, i)] = 1.0

            delta_ll = ll_i - likelihoods[j]
            neighbor_scores[j] = delta_ll

            val = 1.0 + np.log1p(abs(delta_ll))
            delta_ll_entries[(i, j)] = val
            delta_ll_entries[(j, i)] = val

    total_neighbors = len(neighbor_scores)

    assert total_moves == 4 * (num_taxa - 3) * (num_taxa - 2)
    assert total_neighbors == 2 * (num_taxa - 3) * (2 * num_taxa - 7)

    return adj_entries, delta_ll_entries, total_moves, total_neighbors


print("\n#5 Calculating SPR neighborhood for each topology")
with Pool(int(threads)) as p:
    results = list(tqdm(p.imap(calculate_spr_neighborhood, range(num_topologies)), total=num_topologies))

total_adj_entries, total_delta_ll_entries = {}, {}

for adj_entries, delta_ll_entries, total_moves, total_neighbors in results:
    for row_col, val in adj_entries.items():
        total_adj_entries[row_col] = val
    for row_col, val in delta_ll_entries.items():
        total_delta_ll_entries[row_col] = val

adj_rows, adj_cols, adj_vals = [], [], []
delta_rows, delta_cols, delta_vals = [], [], []

for (i, j), val in total_adj_entries.items():
    adj_rows.append(i)
    adj_cols.append(j)
    adj_vals.append(val)

for (i, j), val in total_delta_ll_entries.items():
    delta_rows.append(i)
    delta_cols.append(j)
    delta_vals.append(val)

topology_adj_matrix = csr_matrix(
    (adj_vals, (adj_rows, adj_cols)),
    shape=(num_topologies, num_topologies),
    dtype=np.float32
)

topology_delta_ll_matrix = csr_matrix(
    (delta_vals, (delta_rows, delta_cols)),
    shape=(num_topologies, num_topologies),
    dtype=np.float32
)

# --------------------------------------------
# - Creating SPR graph embeddings using UMAP -
# --------------------------------------------

print("\n#6 Creating 2D SPR graph embeddings using UMAP")
random_state = 42
min_dist = 0.1

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    print(f"• n_neighbors={total_neighbors} - Creating/reusing embedding for adjacency matrix")
    embedding_adj, reusable = evaluate_or_reuse_umap(
        cache_dir=umap_cache_dir,
        matrix=topology_adj_matrix,
        n_neighbors=total_neighbors,
        metric="precomputed",
        random_state=random_state,
        min_dist=min_dist
    )
    if reusable:
        print("  ↳ Reused cached adjacency embedding")

    print(f"• n_neighbors={total_neighbors} - Creating/reusing embedding for delta ll matrix")
    embedding_delta_ll, reusable = evaluate_or_reuse_umap(
        cache_dir=umap_cache_dir,
        matrix=topology_delta_ll_matrix,
        n_neighbors=total_neighbors,
        metric="precomputed",
        random_state=random_state,
        min_dist=min_dist
    )
    if reusable:
        print("  ↳ Reused cached delta-ll embedding")

# --------------------
# - Plotting results -
# --------------------

print("\n#7 Plotting results")
figsize = (10, 8)
point_size = 4.0
alpha = 0.7
colorbar_label = "Log-likelihood",

save_path = f"umap_adj_nn{total_neighbors}.png"
title = "Likelihood landscape (Uniform distance)"
plot_umap_heatmap(embedding_adj, likelihoods, save_path, title)

save_path = f"umap_delta_ll_nn{total_neighbors}.png"
title = "Likelihood landscape (Likelihood weighted distance)"
plot_umap_heatmap(embedding_delta_ll, likelihoods, save_path, title)


plot_umap_heatmap(
    embedding_adj,
    topology_cherry_count,
    save_path=f"umap_adj_cherries_nn{total_neighbors}.png",
    title="SPR graph UMAP colored by cherry count",
    colorbar_label="Cherry count"
)

plot_umap_heatmap(
    embedding_adj,
    topology_split_mean,
    save_path=f"umap_adj_split_mean_nn{total_neighbors}.png",
    title="SPR graph UMAP colored by mean split size",
    colorbar_label="Mean split size"
)

plot_umap_heatmap(
    embedding_adj,
    topology_split_var,
    save_path=f"umap_adj_split_var_nn{total_neighbors}.png",
    title="SPR graph UMAP colored by split size variance",
    colorbar_label="Split size variance"
)

plot_umap_heatmap(
    embedding_adj,
    -topology_rf_to_ml,
    save_path=f"umap_adj_rf_to_ml_nn{total_neighbors}.png",
    title="SPR graph UMAP colored by RF distance to ML topology",
    colorbar_label="RF distance to ML tree",
)

plot_umap_heatmap(
    embedding_adj,
    -topology_ed_to_ml,
    save_path=f"umap_adj_ed_to_ml_nn{total_neighbors}.png",
    title="SPR graph UMAP colored by split size Euclidean distance to ML topology",
    colorbar_label="Euclidean distance to ML tree",
)

plot_umap_heatmap(
    embedding_adj,
    topology_quartet_rgb,
    save_path=f"umap_adj_hamming1d_nn{total_neighbors}.png",
    title="SPR graph UMAP colored global RF distance embedding",
    colorbar_label="RF distance embedding",
    cmap=None,
)
