import numpy as np
from scipy.sparse import csr_matrix
import igraph as ig
import leidenalg


def csr_to_igraph(adj_matrix: csr_matrix) -> ig.Graph:
    """
    Convert a symmetric sparse adjacency matrix to an undirected igraph graph.
    """
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = adj_matrix.tocsr()

    # Keep only upper triangle to avoid adding edges twice
    coo = adj_matrix.tocoo()
    edges = [
        (int(i), int(j))
        for i, j, v in zip(coo.row, coo.col, coo.data)
        if i < j and v != 0
    ]

    g = ig.Graph(n=adj_matrix.shape[0], edges=edges, directed=False)
    return g


def leiden_cluster_topology_graph(
    adj_matrix: csr_matrix,
    resolution: float = 1.0,
    random_state: int = 42,
    use_weights: bool = False,
) -> np.ndarray:
    """
    Run Leiden clustering on a sparse topology adjacency graph.

    Returns
    -------
    labels : np.ndarray of shape (n_nodes,)
        Community label for each topology.
    """
    if not isinstance(adj_matrix, csr_matrix):
        adj_matrix = adj_matrix.tocsr()

    if use_weights:
        coo = adj_matrix.tocoo()
        edges = []
        weights = []
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j and v != 0:
                edges.append((int(i), int(j)))
                weights.append(float(v))

        g = ig.Graph(n=adj_matrix.shape[0], edges=edges, directed=False)
        g.es["weight"] = weights
        partition = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
            n_iterations=-1
        )
    else:
        g = csr_to_igraph(adj_matrix)
        partition = leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            resolution_parameter=resolution,
            seed=random_state,
            n_iterations=-1
        )

    return np.asarray(partition.membership, dtype=np.int32)
