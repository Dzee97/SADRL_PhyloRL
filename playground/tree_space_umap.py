import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from umap import UMAP


def _sha256_bytes(h, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())


def sparse_matrix_fingerprint(matrix: csr_matrix) -> str:
    """
    Stable fingerprint of a CSR sparse matrix based on shape + CSR arrays.
    """
    if not isinstance(matrix, csr_matrix):
        matrix = matrix.tocsr()

    h = hashlib.sha256()
    h.update(b"csr_matrix_v1")
    h.update(str(matrix.shape).encode("utf-8"))

    _sha256_bytes(h, matrix.data)
    _sha256_bytes(h, matrix.indices)
    _sha256_bytes(h, matrix.indptr)

    return h.hexdigest()


def umap_cache_key(
    matrix: csr_matrix,
    *,
    n_neighbors: int,
    metric: str,
    random_state,
    min_dist: float = 0.1,
    n_components: int = 2,
    init: str = "spectral",
    low_memory: bool = True,
) -> dict:
    """
    Build the metadata dict that defines a unique UMAP run.
    """
    return {
        "matrix_sha256": sparse_matrix_fingerprint(matrix),
        "n_neighbors": int(n_neighbors),
        "metric": metric,
        "random_state": random_state,
        "min_dist": float(min_dist),
        "n_components": int(n_components),
        "init": init,
        "low_memory": bool(low_memory),
    }


def evaluate_or_reuse_umap(
    cache_dir: str,
    matrix: csr_matrix,
    *,
    n_neighbors: int,
    metric: str = "precomputed",
    random_state=42,
    min_dist: float = 0.1,
    n_components: int = 2,
    init: str = "spectral",
    low_memory: bool = True,
    force: bool = False,
):
    """
    Reuse a cached UMAP embedding if the sparse matrix and parameters match.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = umap_cache_key(
        matrix,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        min_dist=min_dist,
        n_components=n_components,
        init=init,
        low_memory=low_memory,
    )

    key_json = json.dumps(meta, sort_keys=True)
    run_id = hashlib.sha256(key_json.encode("utf-8")).hexdigest()

    embedding_path = cache_dir / f"{run_id}.embedding.npy"
    meta_path = cache_dir / f"{run_id}.meta.json"

    if not force and embedding_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)

        if old_meta == meta:
            return np.load(embedding_path), True

    reducer = UMAP(
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        min_dist=min_dist,
        n_components=n_components,
        init=init,
        low_memory=low_memory,
    )
    embedding = reducer.fit_transform(matrix)

    np.save(embedding_path, embedding)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return embedding, False


def embedding_to_rgb(embedding_3d: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    rgb = np.empty_like(embedding_3d, dtype=np.float32)

    for k in range(3):
        x = embedding_3d[:, k]
        lo, hi = np.percentile(x, clip_percentiles)
        x = np.clip(x, lo, hi)

        if hi > lo:
            x = (x - lo) / (hi - lo)
        else:
            x = np.zeros_like(x)

        rgb[:, k] = x

    return rgb
