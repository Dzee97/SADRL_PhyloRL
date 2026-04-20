import numpy as np
from itertools import combinations
from copy import deepcopy
from tqdm import tqdm


def _is_compatible(mask1: int, mask2: int, full_mask: int) -> bool:
    comp1 = full_mask ^ mask1
    comp2 = full_mask ^ mask2
    return (
        (mask1 & mask2) == 0
        or (mask1 & comp2) == 0
        or (comp1 & mask2) == 0
        or (comp1 & comp2) == 0
    )


def _bin(mask: int, num_taxa: int) -> str:
    return format(mask, f'0{num_taxa}b')


def split_size(split: int, num_taxa: int) -> int:
    k = split.bit_count()
    return min(k, num_taxa - k)


def generate_all_internal_splits(num_taxa: int) -> tuple[np.ndarray, np.ndarray]:
    splits = [m for m in range(1, 1 << (num_taxa - 1))
              if 2 <= m.bit_count() <= num_taxa - 2]
    split_sizes = [split_size(s, num_taxa) for s in splits]
    return np.asarray(splits), np.asarray(split_sizes)


def calculate_split_compatibility(all_splits: np.ndarray, num_taxa: int) -> np.ndarray:
    full_mask = (1 << num_taxa) - 1
    num_splits = len(all_splits)
    compat = np.zeros((num_splits, num_splits), dtype=bool)

    for i in range(num_splits):
        for j in range(i + 1, num_splits):
            if _is_compatible(all_splits[i], all_splits[j], full_mask):
                compat[i, j] = True
                compat[j, i] = True

    return compat


def generate_all_topologies(all_splits: np.ndarray, compat: np.ndarray, num_taxa: int) -> tuple[np.ndarray, np.ndarray]:
    num_splits = len(all_splits)
    topologies = []
    topologies_splits = []
    current: list[int] = []
    target_splits = num_taxa - 3

    def backtrack(start: int) -> None:
        if len(current) == target_splits:
            topologies.append([all_splits[idx] for idx in current])
            splits = np.zeros(num_splits, dtype=bool)
            for idx in current:
                splits[idx] = True
            topologies_splits.append(splits)
            return

        for i in range(start, num_splits):
            if all(compat[i, j] for j in current):
                current.append(i)
                backtrack(i + 1)
                current.pop()

    backtrack(0)

    return np.asarray(topologies), np.asarray(topologies_splits)


def full_clade_set(internal_masks: np.ndarray, num_taxa: int) -> set[int]:
    full_mask = (1 << num_taxa) - 1
    clades = set()
    for i in range(num_taxa):
        m = 1 << i
        clades.add(m)
        clades.add(full_mask ^ m)
    for m in internal_masks:
        m = int(m)
        clades.add(m)
        clades.add(full_mask ^ m)
    return clades


def generate_clade_adjacency_mapping(internal_masks: np.ndarray, num_taxa: int) -> dict[int, set[int]]:
    all_clades = full_clade_set(internal_masks, num_taxa)
    full_mask = (1 << num_taxa) - 1
    adj_map = {}

    def map_clade_neighbors(clade: int, option_clades: set[int]):
        subs = {c for c in option_clades if c != clade and (c & clade == c)}
        neighbors = {c for c in subs if not any((c != d and (c & d) == c) for d in subs)}
        adj_map[clade] = neighbors

        for c in neighbors:
            if c not in adj_map:
                map_clade_neighbors(c, subs)

        comp = full_mask ^ clade
        if comp not in adj_map:
            map_clade_neighbors(comp, all_clades - subs)

    start_clade = int(internal_masks[0])
    map_clade_neighbors(start_clade, all_clades)
    return adj_map


def find_center_clades(adj_map: dict[int, set[int]], num_taxa: int):
    full_mask = (1 << num_taxa) - 1
    adj_map = deepcopy(adj_map)
    leaves = {1 << i for i in range(num_taxa)}

    def prune_leaves(leaves: set[int]):
        new_leaves = set()
        for c in leaves:
            c_comp = full_mask ^ c
            for n in adj_map[c_comp]:
                n_comp = full_mask ^ n
                adj_map[n_comp].remove(c)
                if not adj_map[n_comp]:
                    new_leaves.add(n_comp)
            adj_map[c_comp].clear()
        return new_leaves

    while True:
        new_leaves = prune_leaves(leaves)
        if len(new_leaves) <= 1:
            break
        leaves = new_leaves

    # if len(leaves) < 3:
    #    final = full_mask
    #    for c in leaves:
    #        final ^= c
    #    leaves.add(final)

    return leaves


def rooted_shape_signature(clade: int, adj_map: dict[int, set[int]]):
    if not adj_map[clade]:
        return None

    children_sig = (rooted_shape_signature(n, adj_map) for n in adj_map[clade])
    return tuple(sorted(children_sig, key=lambda x: str(x)))


def center_shape_signature(center_clades: set[int], adj_map: dict[int, set[int]]):
    tripartition_sig = (rooted_shape_signature(c, adj_map) for c in center_clades)
    return tuple(sorted(tripartition_sig, key=lambda x: str(x)))


def rooted_remainder_parent_map(remainder: int, remainder_clades: set[int]) -> tuple[dict[int, int], set[int]]:
    family = remainder_clades | {remainder}

    parent_map = {}
    nontrivial_regrafts = set()
    for x in remainder_clades:
        supersets = [y for y in family if y != x and (x & y) == x]
        parent = min(supersets, key=lambda y: (y.bit_count(), y))
        parent_map[x] = parent
        if parent != remainder:
            nontrivial_regrafts.add(x)

    return parent_map, nontrivial_regrafts


def partition_clade_set(all_clades: set[int], prune: int, num_taxa: int):
    full_mask = (1 << num_taxa) - 1
    remainder = full_mask ^ prune

    prune_clades: set[int] = set()
    remainder_clades: set[int] = set()
    for c in all_clades:
        if c != prune and (c & prune) == c:
            prune_clades.add(c)
        if c != remainder and (c & remainder) == c:
            remainder_clades.add(c)

    return prune_clades, remainder, remainder_clades


def spr_move_from_clades(parent_map: dict[int, int],
                         prune: int,
                         prune_clades: set[int],
                         remainder: int,
                         remainder_clades: set[int],
                         regraft: int,
                         ) -> set[int]:
    # ancestor chain of regraft edge inside the rooted remainder
    ancestors: set[int] = set()
    y = parent_map[regraft]
    while y != remainder:
        ancestors.add(y)
        y = parent_map[y]

    # keep the prune edge and new graft edge clade
    new_clades = {prune, regraft | prune}
    # keep all clades fully inside prune edge
    new_clades.update(prune_clades)
    # keep remainder clades not on ancestor path
    for c in remainder_clades:
        if c in ancestors:
            new_clades.add(c | prune)
        else:
            new_clades.add(c)

    return new_clades


def canonical_internal_mask_from_clades(clades: set[int], num_taxa: int) -> np.ndarray:
    full_mask = (1 << num_taxa) - 1
    out = []

    for c in clades:
        size = c.bit_count()
        if 2 <= size <= num_taxa - 2:
            # canonical side = side not containing last taxon
            if (c >> (num_taxa - 1)) & 1:
                c = full_mask ^ c
            out.append(c)

    return np.asarray(sorted(set(out)))


def maximal_children(mask: int, clades: set[int]):
    subs = [c for c in clades if c != mask and (c & mask) == c]

    children = []
    for c in subs:
        if not any((c != d and (c & d) == c and (d & mask) == d) for d in subs):
            children.append(c)

    return children


def tree_key(x):
    if isinstance(x, int):
        return (0, x)
    return (1, tuple(tree_key(ch) for ch in x))


def pair_key(pair):
    unlabeled, labeled = pair
    return (tree_key(unlabeled), tree_key(labeled))


def encode_clade_pair(mask: int, children_map: dict[int, list[int]], cache: dict[int, tuple]):
    """
    Return (unlabeled_subtree, labeled_subtree) for one rooted clade.
    Uses memoization for speed.
    """
    if mask in cache:
        return cache[mask]

    if mask.bit_count() == 1:
        leaf = mask.bit_length() - 1
        result = (1, leaf)
        cache[mask] = result
        return result

    children = children_map.get(mask, [])

    if len(children) != 2:
        raise ValueError(f"Unexpected structure for clade {mask:b}: {children}")

    a_unl, a_lab = encode_clade_pair(children[0], children_map, cache)
    b_unl, b_lab = encode_clade_pair(children[1], children_map, cache)

    children_pairs = sorted(
        [(a_unl, a_lab), (b_unl, b_lab)],
        key=pair_key
    )

    unlabeled = tuple(ch[0] for ch in children_pairs)
    labeled = tuple(ch[1] for ch in children_pairs)

    result = (unlabeled, labeled)
    cache[mask] = result
    return result


def canonical_tree_from_clade_set(clades: set[int], num_taxa: int):
    full_mask = (1 << num_taxa) - 1

    children_map = {mask: maximal_children(mask, clades) for mask in clades}
    cache = {}

    candidates = []

    for m in clades:
        if m == full_mask or m.bit_count() < 2:
            continue

        children = children_map.get(m, [])

        if len(children) == 2 and (children[0] | children[1]) == m:
            a_unl, a_lab = encode_clade_pair(children[0], children_map, cache)
            b_unl, b_lab = encode_clade_pair(children[1], children_map, cache)
            c_unl, c_lab = encode_clade_pair(full_mask ^ m, children_map, cache)

            legs = sorted(
                [(a_unl, a_lab), (b_unl, b_lab), (c_unl, c_lab)],
                key=pair_key
            )

            unlabeled = tuple(x[0] for x in legs)
            labeled = tuple(x[1] for x in legs)

            candidates.append((unlabeled, labeled))

    if not candidates:
        raise ValueError("Could not construct a valid top-level tripartition.")

    return min(candidates, key=pair_key)


def newick_from_labeled_tree(tree, taxa_names: list[str]) -> str:
    def rec(node):
        if isinstance(node, int):
            return taxa_names[node]
        return "(" + ",".join(rec(ch) for ch in node) + ")"

    return rec(tree) + ";"


def pair_mask(i: int, j: int) -> int:
    return (1 << i) | (1 << j)


def enumerate_quartet_resolutions(num_taxa: int):
    quartets = []
    resolutions = []
    quartet_col_ranges = []

    col = 0
    for a, b, c, d in combinations(range(num_taxa), 4):
        quartets.append((a, b, c, d))

        ab = pair_mask(a, b)
        ac = pair_mask(a, c)
        ad = pair_mask(a, d)
        bc = pair_mask(b, c)
        bd = pair_mask(b, d)
        cd = pair_mask(c, d)

        resolutions.extend([
            (ab, cd),  # ab|cd
            (ac, bd),  # ac|bd
            (ad, bc),  # ad|bc
        ])
        quartet_col_ranges.append((col, col + 3))
        col += 3

    return quartets, resolutions, quartet_col_ranges


def split_displays_quartet_resolution(split_mask: int, Pmask: int, Qmask: int) -> bool:
    split_mask = int(split_mask)

    return (
        ((split_mask & Pmask) == Pmask and (split_mask & Qmask) == 0)
        or
        ((split_mask & Qmask) == Qmask and (split_mask & Pmask) == 0)
    )


def caclulate_topology_quartet_resolutions(topology: np.ndarray, resolutions: list, quartet_col_ranges: list):
    num_cols = len(resolutions)
    M = np.zeros(num_cols, dtype=bool)

    for q_idx, (start, end) in enumerate(quartet_col_ranges):
        found = False

        for col in range(start, end):
            Pmask, Qmask = resolutions[col]

            if any(split_displays_quartet_resolution(m, Pmask, Qmask) for m in topology):
                M[col] = True
                found = True
                break

        if not found:
            raise ValueError(
                f"Topology {i} did not resolve quartet {q_idx}; "
                "unexpected for a fully resolved binary tree."
            )

    return M


def calculate_topology_quartet_matrix(topologies: np.ndarray, num_taxa: int) -> np.ndarray:
    _, resolutions, quartet_col_ranges = enumerate_quartet_resolutions(num_taxa)

    num_topologies = len(topologies)
    num_cols = len(resolutions)
    M = np.zeros((num_topologies, num_cols), dtype=bool)

    for i, topo in enumerate(topologies):
        for q_idx, (start, end) in enumerate(quartet_col_ranges):
            found = False

            for col in range(start, end):
                Pmask, Qmask = resolutions[col]

                if any(split_displays_quartet_resolution(m, Pmask, Qmask) for m in topo):
                    M[i, col] = True
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Topology {i} did not resolve quartet {q_idx}; "
                    "unexpected for a fully resolved binary tree."
                )

    return M


if __name__ == "__main__":
    num_taxa = 9
    splits, split_sizes = generate_all_internal_splits(9)
    compat = calculate_split_compatibility(splits, num_taxa)
    topologies, topologies_splits = generate_all_topologies(splits, compat, num_taxa)
    shape_sig_set = set()
    for topo in tqdm(topologies):
        adj_map = generate_clade_adjacency_mapping(topo, num_taxa)
        center_clades = find_center_clades(adj_map, num_taxa)
        shape_sig = center_shape_signature(center_clades, adj_map)
        shape_sig_set.add(shape_sig)
    print(len(shape_sig_set))
