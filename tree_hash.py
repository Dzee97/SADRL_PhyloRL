from ete3 import Tree
import hashlib
from collections import defaultdict


def _combine_hashes(child_hashes):
    """Combine child hashes deterministically (order-invariant)."""
    child_hashes.sort()
    combined = "(" + ",".join(child_hashes) + ")"
    return hashlib.sha256(combined.encode()).hexdigest()


def _dfs_hash(node, parent, adj, is_leaf, leaf_name, memo):
    """Compute subtree hash for the part of the tree rooted at node, excluding parent."""
    key = (node, parent)
    if key in memo:
        return memo[key]

    if is_leaf[node]:
        h = hashlib.sha256(leaf_name[node].encode()).hexdigest()
    else:
        child_hashes = []
        for child in adj[node]:
            if child != parent:
                child_hashes.append(_dfs_hash(child, node, adj, is_leaf, leaf_name, memo))
        h = _combine_hashes(child_hashes)

    memo[key] = h
    return h


def unrooted_tree_hash(tree: Tree):
    """
    Compute a canonical, root- and order-invariant hash for an unrooted ETE3 tree,
    based only on leaf names.
    """
    # Build adjacency and leaf name tables
    nodes = list(tree.traverse())
    idx_map = {n: i for i, n in enumerate(nodes)}
    adj = defaultdict(list)
    is_leaf = {}
    leaf_name = {}

    for n in nodes:
        i = idx_map[n]
        if n.is_leaf():
            is_leaf[i] = True
            leaf_name[i] = n.name
        else:
            is_leaf[i] = False
            leaf_name[i] = ""  # ignored
        for c in n.children:
            j = idx_map[c]
            adj[i].append(j)
            adj[j].append(i)

    # Compute hashes for both directions of each edge
    memo = {}
    edge_hashes = []
    for u in adj:
        for v in adj[u]:
            if u < v:  # avoid duplicates
                h1 = _dfs_hash(u, v, adj, is_leaf, leaf_name, memo)
                h2 = _dfs_hash(v, u, adj, is_leaf, leaf_name, memo)
                edge_hash = _combine_hashes([h1, h2])
                edge_hashes.append(edge_hash)

    return min(edge_hashes)


if __name__ == "__main__":
    t_Int0 = Tree("(B:2,C:4,((D:5,(E:6,F:7)Int2:8)Int3:9,(G:3,A:1)Int1:10)Int4:11)Int0;", format=1)
    t_Int1 = Tree("(G:3,A:1,((B:2,C:4)Int0:11,(D:5,(E:6,F:7)Int2:8)Int3:9)Int4:10)Int1;", format=1)
    t_Int2 = Tree("(E:6,F:7,(D:5,((B:2,C:4)Int0:11,(G:3,A:1)Int1:10)Int4:9)Int3:8)Int2;", format=1)
    t_Int3 = Tree("(D:5,(E:6,F:7)Int2:8,((B:2,C:4)Int0:11,(G:3,A:1)Int1:10)Int4:9)Int3;", format=1)
    t_Int4 = Tree("((B:2,C:4)Int0:11,(G:3,A:1)Int1:10,(D:5,(E:6,F:7)Int2:8)Int3:9)Int4;", format=1)

    t_Int5 = Tree("((B,C),(G,A),(D,(E,F)));", format=1)
    t_Int5 = Tree("((C,B),(G,A),(D,(E,F)));", format=1)

    print(unrooted_tree_hash(t_Int0))
    print(unrooted_tree_hash(t_Int1))
    print(unrooted_tree_hash(t_Int2))
    print(unrooted_tree_hash(t_Int3))
    print(unrooted_tree_hash(t_Int4))
    print(unrooted_tree_hash(t_Int5))
