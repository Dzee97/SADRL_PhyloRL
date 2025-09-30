from ete3 import Tree
import math


class TreePreprocessor:
    def __init__(self, tree: Tree):
        self.tree = tree

        # Euler tour arrays
        self.euler = []
        self.depth = []
        self.first_occurrence = {}
        self.tin, self.tout = {}, {}
        self.time = 0

        # Metadata
        self.depth_length = {}
        self.depth_count = {}
        self.subtree_leaves = {}
        self.subtree_length = {}
        self.subtree_max_edge = {}
        self.subtree_split = {}

        # Edge set
        self.edges = []

        # Run preprocessing
        self._dfs(self.tree, None, 0, 0.0)
        self._build_sparse_table()

    def _dfs(self, node, parent, d, dist):
        """DFS that fills all metadata + Euler tour arrays"""
        self.tin[node] = self.time
        self.time += 1

        # Add edge if not root
        if parent is not None:
            self.edges.append((node, parent))

        # depth info
        self.depth_length[node] = dist
        self.depth_count[node] = d

        if node not in self.first_occurrence:
            self.first_occurrence[node] = len(self.euler)
        self.euler.append(node)
        self.depth.append(d)

        # Subtree metrics init
        leaves = 1 if node.is_leaf() else 0
        total_len = node.dist
        max_edge = node.dist
        split = {node.name} if node.is_leaf() else set()

        for child in node.children:
            self._dfs(child, node, d+1, dist + child.dist)
            self.euler.append(node)
            self.depth.append(d)

            # Update subtree metrics
            leaves += self.subtree_leaves[child]
            total_len += self.subtree_length[child]
            max_edge = max(max_edge, self.subtree_max_edge[child])
            split = split.union(self.subtree_split[child])

        self.subtree_leaves[node] = leaves
        self.subtree_length[node] = total_len
        self.subtree_max_edge[node] = max_edge
        self.subtree_split[node] = split

        self.tout[node] = self.time
        self.time += 1

    def _min_tree_split(self, split_set):
        all_set = self.subtree_split[self.tree]
        return split_set if len(split_set) <= self.subtree_leaves[self.tree] / 2 else all_set - split_set

    def _min_combined_split(self, split1: set, split2: set):
        if split1.issuperset(split2):
            combined_split = split1 - split2
        else:
            combined_split = split1.union(split2)
        return self._min_tree_split(combined_split)

    def _min_subtract_split(self, split1: set, split2: set):
        if split1.issuperset(split2):
            return self._min_tree_split(split1 - split2)
        elif split1.issubset(split2):
            return self._min_tree_split(split2 - split1)
        else:
            all_set = self.subtree_split[self.tree]
            split1_comp = all_set - split1
            return self._min_subtract_split(split1_comp, split2)

    def _build_sparse_table(self):
        """Builds sparse table for RMQ on depth array"""
        N = len(self.depth)
        self.log = [0] * (N+1)
        for i in range(2, N+1):
            self.log[i] = self.log[i//2] + 1

        K = self.log[N] + 1
        self.st = [[0]*N for _ in range(K)]
        for i in range(N):
            self.st[0][i] = i

        for k in range(1, K):
            for i in range(N - (1 << k) + 1):
                left = self.st[k-1][i]
                right = self.st[k-1][i + (1 << (k-1))]
                self.st[k][i] = left if self.depth[left] < self.depth[right] else right

    def _rmq(self, l, r):
        """Return index of min depth in [l, r]"""
        k = self.log[r - l + 1]
        left = self.st[k][l]
        right = self.st[k][r - (1 << k) + 1]
        return left if self.depth[left] < self.depth[right] else right

    def _max_edge_excluding_subtrees(self, node, exclude):
        """Return the max edge excluding subtrees"""
        max_edge = node.dist
        for child in node.children:
            if child in exclude:
                exclude.remove(child)
            elif any(self.is_descendant(e, child) for e in exclude):
                max_edge = max(max_edge, self._max_edge_excluding_subtrees(child, exclude))
            else:
                max_edge = max(max_edge, self.subtree_max_edge[child])
        return max_edge

    def find_lca(self, u, v):
        """Return lowest common ancestor of nodes u and v"""
        i, j = self.first_occurrence[u], self.first_occurrence[v]
        if i > j:
            i, j = j, i
        idx = self._rmq(i, j)
        return self.euler[idx]

    def is_descendant(self, u, v):
        """Check if u is in the subtree of v"""
        return self.tin[v] <= self.tin[u] and self.tout[u] <= self.tout[v]

    def get_possible_spr_moves(self, radius=None):
        """
        Generate all valid SPR moves within a given radius.
        Each move is (prune_edge, regraft_edge).
        """
        possible_moves = []

        for i, (u, up_u) in enumerate(self.edges):
            for j, (v, up_v) in enumerate(self.edges):
                # skip if edges are the same or they are connected
                if v == u or v == up_u or up_v == up_u or up_v == u:
                    continue

                # compute distance between edges (using child nodes)
                lca = self.find_lca(u, v)

                child_prune_node = u
                inner_prune_node = u if lca == u else up_u
                outer_prune_node = up_u if inner_prune_node == u else u

                child_regraft_node = v
                inner_regraft_node = v if lca == v else up_v
                outer_regraft_node = up_v if inner_regraft_node == v else v

                dist = self.depth_count[inner_prune_node] + \
                    self.depth_count[inner_regraft_node] - 2 * self.depth_count[lca]

                if radius is None or dist <= radius:
                    possible_moves.append(((inner_prune_node, outer_prune_node, child_prune_node),
                                          (inner_regraft_node, outer_regraft_node, child_regraft_node), lca))

        return possible_moves

    def extract_all_spr_features(self, possible_moves, split_support_nj=None, split_support_upgma=None):
        """
        1: Total branch length
        2: Longest branch
        3-4: Branch length of prune and regraft
        5: Topology distance between prune and regraft
        6: Branch length distance between prune and regraft
        7: New branch length due to pruning
        8-11: Number of leaves in four subtrees (prune subtree "b", remain subtree "c", regraft right "c1", regraft left "c2")
        12-15: Sum of branch lengths in four subtrees
        16-19: Longest branch in four subtrees
        """

        f1 = self.subtree_length[self.tree]
        f2 = self.subtree_max_edge[self.tree]

        def compute_move_specific_features(move):
            prune_edge, regraft_edge, lca = move

            inner_prune_node, outer_prune_node, child_prune_node = prune_edge
            inner_regraft_node, outer_regraft_node, child_regraft_node = regraft_edge

            f3 = child_prune_node.dist
            f4 = child_regraft_node.dist

            f5 = self.depth_count[inner_prune_node] + self.depth_count[inner_regraft_node] - 2 * self.depth_count[lca]
            f6 = self.depth_length[inner_prune_node] + \
                self.depth_length[inner_regraft_node] - 2 * self.depth_length[lca]

            sum_other_children = sum(child.dist for child in inner_prune_node.children if child != child_prune_node)
            f7 = sum_other_children + inner_prune_node.dist if inner_prune_node != child_prune_node \
                else sum_other_children

            if inner_prune_node == child_prune_node:
                c_leaves = self.subtree_leaves[inner_prune_node]
                c_length = self.subtree_length[inner_prune_node] - inner_prune_node.dist
                c_max = max(self.subtree_max_edge[child] for child in inner_prune_node.children)

                b_leaves = self.subtree_leaves[self.tree] - c_leaves
                b_length = self.subtree_length[self.tree] - c_length
                b_max = max(self._max_edge_excluding_subtrees(self.tree, [inner_prune_node]), inner_prune_node.dist)

                c1_leaves = self.subtree_leaves[outer_regraft_node]
                c1_length = self.subtree_length[outer_regraft_node]
                c1_max = self.subtree_max_edge[outer_regraft_node]

                c2_leaves = c_leaves - c1_leaves
                c2_length = c_length - c1_length
                c2_max = max(self._max_edge_excluding_subtrees(
                    child, [outer_regraft_node]) for child in inner_prune_node.children)
            else:
                b_leaves = self.subtree_leaves[outer_prune_node]
                b_length = self.subtree_length[outer_prune_node]
                b_max = self.subtree_max_edge[outer_prune_node]

                c_leaves = self.subtree_leaves[self.tree] - b_leaves
                c_length = self.subtree_length[self.tree] - b_length
                c_max = self._max_edge_excluding_subtrees(self.tree, [outer_prune_node])

                c1_leaves = self.subtree_leaves[outer_regraft_node]
                c1_length = self.subtree_length[outer_regraft_node]
                c1_max = self.subtree_max_edge[outer_regraft_node]

                c2_leaves = c_leaves - c1_leaves
                c2_length = c_length - c1_length
                c2_max = self._max_edge_excluding_subtrees(self.tree, [outer_prune_node, outer_regraft_node])

            f8, f9, f10, f11 = b_leaves, c_leaves, c1_leaves, c2_leaves
            f12, f13, f14, f15 = b_length, c_length, c1_length, c2_length
            f16, f17, f18, f19 = b_max, c_max, c1_max, c2_max

            prune_split: set = self.subtree_split[child_prune_node]
            min_prune_split = prune_split
            # prune_split_support_nj = split_support_nj[min_prune_split]
            # prune_split_support_upgma = split_support_upgma[min_prune_split]

            regraft_split: set = self.subtree_split[child_regraft_node]
            min_regraft_split = regraft_split
            # regraft_split_support_nj = split_support_nj[min_regraft_split]
            # regraft_split_support_upgma = split_support_upgma[min_regraft_split]

            if inner_prune_node == child_prune_node or inner_prune_node == self.tree:
                untouched_subtree = [
                    child for child in inner_prune_node.children if not self.is_descendant(outer_regraft_node, child)][0]
            elif self.is_descendant(outer_regraft_node, inner_prune_node):
                untouched_subtree = inner_prune_node
            else:
                untouched_subtree = [child for child in inner_prune_node.children if child != child_prune_node][0]
            new_prune_split = self.subtree_split[untouched_subtree]
            # new_branch_split_support_nj = split_support_nj[new_branch_split]
            # new_branch_split_support_upgma = split_support_upgma[new_branch_split]

            new_regraft_split = self._min_subtract_split(prune_split, regraft_split)

            return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
                    min_prune_split, min_regraft_split, new_prune_split, new_regraft_split]

        results = [compute_move_specific_features(move) for move in possible_moves]

        return results


def perform_spr_move(tree: Tree, move):
    prune_edge, regraft_edge, lca = move
    inner_prune_node, outer_prune_node, child_prune_node = prune_edge
    inner_regraft_node, outer_regraft_node, child_regraft_node = regraft_edge

    new_tree = tree.copy()

    new_child_regraft_node = (new_tree & child_regraft_node.name)
    new_tree.set_outgroup(new_child_regraft_node)
    new_tree.name = "NEWROOT"
    (new_tree & "").name = tree.name

    new_child_prune_node = (new_tree & outer_prune_node.name)
    new_parent_prune_node = new_child_prune_node.up

    new_child_prune_node.detach()
    new_tree.add_child(new_child_prune_node)

    remaining_child_node = new_parent_prune_node.children[0]
    remaining_child_node.detach()
    grandparent_prune_node = new_parent_prune_node.up
    new_parent_prune_node.detach()
    grandparent_prune_node.add_child(remaining_child_node)
    remaining_child_node.dist += new_parent_prune_node.dist
    new_tree.name = new_parent_prune_node.name

    return new_tree


def print_readable_features(f):
    print(f"Total length: {f[0]} \tLongest: {f[1]}\n"
          f"Prune length {f[2]} \tRegraft length {f[3]}\n"
          f"Topo distance {f[4]} \tLength distance {f[5]}\n"
          f"New branch length {f[6]}\n\n"
          f"Subtree b:\n Leafs: {f[7]}, Length: {f[11]}, Max: {f[15]}\n"
          f"Subtree c:\n Leafs: {f[8]}, Length: {f[12]}, Max: {f[16]}\n"
          f"Subtree c1:\n Leafs: {f[9]}, Length: {f[13]}, Max: {f[17]}\n"
          f"Subtree c2:\n Leafs: {f[10]}, Length {f[14]}, Max: {f[18]}\n"
          )


if __name__ == "__main__":
    # Build tree with branch lengths
    t = Tree("(B:2,C:4,((D:5,(E:6,F:7)Int2:8)Int3:9,(G:3,A:1)Int1:10)Int4:11)Int0;", format=1)
    print(t.get_ascii(attributes=["name", "dist"]))

    # Preprocess
    lca_tree = TreePreprocessor(t)

    # Print SPR moves
    moves = lca_tree.get_possible_spr_moves(radius=1)
    # moves = [
    #     # 1. Root in prune subtree
    #     # (inner_prune_node: child, inner_regraft_node: parent)
    #     ((t & "Int3", t & "Int4", t & "Int3"),
    #      (t & "Int2", t & "F", t & "F"), t & "Int3"),
    #     # 2. Regraft edge is descendant from prune parent
    #     # (inner_prune_node: parent, inner_regraft_node: parent)
    #     ((t & "Int3", t & "D", t & "D"),
    #      (t & "Int2", t & "F", t & "F"), t & "Int3"),
    #     # 3. Prune edge is descendant from regraft child
    #     # (inner_prune_node: parent, inner_regraft_node: child)
    #     ((t & "Int2", t & "F", t & "F"),
    #      (t & "Int3", t & "Int4", t & "Int3"), t & "Int3"),
    #     # 4. Prune edge is descendant from regraft parent
    #     # (inner_prune_node: parent, inner_regraft_node: parent)
    #     ((t & "Int2", t & "F", t & "F"),
    #      (t & "Int3", t & "D", t & "D"), t & "Int3"),
    #     # 5. Prune and regraft edges have lca node not in both edges
    #     # (inner_prune_node: parent, inner_regraft_node: parent)
    #     ((t & "Int2", t & "F", t & "F"),
    #      (t & "Int1", t & "G", t & "G"), t & "Int4"),
    #     # 6. Prune parent is root
    #     # (inner_prune_node: parent, inner_regraft_node: parent)
    #     ((t & "Int0", t & "C", t & "C"),
    #      (t & "Int4", t & "Int3", t & "Int3"), t & "Int0")
    # ]
    feats = lca_tree.extract_all_spr_features(moves)
    print(f"Found {len(moves)} SPR moves:")
    for i, move in enumerate(moves):
        prune_edge, regraft_edge, lca = move
        print(f"Prune {prune_edge[0].name}->{prune_edge[1].name}, "
              f"Regraft {regraft_edge[0].name}->{regraft_edge[1].name}, "
              f"LCA {lca.name}")
        print_readable_features(feats[i])
        # print(perform_spr_move(t, move).get_ascii(attributes=["name", "dist"]))
