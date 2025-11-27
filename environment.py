import pickle
import random
import tempfile
from pathlib import Path
from ete3 import Tree

from spr_feature_extractor import TreePreprocessor, perform_spr_move
from sample_datasets import run_cmd
from tree_hash import unrooted_tree_hash

# Import GNN data structures (optional - only if using GNN agent)
try:
    from agents import GraphData, ActionEmbedding, make_edges_bidirectional
    import torch
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


class PhyloEnv:
    """
    Gym-like environment for reinforcement learning in phylogenetic tree search.
    """

    def __init__(self, samples_parent_dir: Path, raxmlng_path: Path, horizon: int, use_gnn: bool = False):
        self.samples_parent_dir = Path(samples_parent_dir)
        self.raxmlng_path = Path(raxmlng_path)
        self.horizon = horizon
        self.use_gnn = use_gnn
        
        if use_gnn and not GNN_AVAILABLE:
            raise RuntimeError("use_gnn=True but GNN components not available. Install torch-geometric and ensure agents.py is present.")

        self.samples = []
        self.num_train_start_trees = []
        self.num_test_start_trees = []
        for sample_dir in self.samples_parent_dir.glob("sample_*"):
            sample = {
                "dir": sample_dir,
                "msa": sample_dir / "sample_aln.fasta",
                "rand_train_trees": sample_dir / "raxml_rand_train.raxml.startTree",
                "rand_test_trees": sample_dir / "raxml_rand_test.raxml.startTree",
                "rand_test_trees_ml": sample_dir / "raxml_rand_test_ml.raxml.log",
                "pars_model": sample_dir / "raxml_eval_pars.raxml.bestModel",
                "pars_log": sample_dir / "raxml_eval_pars.raxml.log",
                "split_support_upgma": sample_dir / "split_support_upgma.pkl",
                "split_support_nj": sample_dir / "split_support_nj.pkl"
            }
            # Extract individual newick trees from random trees files
            with open(sample["rand_train_trees"]) as f:
                sample["rand_train_trees_list"] = [line for line in f]
                self.num_train_start_trees.append(len(sample["rand_train_trees_list"]))
            with open(sample["rand_test_trees"]) as f:
                sample["rand_test_trees_list"] = [line for line in f]
                self.num_test_start_trees.append(len(sample["rand_test_trees_list"]))
            # Extract RAxML-NG search results for random test trees
            with open(sample["rand_test_trees_ml"]) as f:
                rand_test_trees_ml_list = []
                for line in f:
                    if "ML tree search #" in line:
                        rand_test_trees_ml_list.append(float(line.strip().split()[-1]))
                    if line.startswith("Final LogLikelihood:"):
                        sample["rand_test_trees_ml_best"] = float(line.strip().split()[-1])
                sample["rand_test_trees_ml_list"] = rand_test_trees_ml_list
            # Extract parimony normalization likelihood from log
            with open(sample["pars_log"]) as f:
                for line in f:
                    if line.startswith("Final LogLikelihood:"):
                        sample["pars_ll"] = float(line.strip().split()[-1])
                        break
            # Load the split support dicts
            with open(sample["split_support_upgma"], "rb") as f:
                sample["split_support_upgma_counter"] = pickle.load(f)
            with open(sample["split_support_nj"], "rb") as f:
                sample["split_support_nj_counter"] = pickle.load(f)

            self.samples.append(sample)

        self.current_sample = None
        self.current_tree = None
        self.current_ll = None
        self.current_moves = None
        self.current_feats = None  # Hand-crafted features
        self.current_graph = None  # GNN: GraphData
        self.current_actions = None  # GNN: List[ActionEmbedding]

        self.tree_cache = {}
        self.cache_hits = 0

    def reset(self, sample_num: int = None, start_tree_set: str = "train", start_tree_num: int = None):
        """Pick a random sample and random starting tree, prepare RAxML working dir in RAM."""
        self.step_count = 0
        self.cache_hits = 0

        sample_num = random.randrange(0, len(self.samples)) if sample_num is None else sample_num
        self.current_sample = self.samples[sample_num]

        if start_tree_set == "train":
            rand_tree_list = self.current_sample["rand_train_trees_list"]
        elif start_tree_set == "test":
            rand_tree_list = self.current_sample["rand_test_trees_list"]
        else:
            raise ValueError('start_tree_set must either be "train" or "test"')
        start_tree_num = random.randrange(
            0, len(rand_tree_list)) if start_tree_num is None else start_tree_num
        start_tree_nwk = rand_tree_list[start_tree_num]
        start_tree = Tree(start_tree_nwk, format=1)

        tree_hash = unrooted_tree_hash(start_tree)
        if tree_hash in self.tree_cache:
            start_tree_optim, self.current_ll = self.tree_cache[tree_hash]
            self.cache_hits += 1
        else:
            start_tree_optim, self.current_ll = self._evaluate_likelihood(start_tree)
            self.tree_cache[tree_hash] = (start_tree_optim, self.current_ll)

        if self.use_gnn:
            self.current_tree, self.current_moves, self.current_graph, self.current_actions = self._extract_features_gnn(start_tree_optim)
            return tree_hash, (self.current_graph, self.current_actions)
        else:
            self.current_tree, self.current_moves, self.current_feats = self._extract_features(start_tree_optim)
            return tree_hash, self.current_feats

    def step(self, move_idx):
        """
        Perform one SPR move and evaluate reward.
        `move` is a tuple of (prune_edge, regraft_edge) produced by TreePreprocessor.
        """
        move = self.current_moves[move_idx]
        neighbor_tree = perform_spr_move(self.current_tree, move)
        tree_hash = unrooted_tree_hash(neighbor_tree)
        if tree_hash in self.tree_cache:
            neighbor_tree_optim, neighbor_ll = self.tree_cache[tree_hash]
            self.cache_hits += 1
        else:
            neighbor_tree_optim, neighbor_ll = self._evaluate_likelihood(neighbor_tree)
            self.tree_cache[tree_hash] = (neighbor_tree_optim, neighbor_ll)

        reward = (neighbor_ll - self.current_ll)  # / abs(self.current_sample["norm_ll"])

        if self.use_gnn:
            self.current_tree, self.current_moves, self.current_graph, self.current_actions = self._extract_features_gnn(neighbor_tree_optim)
            feats_return = (self.current_graph, self.current_actions)
        else:
            self.current_tree, self.current_moves, self.current_feats = self._extract_features(neighbor_tree_optim)
            feats_return = self.current_feats
            
        self.current_ll = neighbor_ll
        self.step_count += 1
        done = self.step_count >= self.horizon

        return tree_hash, feats_return, reward, done

    def preview_step(self, move_idx, calc_reward=False):
        """
        Preview the resulting tree hash from performing one SPR move, but keep the current tree state
        """
        move = self.current_moves[move_idx]
        neighbor_tree = perform_spr_move(self.current_tree, move)
        tree_hash = unrooted_tree_hash(neighbor_tree)

        if not calc_reward:
            return tree_hash

        if tree_hash in self.tree_cache:
            neighbor_tree_optim, neighbor_ll = self.tree_cache[tree_hash]
        else:
            neighbor_tree_optim, neighbor_ll = self._evaluate_likelihood(neighbor_tree)
            self.tree_cache[tree_hash] = (neighbor_tree_optim, neighbor_ll)

        reward = (neighbor_ll - self.current_ll)
        return tree_hash, reward

    def _extract_features(self, tree: Tree):
        """Compute the hand-crafted feature vectors for current tree."""
        preproc = TreePreprocessor(tree)
        annotated_tree, possible_moves = preproc.get_possible_spr_moves()
        feats = preproc.extract_all_spr_features(
            possible_moves,
            split_support_upgma=self.current_sample["split_support_upgma_counter"],
            split_support_nj=self.current_sample["split_support_nj_counter"])
        return annotated_tree, possible_moves, feats
    
    def _extract_features_gnn(self, tree: Tree):
        """Extract GNN-compatible graph data and action embeddings."""
        preproc = TreePreprocessor(tree)
        annotated_tree, possible_moves = preproc.get_possible_spr_moves()
        
        # Convert TreePreprocessor to GraphData
        graph_data = self._preprocessor_to_graph_data(preproc)
        
        # Convert SPRMoves to ActionEmbeddings (Tensor)
        action_embeddings = self._moves_to_action_embeddings(possible_moves, preproc, graph_data.node_name_to_idx)
        
        return annotated_tree, possible_moves, graph_data, action_embeddings
    
    def _preprocessor_to_graph_data(self, preproc: TreePreprocessor) -> GraphData:
        """Convert TreePreprocessor to GraphData with bidirectional edges.
        
        Node features: [is_leaf] (1-dim binary)
        Edge features: [branch_length] (1-dim)
        """
        # Build node features and name mapping
        node_names = []
        node_features_list = []
        
        # Use DFS order to assign consistent indices
        def collect_nodes(node):
            node_names.append(node.name)
            # ATOMIC NODE FEATURE: Only "is_leaf" (1.0 if leaf, 0.0 if internal)
            node_features_list.append([1.0 if node.is_leaf() else 0.0])
            for child in node.children:
                collect_nodes(child)
        
        collect_nodes(preproc.tree)
        
        # Create name to index mapping
        node_name_to_idx = {name: idx for idx, name in enumerate(node_names)}
        
        # Create node features tensor [num_nodes, 1]
        node_features = torch.tensor(node_features_list, dtype=torch.float32)
        
        # Build DIRECTED edges from preprocessor (child→parent)
        edge_list_src = []
        edge_list_dst = []
        edge_features_list = []
        
        for child_node, parent_node in preproc.edges:
            child_idx = node_name_to_idx[child_node.name]
            parent_idx = node_name_to_idx[parent_node.name]
            
            edge_list_src.append(child_idx)
            edge_list_dst.append(parent_idx)
            
            # ATOMIC EDGE FEATURE: Only "branch_length"
            edge_features_list.append([child_node.dist])
        
        # Convert to tensors
        edge_index_directed = torch.tensor([edge_list_src, edge_list_dst], dtype=torch.long)
        edge_features_directed = torch.tensor(edge_features_list, dtype=torch.float32)
        
        # CRITICAL: Make edges BIDIRECTIONAL for unrooted tree representation!
        edge_index, edge_features = make_edges_bidirectional(edge_index_directed, edge_features_directed)
        
        return GraphData(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_name_to_idx=node_name_to_idx
        )
    
    def _moves_to_action_embeddings(self, possible_moves, preproc: TreePreprocessor, node_name_to_idx: dict):
        """Convert SPRMoves to tensor using stable node names."""
        action_data = []
        
        for move in possible_moves:
            prune_edge, regraft_edge, lca = move
            
            # Extract node names from edges
            prune_inner, prune_outer, _ = prune_edge
            regraft_inner, regraft_outer, _ = regraft_edge
            
            # Compute topological distance
            distance = preproc.depth_count[prune_inner] + preproc.depth_count[regraft_inner] - 2 * preproc.depth_count[lca]
            
            # Branch lengths
            prune_dist = prune_edge.child_node.dist
            regraft_dist = regraft_edge.child_node.dist
            
            action_data.append([
                node_name_to_idx[prune_inner.name],
                node_name_to_idx[prune_outer.name],
                node_name_to_idx[regraft_inner.name],
                node_name_to_idx[regraft_outer.name],
                float(distance),
                float(prune_dist),
                float(regraft_dist)
            ])
        
        return torch.tensor(action_data, dtype=torch.float32)

    def _evaluate_likelihood(self, tree: Tree):
        """
        Run RAxML-NG to evaluate the likelihood of the given tree,
        optimizing branch lengths. Returns (optimized_tree, log_likelihood).
        """
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            tmp_path = Path(tmp)
            treefile = tmp_path / "tree.nwk"
            tree.write(outfile=str(treefile), format=1)

            prefix = tmp_path / "eval"
            cmd = [
                str(self.raxmlng_path),
                "--evaluate",
                "--msa", str(self.current_sample["msa"]),
                "--model", str(self.current_sample["pars_model"]),
                "--tree", str(treefile),
                "--prefix", str(prefix),
                "--opt-model", "off",
                "--opt-branches", "on"
            ]
            run_cmd(cmd, quiet=True)

            # --- parse likelihood from log file ---
            log_file = prefix.with_suffix(".raxml.log")
            ll = None
            with open(log_file) as f:
                for line in f:
                    if line.startswith("Final LogLikelihood:"):
                        ll = float(line.strip().split()[-1])
                        break
            if ll is None:
                raise RuntimeError("Could not parse likelihood from RAxML-NG log.")

            # --- load optimized tree with branch lengths ---
            best_tree_file = prefix.with_suffix(".raxml.bestTree")
            if not best_tree_file.exists():
                raise FileNotFoundError("RAxML-NG did not produce a .bestTree file.")
            optimized_tree = Tree(open(best_tree_file).read(), format=1)
            # fix removal of root node name
            optimized_tree.name = tree.name

            return optimized_tree, ll


if __name__ == "__main__":
    env = PhyloEnv(
        samples_parent_dir=Path("OUTTEST"),
        raxmlng_path=Path("raxmlng/raxml-ng"),
        horizon=20
    )
    feats = env.reset()
    done = False
    total_reward = 0

    while not done:
        move_idx = random.randrange(0, len(feats))
        feats, reward, done = env.step(move_idx)
        total_reward += reward

    print(f"Total reward: {total_reward}")
