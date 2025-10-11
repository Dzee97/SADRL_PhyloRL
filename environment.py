import pickle
import random
import tempfile
from pathlib import Path
from ete3 import Tree

from spr_feature_extractor import TreePreprocessor, perform_spr_move
from sample_datasets import run_cmd
from hyperparameters import HORIZON


class PhyloEnv:
    """
    Gym-like environment for reinforcement learning in phylogenetic tree search.
    """

    def __init__(self, samples_parent_dir: Path, raxmlng_path: Path, horizon: int):
        self.samples_parent_dir = Path(samples_parent_dir)
        self.raxmlng_path = Path(raxmlng_path)
        self.horizon = horizon

        self.samples = []
        for sample_dir in self.samples_parent_dir.glob("sample_*"):
            sample = {
                "dir": sample_dir,
                "msa": sample_dir / "sample.fasta",
                "rand_trees": sample_dir / "raxml_rand.raxml.startTree",
                "pars_model": sample_dir / "raxml_eval_pars.raxml.bestModel",
                "pars_log": sample_dir / "raxml_eval_pars.raxml.log",
                "split_support_upgma": sample_dir / "split_support_upgma.pkl",
                "split_support_nj": sample_dir / "split_support_nj.pkl"
            }
            # Extract individual newick trees from ranfom trees file
            try:
                with open(sample["rand_trees"]) as f:
                    sample["rand_trees_list"] = []
                    for line in f:
                        line = line.strip()
                        if line:  # Only add non-empty lines
                            sample["rand_trees_list"].append(line)
            except FileNotFoundError:
                sample["rand_trees_list"] = []
            
            # Extract normalization likelihood from log
            try:
                with open(sample["pars_log"]) as f:
                    for line in f:
                        if line.startswith("Final LogLikelihood:"):
                            sample["norm_ll"] = float(line.strip().split()[-1])
                            break
            except FileNotFoundError:
                continue
            
            # Load the split support dicts
            try:
                with open(sample["split_support_upgma"], "rb") as f:
                    sample["split_support_upgma_counter"] = pickle.load(f)
                with open(sample["split_support_nj"], "rb") as f:
                    sample["split_support_nj_counter"] = pickle.load(f)
            except FileNotFoundError:
                continue

            # Only add sample if it has starting trees
            if sample["rand_trees_list"]:
                self.samples.append(sample)

        self.current_sample = None
        self.current_tree = None
        self.current_ll = None
        self.current_moves = None
        self.current_feats = None

        self.neighbor_cache = {}
        self.cache_hits = 0

    def reset(self):
        """Pick a random sample and random starting tree, prepare RAxML working dir in RAM."""
        self.current_sample = random.choice(self.samples)
        start_tree_nwk = random.choice(self.current_sample["rand_trees_list"])
        start_tree = Tree(start_tree_nwk, format=1)
        start_tree_optim, self.current_ll = self._evaluate_likelihood(start_tree)
        self.current_tree, self.current_moves, self.current_feats = self._extract_features(start_tree_optim)
        self.step_count = 0
        self.cache_hits = 0
        return self.current_feats

    def reset_to_specific_tree(self, sample, tree_nwk):
        """Reset to a specific sample and starting tree."""
        self.current_sample = sample
        start_tree = Tree(tree_nwk, format=1)
        start_tree_optim, self.current_ll = self._evaluate_likelihood(start_tree)
        self.current_tree, self.current_moves, self.current_feats = self._extract_features(start_tree_optim)
        self.step_count = 0
        self.cache_hits = 0
        return self.current_tree, self.current_moves, self.current_feats

    def step(self, move_idx):
        """
        Perform one SPR move and evaluate reward.
        `move` is a tuple of (prune_edge, regraft_edge) produced by TreePreprocessor.
        """
        feats_hash = str(self.current_feats[move_idx])
        if feats_hash in self.neighbor_cache:
            neighbor_tree_optim, neighbor_ll = self.neighbor_cache[feats_hash]
            self.cache_hits += 1
        else:
            move = self.current_moves[move_idx]
            neighbor_tree = perform_spr_move(self.current_tree, move)
            neighbor_tree_optim, neighbor_ll = self._evaluate_likelihood(neighbor_tree)
            self.neighbor_cache[feats_hash] = (neighbor_tree_optim, neighbor_ll)

        reward = (neighbor_ll - self.current_ll)  # / abs(self.current_sample["norm_ll"])

        self.current_tree, self.current_moves, self.current_feats = self._extract_features(neighbor_tree_optim)
        self.current_ll = neighbor_ll
        self.step_count += 1
        done = self.step_count >= self.horizon

        return self.current_feats, reward, done

    def _extract_features(self, tree: Tree):
        """Compute the feature vector for current tree."""
        preproc = TreePreprocessor(tree)
        annotated_tree, possible_moves = preproc.get_possible_spr_moves()
        feats = preproc.extract_all_spr_features(
            possible_moves,
            split_support_upgma=self.current_sample["split_support_upgma_counter"],
            split_support_nj=self.current_sample["split_support_nj_counter"])
        return annotated_tree, possible_moves, feats

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
        horizon=HORIZON
    )
    feats = env.reset()
    done = False
    total_reward = 0

    while not done:
        move_idx = random.randrange(0, len(feats))
        feats, reward, done = env.step(move_idx)
        total_reward += reward

    print(f"Total reward: {total_reward}")
