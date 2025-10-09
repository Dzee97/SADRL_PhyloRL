import pickle
import random
import tempfile
import numpy as np
from pathlib import Path
from ete3 import Tree

from spr_feature_extractor import TreePreprocessor, perform_spr_move
from sample_datasets import run_cmd


class PhyloEnvNoEval:
    """
    Phylogenetic environment that doesn't evaluate likelihood immediately.
    Returns pending trees for batch evaluation.
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
            with open(sample["rand_trees"]) as f:
                sample["rand_trees_list"] = [line for line in f]
            # Extract normalization likelihood from log
            with open(sample["pars_log"]) as f:
                for line in f:
                    if line.startswith("Final LogLikelihood:"):
                        sample["norm_ll"] = float(line.strip().split()[-1])
                        break
            # Load the split support dicts
            with open(sample["split_support_upgma"], "rb") as f:
                sample["split_support_upgma_counter"] = pickle.load(f)
            with open(sample["split_support_nj"], "rb") as f:
                sample["split_support_nj_counter"] = pickle.load(f)

            self.samples.append(sample)

        self.current_sample = None
        self.current_sample_idx = None
        self.current_tree = None
        self.current_ll = None
        self.step_count = 0

    def reset(self):
        """Pick a random sample and random starting tree."""
        self.current_sample_idx = random.randint(0, len(self.samples) - 1)
        self.current_sample = self.samples[self.current_sample_idx]
        start_tree = random.choice(self.current_sample["rand_trees_list"])
        self.current_tree = Tree(start_tree, format=1)
        self.current_ll = None  # Will be set after batch evaluation
        self.step_count = 0
        return self.current_tree, self.current_sample_idx

    def step(self, annotated_tree, move):
        """
        Perform one SPR move but DON'T evaluate likelihood yet.
        Returns the new tree and whether episode is done.
        """
        new_tree = perform_spr_move(annotated_tree, move)
        self.current_tree = new_tree
        self.step_count += 1
        done = self.step_count >= self.horizon
        return new_tree, done

    def set_likelihood(self, ll):
        """Set the likelihood after batch evaluation."""
        prev_ll = self.current_ll
        self.current_ll = ll
        return prev_ll

    def extract_features(self):
        """Compute the feature vector for current tree."""
        preproc = TreePreprocessor(self.current_tree)
        annotated_tree, possible_moves = preproc.get_possible_spr_moves()
        feats = preproc.extract_all_spr_features(
            possible_moves,
            split_support_upgma=self.current_sample["split_support_upgma_counter"],
            split_support_nj=self.current_sample["split_support_nj_counter"])
        return annotated_tree, possible_moves, feats


class VectorizedPhyloEnv:
    """
    Manages multiple PhyloEnv instances and batches RAxML-NG evaluations.
    Similar to gym's VectorEnv but optimized for phylogenetic tree evaluation.
    """

    def __init__(self, samples_parent_dir: Path, raxmlng_path: Path,
                 horizon: int, num_envs: int = 4):
        self.samples_parent_dir = Path(samples_parent_dir)
        self.raxmlng_path = Path(raxmlng_path)
        self.horizon = horizon
        self.num_envs = num_envs

        # Create multiple environment instances
        self.envs = [
            PhyloEnvNoEval(samples_parent_dir, raxmlng_path, horizon)
            for _ in range(num_envs)
        ]

    def reset(self):
        """Reset all environments and return initial features after batch evaluation."""
        trees = []
        sample_indices = []

        for i, env in enumerate(self.envs):
            tree, sample_idx = env.reset()
            trees.append(tree)
            sample_indices.append(sample_idx)

        # Batch evaluate all initial trees
        likelihoods = self._batch_evaluate(trees, sample_indices)

        # Set likelihoods and extract features
        features_list = []
        for env, ll in zip(self.envs, likelihoods):
            env.set_likelihood(ll)
            features_list.append(env.extract_features())

        return features_list

    def step(self, actions):
        """
        Take a step in each environment with the given actions.

        Args:
            actions: List of (annotated_tree, move) tuples, one per environment

        Returns:
            features_list: List of (annotated_tree, moves, feats) for each env
            rewards: Array of rewards for each env
            dones: Array of done flags for each env
        """
        trees = []
        sample_indices = []
        prev_lls = []
        dones = []

        # All environments take a step
        for env, (annotated_tree, move) in zip(self.envs, actions):
            prev_lls.append(env.current_ll)
            new_tree, done = env.step(annotated_tree, move)
            trees.append(new_tree)
            dones.append(done)
            sample_indices.append(env.current_sample_idx)

        # Batch evaluate all new trees
        new_likelihoods = self._batch_evaluate(trees, sample_indices)

        # Compute rewards
        rewards = [new_ll - prev_ll for new_ll, prev_ll in zip(new_likelihoods, prev_lls)]

        # Set new likelihoods
        for env, new_ll in zip(self.envs, new_likelihoods):
            env.set_likelihood(new_ll)

        # Extract features for next step
        features_list = [env.extract_features() for env in self.envs]

        return features_list, rewards, dones

    def _batch_evaluate(self, trees, sample_indices):
        """
        Evaluate multiple trees in a single RAxML-NG call.

        Args:
            trees: List of Tree objects to evaluate
            sample_indices: List of sample indices (which dataset each tree belongs to)

        Returns:
            List of log-likelihoods
        """


if __name__ == "__main__":
    env = PhyloEnv(
        samples_parent_dir=Path("OUTTEST"),
        raxmlng_path=Path("raxmlng/raxml-ng"),
        horizon=20
    )
    tree, moves, feats = env.reset()
    done = False
    total_reward = 0

    while not done:
        move = random.choice(moves)
        (tree, moves, feats), reward, done = env.step(tree, move)
        total_reward += reward

    print(f"Total reward: {total_reward}")
    print(f"Real scale: {total_reward * abs(env.current_sample['norm_ll'])}")
