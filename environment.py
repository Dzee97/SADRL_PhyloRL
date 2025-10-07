import os
import pickle
import shutil
import random
import tempfile
import subprocess
from pathlib import Path
from ete3 import Tree

import numpy as np
from spr_feature_extractor import TreePreprocessor, perform_spr_move


class PhyloEnv:
    """
    Gym-like environment for reinforcement learning in phylogenetic tree search.
    """

    def __init__(self, samples_parent_dir: Path, raxmlng_path: Path):
        self.samples_parent_dir = Path(samples_parent_dir)
        self.raxmlng_path = Path(raxmlng_path)

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
                sample["rand_trees_list"] = []
                for line in f:
                    sample["rand_trees_list"].append(line)
            # Extract normalization likelihood from log
            with open(sample["pars_log"]) as f:
                for line in f:
                    if line.startswith("Final LogLikelihood"):
                        sample["norm_ll"] = float(line.strip().split()[-1])
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
        self.tmp_dir = None

    def reset(self):
        """Pick a random sample and random starting tree, prepare RAxML working dir in RAM."""
        self.current_sample = random.choice(self.samples)
        start_tree = random.choice(self.current_sample["rand_trees_list"])
        self.current_tree = Tree(start_tree, format=1)
        self.current_tree, self.current_ll = self._evaluate_likelihood(self.current_tree)
        self.tmp_dir = Path(tempfile.mkdtemp(dir="/dev/shm"))
        return self._extract_features()

    def step(self, move):
        """
        Perform one SPR move and evaluate reward.
        `move` is a tuple of (prune_edge, regraft_edge) produced by TreePreprocessor.
        """
        new_tree = perform_spr_move(self.current_tree, move)
        new_tree, new_ll = self._evaluate_likelihood(new_tree)
        reward = (new_ll - self.current_ll) / abs(self.current_sample["norm_ll"])

        self.current_tree = new_tree
        self.current_ll = new_ll
        return self._extract_features(), reward

    def _extract_features(self):
        preproc = TreePreprocessor(self.current_tree)
        possible_moves = preproc.get_possible_spr_moves()
        feats = preproc.extract_all_spr_features(
            possible_moves,
            split_support_upgma=self.current_sample["split_support_upgma_counter"],
            split_support_nj=self.current_sample["split_support_nj_counter"])


if __name__ == "__main__":
    env = PhyloEnv(
        samples_parent_dir=Path("OUTTEST"),
        raxmlng_path=Path("raxmlng/raxml-ng")
    )
    breakpoint()
