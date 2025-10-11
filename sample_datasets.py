import subprocess
import pickle
from pathlib import Path
from typing import List, Set, FrozenSet
from collections import Counter

import numpy as np
from Bio import AlignIO, Phylo, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from joblib import Parallel, delayed

from hyperparameters import (
    NUM_SAMPLES, SAMPLE_SIZE, NUM_PARS_TREES, NUM_RAND_TREES,
    NUM_BOOTSTRAP, EVO_MODEL, N_JOBS_BOOTSTRAP
)

from hyperparameters import (
    NUM_SAMPLES, SAMPLE_SIZE, NUM_PARS_TREES, NUM_RAND_TREES,
    NUM_BOOTSTRAP, EVO_MODEL, N_JOBS_BOOTSTRAP
)

# ---------- Helpers ----------


def run_cmd(cmd: List[str], check: bool = True, quiet: bool = False):
    """Run a shell command, optionally silencing output."""
    if quiet:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {proc.returncode}")
    return proc


def create_subsample_fasta(full_fasta: Path, out_fasta: Path, sample_names: List[str]):
    seqs = SeqIO.to_dict(SeqIO.parse(str(full_fasta), "fasta"))
    records = [seqs[n] for n in sample_names]
    SeqIO.write(records, str(out_fasta), "fasta")


def tree_splits(tree: Phylo.BaseTree.Tree) -> Set[FrozenSet[str]]:
    leaves = [leaf.name for leaf in tree.get_terminals()]
    all_set = set(leaves)
    splits = set(frozenset({leaf}) for leaf in leaves)
    for clade in tree.get_nonterminals():
        side = set(t.name for t in clade.get_terminals())
        if len(side) in (0, 1, len(all_set)):
            continue
        complement = all_set - side
        if len(side) < len(complement) or (len(side) == len(complement) and sorted(all_set)[0] in side):
            canonical = side
        else:
            canonical = complement
        splits.add(frozenset(canonical))
    return splits

# --- More Efficient Bootstrap ----


def aln_to_numpy(aln: MultipleSeqAlignment):
    # Convert alignment to a NumPy array of shape (n_sequences, n_columns)
    arr = np.array([list(str(rec.seq)) for rec in aln], dtype="U1")
    # Boolean mask for columns that are *not* all gaps
    non_gap_mask = ~(arr == '-').all(axis=0)
    # Apply mask to trim all-gap columns
    trimmed_arr = arr[:, non_gap_mask]
    return trimmed_arr


def precompute_pairwise_diffs(arr: np.ndarray):
    nseqs, ncols = arr.shape
    diffs = np.zeros((nseqs, nseqs, ncols), dtype=np.bool)
    for i in range(nseqs):
        diffs[i] = arr[i][None, :] != arr
    return diffs


def bootstrap_one(diffs: np.ndarray, ncols: int, ids: List[str]):
    indices = np.random.randint(0, ncols, size=ncols)
    boot_diffs = diffs[:, :, indices]
    distances = boot_diffs.sum(axis=2) / ncols
    nseqs = len(ids)
    tri_matrix = []
    for i in range(nseqs):
        row = [float(distances[i, j]) for j in range(i + 1)]
        tri_matrix.append(row)
    dm = DistanceMatrix(names=ids, matrix=tri_matrix)
    tree_upgma = DistanceTreeConstructor().upgma(dm)
    tree_nj = DistanceTreeConstructor().nj(dm)
    return tree_splits(tree_upgma), tree_splits(tree_nj)


def compute_bootstrap_support(aln: MultipleSeqAlignment, num_bootstrap: int, n_jobs: int = -1):
    arr = aln_to_numpy(aln)
    diffs = precompute_pairwise_diffs(arr)
    ids = [rec.id for rec in aln]
    ncols = arr.shape[1]

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(bootstrap_one)(diffs, ncols, ids) for _ in range(num_bootstrap)
    )

    split_counts_upgma, split_counts_nj = Counter(), Counter()
    for splits in results:
        split_counts_upgma.update(splits[0])
        split_counts_nj.update(splits[1])
    for split_counts in (split_counts_upgma, split_counts_nj):
        for s in split_counts:
            split_counts[s] /= num_bootstrap

    return split_counts_upgma, split_counts_nj


# ---------- Main function ----------


def sample_dataset(input_fasta: Path, outdir: Path, num_samples: int = NUM_SAMPLES,
                   sample_size: int = SAMPLE_SIZE, num_pars_trees: int = NUM_PARS_TREES,
                   num_rand_trees: int = NUM_RAND_TREES, num_bootstrap: int = NUM_BOOTSTRAP,
                   raxmlng_path: Path = Path("raxmlng/raxml-ng"), evo_model: str = EVO_MODEL):
    outdir.mkdir(parents=True, exist_ok=True)

    seq_records = list(SeqIO.parse(str(input_fasta), "fasta"))
    all_ids = [r.id for r in seq_records]
    num_seqs = len(all_ids)
    if sample_size > num_seqs:
        raise ValueError(f"Sample size {sample_size} > total sequences {num_seqs}")

    for s in range(1, num_samples+1):
        # Setup sample dir
        sample_dir = outdir / f"sample_{s:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_fasta = sample_dir / "sample.fasta"

        # Generate sample and write to fasta
        sample_ids = np.random.choice(all_ids, size=sample_size, replace=False)
        create_subsample_fasta(input_fasta, sample_fasta, sample_ids)

        # Create starting parsiminy trees via RAxML-NG
        prefix_pars = "raxml_pars"
        cmd_pars = [
            raxmlng_path, "--start", "--msa", str(sample_fasta), "--model", evo_model,
            "--prefix", str(sample_dir / prefix_pars), "--tree", f"pars{{{num_pars_trees}}}"
        ]
        run_cmd(cmd_pars, quiet=True)
        # Save starting trees file, remove the rest
        for out_file in sample_dir.glob(f"{prefix_pars}*"):
            if out_file.suffix == ".startTree":
                par_file = out_file
            else:
                out_file.unlink()

        # Evaluate parsimony trees
        prefix_eval_pars = "raxml_eval_pars"
        cmd_eval_pars = [
            raxmlng_path, "--evaluate", "--msa", str(sample_fasta), "--model", evo_model,
            "--prefix", str(sample_dir / prefix_eval_pars), "--tree", str(par_file)
        ]
        run_cmd(cmd_eval_pars, quiet=True)
        # Remove starting trees after evaluation
        par_file.unlink()
        # Remove all files except model params and best likelihood (in the log)
        for out_file in sample_dir.glob(f"{prefix_eval_pars}*"):
            if out_file.suffix not in [".bestModel", ".log"]:
                out_file.unlink()

        # Create starting random trees via RAxML-NG
        prefix_rand = "raxml_rand"
        cmd_rand = [
            raxmlng_path, "--start", "--msa", str(sample_fasta), "--model", evo_model,
            "--prefix", str(sample_dir / prefix_rand), "--tree", f"rand{{{num_rand_trees}}}"
        ]
        run_cmd(cmd_rand, quiet=True)
        # Remove all files except starting trees
        for out_file in sample_dir.glob(f"{prefix_rand}*"):
            if out_file.suffix != ".startTree":
                out_file.unlink()

        # Create bootstrap trees and count splits
        aln = AlignIO.read(str(sample_fasta), "fasta")
        split_support_upgma, split_support_nj = compute_bootstrap_support(aln, num_bootstrap, n_jobs=N_JOBS_BOOTSTRAP)

        split_support_upgma_pkl = sample_dir / "split_support_upgma.pkl"
        with open(split_support_upgma_pkl, "wb") as f:
            pickle.dump(split_support_upgma, f)

        split_support_nj_pkl = sample_dir / "split_support_nj.pkl"
        with open(split_support_nj_pkl, "wb") as f:
            pickle.dump(split_support_nj, f)


if __name__ == "__main__":
    sample_dataset(input_fasta=Path("datasets/051_856_p__Basidiomycota_c__Agaricomycetes_o__Russulales.fasta"),
                   outdir=Path("OUTTEST10"))
