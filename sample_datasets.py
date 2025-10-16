import subprocess
import pickle
import shutil
from pathlib import Path
from typing import List, Set, FrozenSet
from collections import Counter

import numpy as np
from Bio import AlignIO, Phylo, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from joblib import Parallel, delayed


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
    """Write a subset of sequences by ID to a new FASTA."""
    seqs = SeqIO.to_dict(SeqIO.parse(str(full_fasta), "fasta"))
    records = [seqs[n] for n in sample_names]
    SeqIO.write(records, str(out_fasta), "fasta")


def tree_splits(tree: Phylo.BaseTree.Tree) -> Set[FrozenSet[str]]:
    """Return canonical set of bipartition splits from a tree."""
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


# ---------- Bootstrap support ----------

def aln_to_numpy(aln: MultipleSeqAlignment):
    """Convert alignment to a numpy array and remove all-gap columns."""
    arr = np.array([list(str(rec.seq)) for rec in aln], dtype="U1")
    non_gap_mask = ~(arr == '-').all(axis=0)
    trimmed_arr = arr[:, non_gap_mask]
    return trimmed_arr


def precompute_pairwise_diffs(arr: np.ndarray):
    """Precompute all pairwise differences between sequences."""
    nseqs, ncols = arr.shape
    diffs = np.zeros((nseqs, nseqs, ncols), dtype=bool)
    for i in range(nseqs):
        diffs[i] = arr[i][None, :] != arr
    return diffs


def bootstrap_one(diffs: np.ndarray, ncols: int, ids: List[str]):
    """Perform one bootstrap replicate, returning UPGMA and NJ splits."""
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
    """Compute bootstrap support for all splits."""
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


# ---------- Main sampling and analysis ----------

def sample_dataset(input_fasta: Path, outdir: Path, num_samples: int, sample_size: int,
                   num_pars_trees: int, num_rand_train_trees: int, num_rand_test_trees: int, num_bootstrap: int,
                   raxmlng_path: Path, evo_model: str, mafft_path: Path):
    """
    Sample smaller FASTAs from an unaligned full FASTA, align each using MAFFT,
    then perform RAxML-NG and bootstrap analyses.
    """
    # ---- Check for existing output directory ----
    if outdir.exists():
        answer = input(f"Output directory '{outdir}' already exists. Overwrite? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborting — existing output directory preserved.")
            return
        print(f"Removing existing directory: {outdir}")
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load sequences ----
    seq_records = list(SeqIO.parse(str(input_fasta), "fasta"))
    all_ids = [r.id for r in seq_records]
    num_seqs = len(all_ids)
    if sample_size > num_seqs:
        raise ValueError(f"Sample size {sample_size} > total sequences {num_seqs}")

    for s in range(1, num_samples + 1):
        print(f"=== Processing sample {s}/{num_samples} ===")
        sample_dir = outdir / f"sample_{s:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Sampling
        sample_ids = np.random.choice(all_ids, size=sample_size, replace=False)
        sample_fasta = sample_dir / "sample_unaligned.fasta"
        create_subsample_fasta(input_fasta, sample_fasta, sample_ids)

        # Step 2: Align with MAFFT
        sample_aln = sample_dir / "sample_aln.fasta"
        cmd_mafft = [str(mafft_path), "--auto", str(sample_fasta)]
        print("Running MAFFT...")
        with open(sample_aln, "w") as f_out:
            subprocess.run(cmd_mafft, stdout=f_out, stderr=subprocess.DEVNULL, check=True)

        # Step 3: RAxML-NG parsimony starting trees
        prefix_pars = "raxml_pars"
        cmd_pars = [
            str(raxmlng_path), "--start", "--msa", str(sample_aln), "--model", evo_model,
            "--prefix", str(sample_dir / prefix_pars), "--tree", f"pars{{{num_pars_trees}}}"
        ]
        run_cmd(cmd_pars, quiet=True)

        par_file = None
        for out_file in sample_dir.glob(f"{prefix_pars}*"):
            if out_file.suffix == ".startTree":
                par_file = out_file
            else:
                out_file.unlink()

        # Step 4: Evaluate parsimony trees
        prefix_eval_pars = "raxml_eval_pars"
        cmd_eval_pars = [
            str(raxmlng_path), "--evaluate", "--msa", str(sample_aln), "--model", evo_model,
            "--prefix", str(sample_dir / prefix_eval_pars), "--tree", str(par_file)
        ]
        run_cmd(cmd_eval_pars, quiet=True)
        par_file.unlink()
        for out_file in sample_dir.glob(f"{prefix_eval_pars}*"):
            if out_file.suffix not in [".bestModel", ".log"]:
                out_file.unlink()

        # Step 5: RAxML-NG random starting trees for training and testing
        for prefix_rand, num_rand_trees in [
                ("raxml_rand_train", num_rand_train_trees), ("raxml_rand_test", num_rand_test_trees)]:
            cmd_rand = [
                str(raxmlng_path), "--start", "--msa", str(sample_aln), "--model", evo_model,
                "--prefix", str(sample_dir / prefix_rand), "--tree", f"rand{{{num_rand_trees}}}"
            ]
            run_cmd(cmd_rand, quiet=True)
            for out_file in sample_dir.glob(f"{prefix_rand}*"):
                if out_file.suffix != ".startTree":
                    out_file.unlink()

        # Step 6: Bootstrap split support
        print("Computing bootstrap supports...")
        aln = AlignIO.read(str(sample_aln), "fasta")
        split_support_upgma, split_support_nj = compute_bootstrap_support(aln, num_bootstrap, n_jobs=-1)

        with open(sample_dir / "split_support_upgma.pkl", "wb") as f:
            pickle.dump(split_support_upgma, f)
        with open(sample_dir / "split_support_nj.pkl", "wb") as f:
            pickle.dump(split_support_nj, f)

        print(f"Sample {s} completed.\n")


# ---------- Command-line entry point ----------

if __name__ == "__main__":
    sample_dataset(
        input_fasta=Path("datasets/051_856_p__Basidiomycota_c__Agaricomycetes_o__Russulales.fasta"),
        outdir=Path("output/OUT_MAFFT_SAMPLES"),
        num_samples=10,
        sample_size=7,
        num_pars_trees=10,
        num_rand_train_trees=10,
        num_rand_test_trees=10,
        num_bootstrap=1000,
        raxmlng_path=Path("dependencies/raxmlng/raxml-ng"),
        evo_model="GTR+I+G",
        mafft_path=Path("dependencies/mafft-linux64/mafft.bat")
    )
