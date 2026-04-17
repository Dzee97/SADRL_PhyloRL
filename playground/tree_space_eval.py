import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm


# -----------------------------
# 1. Alignment parsing / writing
# -----------------------------

def read_fasta(path: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    seq_chunks: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                if header is None:
                    raise ValueError(f"Invalid FASTA: sequence before header in {path}")
                seq_chunks.append(line)

    if header is not None:
        records.append((header, "".join(seq_chunks)))

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    lengths = {len(seq) for _, seq in records}
    if len(lengths) != 1:
        raise ValueError("Alignment sequences are not all the same length.")

    return records


def sanitize_taxon_name(name: str) -> str:
    token = name.strip().split()[0]
    token = re.sub(r"[^A-Za-z0-9_.|:+\-]", "_", token)
    if not token:
        raise ValueError(f"Could not derive a valid taxon name from header: {name!r}")
    return token


def prepare_alignment(input_fasta: str) -> tuple[list[str], list[tuple[str, str]]]:
    records = read_fasta(input_fasta)
    taxa_names = [sanitize_taxon_name(header) for header, _ in records]

    if len(set(taxa_names)) != len(taxa_names):
        raise ValueError(
            "Duplicate taxon names after sanitization. "
            "Please make headers unique before running this script."
        )

    sanitized_records = [(taxon, seq) for taxon, (_, seq) in zip(taxa_names, records)]

    return taxa_names, sanitized_records


# -----------------------------
# 2. Helpers
# -----------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_newick_file(newicks: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for nwk in newicks:
            handle.write(nwk + "\n")


def write_fasta_file(records: list[tuple[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            handle.write(seq + "\n")


# -----------------------------
# 6. RAxML-NG evaluation cache
# -----------------------------

LL_LINE_PATTERN = re.compile(
    r"Tree\s*#\s*(\d+)\s*,\s*final\s+log[Ll]ikelihood\s*:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def evaluate_or_reuse_likelihoods(eval_dir: str, raxml_ng_path: str, records: list[tuple[str, str]], model_path: str,
                                  newicks: list[str], threads: str, force: bool = False) -> tuple[np.ndarray, bool]:
    eval_dir_path = Path(eval_dir)
    eval_dir_path.mkdir(parents=True, exist_ok=True)

    cleaned_alignment = eval_dir_path / "alignment.fasta"
    eval_newick_path = eval_dir_path / "all_topologies.nwk"
    likelihood_path = eval_dir_path / "likelihoods.npy"
    meta_path = eval_dir_path / "eval_meta.json"

    write_fasta_file(records, str(cleaned_alignment))
    write_newick_file(newicks, str(eval_newick_path))

    alignment_hash = sha256_file(str(cleaned_alignment))
    model_hash = sha256_file(model_path)
    newick_hash = sha256_file(str(eval_newick_path))

    expected_meta = {
        "alignment_sha256": alignment_hash,
        "model_sha256": model_hash,
        "newick_sha256": newick_hash,
    }

    reusable = (
        not force
        and likelihood_path.exists()
        and meta_path.exists()
        and load_json(str(meta_path)) == expected_meta
    )

    if reusable:
        return np.load(likelihood_path), reusable

    prefix = str(eval_dir_path / "raxml_eval")

    cmd = [
        raxml_ng_path,
        "--evaluate",
        "--msa", str(cleaned_alignment),
        "--model", model_path,
        "--tree", str(eval_newick_path),
        "--prefix", prefix,
        "--threads", threads,
        "--opt-model", "off",
        "--opt-branches", "on",
        "--force",
        "--nofiles",
    ]

    num_trees = len(newicks)
    seen = 0
    values: dict[int, float] = {}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          universal_newlines=True) as proc, tqdm(total=num_trees, desc="RAxML-NG") as pbar:
        for line in proc.stdout:
            m = LL_LINE_PATTERN.search(line)
            if m:
                tree_idx = int(m.group(1))
                values[tree_idx] = float(m.group(2))
                if tree_idx > seen:
                    pbar.update(tree_idx - seen)
                    seen = tree_idx
        returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError("RAxML-NG failed")

    likelihoods = np.array([values[i] for i in range(1, num_trees + 1)], dtype=float)
    np.save(likelihood_path, likelihoods)
    save_json(str(meta_path), expected_meta)

    return likelihoods, reusable
