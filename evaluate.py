import csv
import json
import os
from pathlib import Path
from typing import List
import torch
from environment import PhyloEnv
from dqn_agent import DQNAgent
from sample_datasets import run_cmd
from hyperparameters import (
    DEFAULT_EVALUATION_CHECKPOINTS_DIR,
    DEFAULT_EVALUATION_SAMPLES_DIR,
    DEFAULT_EVALUATION_OUTPUT_DIR,
    DEFAULT_RAXML_PATH,
    HORIZON
)


def main():
    checkpoints_dir = DEFAULT_EVALUATION_CHECKPOINTS_DIR
    samples_dir = DEFAULT_EVALUATION_SAMPLES_DIR
    output_dir = DEFAULT_EVALUATION_OUTPUT_DIR
    raxml_path = DEFAULT_RAXML_PATH
    
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get checkpoints and samples
    checkpoints = get_checkpoints(checkpoints_dir)
    samples = get_samples(samples_dir)
    
    print(f"Found {len(checkpoints)} checkpoints and {len(samples)} samples")
    
    # Initialize environment
    env = PhyloEnv(
        samples_parent_dir=Path(samples_dir),
        raxmlng_path=Path(raxml_path),
        horizon=HORIZON
    )
    
    # Get feature dimension
    if samples:
        sample = env.samples[0]  # Use first sample to get feature dim
        trees = get_starting_trees(sample["dir"])
        if trees:
            _, tree_nwk = trees[0]
            _, _, feats = env.reset_to_specific_tree(sample, tree_nwk)
            feature_dim = feats.shape[1]
        else:
            print("No starting trees found")
            return
    else:
        print("No samples found")
        return
    
    results = []
    
    # Cache RAxML results for each unique starting tree
    raxml_cache = {}
    
    # First pass: collect all unique starting trees and compute RAxML benchmarks
    unique_trees = set()
    for sample in env.samples:
        sample_dir = sample["dir"]
        trees = get_starting_trees(sample_dir)
        for tree_type, tree_nwk in trees:
            tree_key = (sample_dir.name, tree_type, tree_nwk)
            unique_trees.add(tree_key)
    
    print(f"Computing RAxML benchmarks for {len(unique_trees)} unique starting trees...")
    for sample_name, tree_type, tree_nwk in unique_trees:
        sample_dir = None
        for sample in env.samples:
            if sample["dir"].name == sample_name:
                sample_dir = sample["dir"]
                break
        if sample_dir:
            raxml_ll = evaluate_raxml_likelihood(sample_dir, tree_nwk, raxml_path)
            cache_key = (sample_name, tree_type)
            raxml_cache[cache_key] = raxml_ll
            print(f"Cached RAxML for {sample_name} {tree_type}: {raxml_ll:.4f}")
    
    # Now run evaluations using cached RAxML results
    for ckpt_path in checkpoints:
        print(f"Evaluating checkpoint: {ckpt_path.name}")
        
        # Load agent
        agent = DQNAgent(feature_dim)
        agent.load(ckpt_path)
        agent.epsilon_start = 0.0  # No exploration during evaluation
        agent.epsilon_end = 0.0
        
        for sample in env.samples:
            sample_dir = sample["dir"]
            max_pars_ll = get_max_pars_likelihood(sample_dir)
            
            trees = get_starting_trees(sample_dir)
            
            for tree_type, tree_nwk in trees:
                # Use cached RAxML likelihood
                cache_key = (sample_dir.name, tree_type)
                raxml_ll = raxml_cache.get(cache_key)
                
                # Evaluate agent
                likelihoods = evaluate_agent_on_tree(env, agent, sample, tree_nwk)
                
                # Record result
                result = {
                    "checkpoint": ckpt_path.name,
                    "sample": sample_dir.name,
                    "tree_type": tree_type,
                    "starting_ll": likelihoods[0],
                    "raxml_ll": raxml_ll,
                    "max_pars_ll": max_pars_ll,
                    "likelihoods": likelihoods,
                    "steps": len(likelihoods) - 1
                }
                results.append(result)
    
    # Save results
    results_file = output_dir_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for easier analysis
    csv_file = output_dir_path / "evaluation_results.csv"
    if results:
        fieldnames = ["checkpoint", "sample", "tree_type", "starting_ll", "raxml_ll", "max_pars_ll", "final_ll", "steps"]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow({
                    "checkpoint": result["checkpoint"],
                    "sample": result["sample"],
                    "tree_type": result["tree_type"],
                    "starting_ll": result["starting_ll"],
                    "raxml_ll": result["raxml_ll"],
                    "max_pars_ll": result["max_pars_ll"],
                    "final_ll": result["likelihoods"][-1],
                    "steps": result["steps"]
                })
    
    print(f"Evaluation complete. Results saved to {output_dir_path}")


def get_checkpoints(checkpoints_dir):
    """Get all checkpoint files."""
    checkpoints = []
    for file in Path(checkpoints_dir).glob("*.pt"):
        checkpoints.append(file)
    # Sort by episode number if possible
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
    return checkpoints


def get_samples(samples_dir):
    """Get all sample directories."""
    samples = []
    for sample_dir in Path(samples_dir).glob("sample_*"):
        samples.append(sample_dir)
    return samples


def get_starting_trees(sample_dir):
    """Get all starting trees for a sample."""
    trees = []
    # Parsimony trees
    pars_file = sample_dir / "raxml_pars.raxml.startTree"
    if pars_file.exists():
        with open(pars_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    trees.append(("pars", line))
    
    # Random trees
    rand_file = sample_dir / "raxml_rand.raxml.startTree"
    if rand_file.exists():
        with open(rand_file) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    trees.append((f"rand_{i}", line))
    
    return trees


def get_max_pars_likelihood(sample_dir):
    """Get the maximum parsimony likelihood from the log."""
    log_file = sample_dir / "raxml_eval_pars.raxml.log"
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if line.startswith("Final LogLikelihood:"):
                    return float(line.strip().split()[-1])
    return None


def evaluate_raxml_likelihood(sample_dir, tree_nwk, raxml_path):
    """Evaluate the likelihood of a tree using RAxML with full optimization."""
    import tempfile
    from ete3 import Tree
    
    tree = Tree(tree_nwk, format=1)
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
        tmp_path = Path(tmp)
        treefile = tmp_path / "tree.nwk"
        tree.write(outfile=str(treefile), format=1)
        
        msa_file = sample_dir / "sample.fasta"
        prefix = tmp_path / "eval"
        
        # Run RAxML with full search starting from the given tree
        cmd = [
            raxml_path,
            "--search",
            "--msa", str(msa_file),
            "--model", "GTR+I+G",  # Use standard model
            "--tree", str(treefile),
            "--prefix", str(prefix),
            "--seed", "12345"  # For reproducibility
        ]
        run_cmd(cmd, quiet=True)
        
        log_file = prefix.with_suffix(".raxml.log")
        ll = None
        with open(log_file) as f:
            for line in f:
                if line.startswith("Final LogLikelihood:"):
                    ll = float(line.strip().split()[-1])
                    break
        return ll


def evaluate_agent_on_tree(env, agent, sample, tree_nwk, max_steps=HORIZON):
    """Run agent on a specific tree and record likelihood trajectory."""
    annotated_tree, moves, feats = env.reset_to_specific_tree(sample, tree_nwk)
    likelihoods = [env.current_ll]
    steps = 0
    
    while steps < max_steps:
        action_idx, _ = agent.select_action(feats)
        feats, reward, done = env.step(action_idx)
        likelihoods.append(env.current_ll)
        steps += 1
        if done:
            break
    
    return likelihoods


if __name__ == "__main__":
    main()