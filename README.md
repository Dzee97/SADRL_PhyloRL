# SADRL_PhyloRL Experiment Guide

This guide explains how to switch between feature sets (Hand-Crafted vs. GNN), and run training/evaluation.

## 1. Choosing a Method (Switching Branches)

The project uses different git branches to separate the feature extraction methods.

### Hand-Crafted Features (Baseline)
Use the `Daniel` branch for agents that use hand-crafted topological features (DQN, SoftQ).
```bash
git checkout Daniel
```

### GNN-Based Features
Use the `main` branch for agents that use Graph Neural Networks (GNN) to learn features directly from the tree structure.
```bash
git checkout main
```

---

## 2. Configuring and Running Experiments

The main entry point for running experiments is **`experiment.py`**.

### Configuration
Open `experiment.py` to modify settings.

#### 1. Execution Flags (Bottom of file)
At the bottom of `experiment.py`, you will find flags to control the execution flow:
```python
if __name__ == "__main__":
    # Toggle these flags to control which parts run
    RUN_SAMPLING = False      # Run dataset generation
    RUN_TRAINING = True       # Run agent training
    RUN_EVALUATION = True     # Run evaluation on checkpoints
    
    # Set algorithm
    ALGORITHM = "GNN"         # Options: "DQN", "SQL", "GNN"
```
*Important: When on the `main` branch (Hand-Crafted), ensure you set `ALGORITHM` to `"DQN"` or `"SQL"`. When on `gnn`, use `"GNN"`.*

#### 2. Hyperparameters
Hyperparameters are defined in dictionaries within `experiment.py`:
- **`train_common`**: Shared settings like `episodes`, `horizon`, `learning_rate`.
- **`train_dqn` / `train_softq`**: Specific settings for those algorithms.
- **`train_gnn_cfg`** (on `gnn` branch): Architecture settings for GNN.

#### 3. Datasets
Experiment datasets are defined in the `EXPERIMENTS` dictionary:
```python
EXPERIMENTS = {
    "Size9Samples100Train100Test20": dict(
        sample_size=9, 
        num_samples=100,
        # ...
    ),
}
```

### Running the Experiment

Once configured, run the experiment using Python:

```bash
python experiment.py
```

After evaluation, results and visulatizations are saved in their respective output directories.