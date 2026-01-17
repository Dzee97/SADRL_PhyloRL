#!/bin/bash
#SBATCH --job-name=gnn_phylo_experiment
#SBATCH --partition=gpu-a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=167:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===================================================================
# GNN Phylogenetic Tree Search Experiment
# ===================================================================
# This script runs training and evaluation for GNN-based Soft Q agents
# on phylogenetic tree search tasks using 80GB A100 GPU.
# ===================================================================

echo "======================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "======================================"

# --- Module Loading ---
# Note: NOT loading PyTorch/cuDNN modules to avoid conflicts with conda environment
# The conda environment already has all necessary packages installed
if command -v module &>/dev/null; then
    echo "Loading CUDA module only..."
    module load CUDA/12.1.1
    echo "CUDA module loaded successfully"
else
    echo "Warning: module command not found, skipping module loads"
fi

# --- Environment Setup ---
echo "Setting up Python environment: gnn"
export PATH="$HOME/.conda/envs/gnn/bin:$PATH"
export PYTHONPATH="$HOME/.conda/envs/gnn/lib/python3.9/site-packages:$PYTHONPATH"

# Verify Python is accessible
if ! command -v python &>/dev/null; then
    echo "ERROR: Python not found in conda environment"
    exit 1
fi

echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"

# --- Environment Variables ---
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# --- Working Directory ---
cd /home/s4483480/IMPROV/SADRL_PhyloRL
echo "Working directory: $(pwd)"

# ===================================================================
# EXPERIMENT CONFIGURATION
# ===================================================================
# Set the algorithm to run: DQN, SQL, or GNN
export ALGORITHM="GNN"

# Toggle experiment phases
export RUN_SAMPLING=False    # Set to True if you need to sample datasets
export RUN_TRAINING=True     # Train GNN agents
export RUN_EVALUATION=True   # Evaluate trained agents

echo "======================================"
echo "Experiment Configuration:"
echo "  Algorithm: $ALGORITHM"
echo "  Run Sampling: $RUN_SAMPLING"
echo "  Run Training: $RUN_TRAINING"
echo "  Run Evaluation: $RUN_EVALUATION"
echo "======================================"

# --- Run Experiment ---
echo "Starting experiment at $(date)..."

python experiment.py <<EOF
# This inline script modifies the flags in experiment.py
import sys
sys.exit(0)  # experiment.py reads from environment or uses defaults
EOF

# Alternatively, you can directly modify experiment.py flags:
python -c "
import sys
sys.path.insert(0, '.')
from experiment import *

# Override flags
RUN_SAMPLING = ${RUN_SAMPLING}
RUN_TRAINING = ${RUN_TRAINING}
RUN_EVALUATION = ${RUN_EVALUATION}
ALGORITHM = '${ALGORITHM}'

print(f'Running with: ALGORITHM={ALGORITHM}, TRAINING={RUN_TRAINING}, EVALUATION={RUN_EVALUATION}')

if RUN_SAMPLING:
    run_sampling()
if RUN_TRAINING:
    run_training(algorithm=ALGORITHM)
if RUN_EVALUATION:
    run_evaluation(algorithm=ALGORITHM)
"

exit_code=$?

# ===================================================================
# JOB COMPLETION
# ===================================================================
echo "======================================"
echo "Job finished at: $(date)"
echo "Exit code: $exit_code"
echo "======================================"

# Print GPU memory usage summary
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi

exit $exit_code
