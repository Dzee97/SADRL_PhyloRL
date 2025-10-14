# Hyperparameters for the SADRL_PhyloRL project
# This file centralizes all hyperparameters for easy tuning and experimentation

# =============================================================================
# DATA SAMPLING HYPERPARAMETERS
# =============================================================================

# Number of samples to generate from the dataset
NUM_SAMPLES = 1

# Number of sequences to sample from each dataset
SAMPLE_SIZE = 7

# Number of parsimony trees to generate as starting points
NUM_PARS_TREES = 10

# Number of random trees to generate as starting points
NUM_RAND_TREES = 10

# Number of bootstrap replicates for support calculation
NUM_BOOTSTRAP = 1000

# Evolutionary model for RAxML-NG
EVO_MODEL = "GTR+I+G"

# Number of parallel jobs for bootstrap computation (-1 for all available cores)
N_JOBS_BOOTSTRAP = -1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Number of training episodes
EPISODES = 2000

# Maximum number of steps per episode (horizon)
HORIZON = 20

# Batch size for experience replay
BATCH_SIZE = 128

# =============================================================================
# MULTI-AGENT TRAINING HYPERPARAMETERS
# =============================================================================

# Number of agents to train in parallel
N_AGENTS = 5

# Number of CPU cores to use (None for auto-detection)
N_CORES = None

# Frequency of checkpointing agent models (every N episodes)
CHECKPOINT_EVERY = 100

# Frequency of DQN updates (every N steps)
UPDATE_FREQUENCY = 4

# =============================================================================
# DQN AGENT HYPERPARAMETERS
# =============================================================================

# Learning rate for the optimizer
LEARNING_RATE = 1e-4

# Discount factor for future rewards
GAMMA = 0.9

# Initial epsilon for epsilon-greedy exploration
EPSILON_START = 1.0

# Final epsilon for epsilon-greedy exploration
EPSILON_END = 0.05

# Rate of epsilon decay (higher = slower decay)
EPSILON_DECAY = 10_000

# Frequency of target network updates (every N steps)
TARGET_UPDATE = 1000

# =============================================================================
# RAINBOW DQN EXTENSIONS HYPERPARAMETERS
# =============================================================================

# Toggle for Double Q-learning
DOUBLE_Q = True

# Toggle for Prioritized Experience Replay
PRIORITIZED_REPLAY = True

# Priority exponent for Prioritized Experience Replay (alpha)
PRIORITY_ALPHA = 0.6

# Initial importance sampling weight for Prioritized Experience Replay (beta_start)
PRIORITY_BETA_START = 0.4

# Final importance sampling weight for Prioritized Experience Replay (beta_end)
PRIORITY_BETA_END = 1.0

# Steps to anneal beta from start to end
PRIORITY_BETA_ANNEAL_STEPS = 100_000

# Toggle for Dueling Network Architecture
DUELING = True

# Toggle for Multi-step learning
MULTI_STEP = True

# Number of steps for multi-step returns (n)
N_STEPS = 3

# Toggle for Distributional RL (C51)
DISTRIBUTIONAL = True

# Minimum value for distributional RL support
V_MIN = -10.0

# Maximum value for distributional RL support
V_MAX = 10.0

# Number of atoms for distributional RL
NUM_ATOMS = 51

# Toggle for Noisy Nets
NOISY_NETS = True

# Initial standard deviation for Noisy Nets
NOISY_SIGMA_INIT = 0.5

# =============================================================================
# INTRINSIC CURIOSITY HYPERPARAMETERS
# =============================================================================

# Toggle for Intrinsic Curiosity Module
INTRINSIC_CURIOSITY = False

# Intrinsic reward scaling factor
INTRINSIC_REWARD_SCALE = 0.1

# Learning rate for curiosity modules
CURIOSITY_LR = 1e-3

# =============================================================================
# NEURAL NETWORK ARCHITECTURE HYPERPARAMETERS
# =============================================================================

# Hidden layer dimension for Q-network
HIDDEN_DIM = 256

# =============================================================================
# REPLAY BUFFER HYPERPARAMETERS
# =============================================================================

# Maximum capacity of the replay buffer
REPLAY_BUFFER_CAPACITY = 10_000

# =============================================================================
# ENVIRONMENT HYPERPARAMETERS
# =============================================================================

# (Horizon is defined above in training section)

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

# Default samples directory
DEFAULT_SAMPLES_DIR = "OUTTEST"

# Default RAxML-NG executable path
DEFAULT_RAXML_PATH = "raxmlng/raxml-ng"

# Default save directory for checkpoints
DEFAULT_CHECKPOINTS_DIR = "checkpoints"