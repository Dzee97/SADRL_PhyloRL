import json
import hashlib
from pathlib import Path
from functools import partial
from sample_datasets import sample_dataset
from train_agents import run_parallel_training
from evaluation import evaluate_checkpoints, plot_final_checkpoint_tables, accuracy_over_checkpoints


# === CONFIGURATION ===

# Directories
BASE_DIR = Path("output")
DEPS_DIR = Path("dependencies")
DATASETS_DIR = Path("datasets")

# Dependencies
raxmlng_path = DEPS_DIR / "raxmlng" / "raxml-ng"
mafft_path = DEPS_DIR / "mafft-linux64" / "mafft.bat"
input_fasta = DATASETS_DIR / "051_856_p__Basidiomycota_c__Agaricomycetes_o__Russulales.fasta"

# Sampling parameters
sampling_cfg = dict(
    input_fasta=input_fasta,
    num_pars_trees=10,
    num_bootstrap=10_000,
    evo_model="GTR+I+G",
    raxmlng_path=raxmlng_path,
    mafft_path=mafft_path
)

# Experiment sets
EXPERIMENTS = {
    # Sample size 9
    # "Size9Samples1Train100Test20": dict(sample_size=9, num_samples=1,
    #                                    num_rand_train_trees=100, num_rand_test_trees=20),
    "Size9Samples100Train100Test20": dict(sample_size=9, num_samples=100,
                                          num_rand_train_trees=100, num_rand_test_trees=20),
    "Size9ValidationSet50": dict(sample_size=9, num_samples=50, num_rand_train_trees=0, num_rand_test_trees=20),
}

# Set number of cores for parallel agent training and evaluation
n_cores = 5
# Set number of agents to train in parallel
n_agents = 5

# Training parameters (shared)
train_common = dict(
    episodes=20_000,
    horizon=20,
    checkpoint_freq=1000,
    update_freq=1,
    batch_size=128,
    hidden_dim=256,
    dropout_p=0.2,
    replay_size=10_000,
    min_replay_start=1000,
    learning_rate=1e-5,
    weight_decay=1e-2,
    gamma=0.9,
    tau=0.005
)

# DQN Boltzmann-based agent parameters
dqn_cfg = dict(
    temp=1.0,
    double_q=False,
    replay_alpha=0.0,
    replay_beta_start=0.4,
    replay_beta_frames=400_000
)

# Soft DQN agent parameters
soft_cfg = dict(
    replay_alpha=0.0,
    replay_beta_start=0.4,
    replay_beta_frames=400_000,
    temp_alpha_init=4.0,
    entropy_frames=400_000,
    entropy_start=0.5,
    entropy_end=0.5
)

# Hash full parameters for file names
full_dqn_cfg = train_common | dqn_cfg
full_soft_cfg = train_common | soft_cfg

# Evaluation config
evaluate_cfg = dict(
    hidden_dim=train_common["hidden_dim"],
    raxmlng_path=raxmlng_path,
    horizon=train_common["horizon"],
    top_k_reward=1,
    n_jobs=n_cores,
)


# === HELPER FUNCTIONS ===

def stable_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def run_sampling():
    """Run dataset sampling for all experiment sets."""
    print("\n=== Sampling datasets ===")
    sample_fn = partial(sample_dataset, **sampling_cfg)
    for name, cfg in EXPERIMENTS.items():
        outdir = BASE_DIR / name
        print(f"Sampling → {name}")
        sample_fn(outdir=outdir, **cfg)


def run_training(algorithm):
    """Run training for all experiments."""
    print("\n=== Training agents ===")

    if algorithm == "DQN":
        training_hps = full_dqn_cfg
    elif algorithm == "SQL":
        training_hps = full_soft_cfg
    else:
        raise ValueError("Invalid alogirhtm name")

    hps_hash = stable_hash(training_hps)

    train_fn = partial(run_parallel_training, algorithm=algorithm, raxmlng_path=raxmlng_path, n_cores=n_cores,
                       n_agents=n_agents, training_hps=training_hps)

    for name, cfg in EXPERIMENTS.items():
        # Don't train when the dataset has no training trees
        if cfg["num_rand_train_trees"] == 0:
            continue

        samples_dir = BASE_DIR / name

        cfg_dir = samples_dir / f"{algorithm}_{hps_hash}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps(training_hps, indent=4))
        train_fn(samples_dir=samples_dir, checkpoint_dir=cfg_dir / "checkpoints")


def run_evaluation(algorithm, set_type="test"):
    """Evaluate trained checkpoints on train/test sets."""
    print("\n=== Evaluating agents ===")

    if algorithm == "DQN":
        training_hps = full_dqn_cfg
    elif algorithm == "SQL":
        training_hps = full_soft_cfg
    else:
        raise ValueError("Invalid alogirhtm name")

    hps_hash = stable_hash(training_hps)

    evaluate_fn = partial(evaluate_checkpoints, **evaluate_cfg)

    for name, cfg in EXPERIMENTS.items():
        # Datsets without training trees have no checkpoints to evaluate
        if cfg["num_rand_train_trees"] == 0:
            continue

        samples_dir = BASE_DIR / name

        # Evaluate checkpoints on test trees on self, and on validation datasets with only test trees
        evaluate_samples_dirs = {n: BASE_DIR / n for n,
                                 c in EXPERIMENTS.items() if (n == name or c["num_rand_train_trees"] == 0)
                                 and c["num_samples"] <= 50}

        for eval_name, evaluate_samples_dir in evaluate_samples_dirs.items():
            checkpoints_dir = samples_dir / f"{algorithm}_{hps_hash}" / "checkpoints"
            evaluate_dir = samples_dir / f"{algorithm}_{hps_hash}" / \
                f"evaluate_{eval_name}_topk{evaluate_cfg['top_k_reward']}"
            evaluate_fn(
                samples_dir=evaluate_samples_dir,
                start_tree_set=set_type,
                checkpoints_dir=checkpoints_dir,
                evaluate_dir=evaluate_dir
            )
            plot_final_checkpoint_tables(evaluate_dir=evaluate_dir, dataset_name=name,
                                         algorithm_name=algorithm)
            accuracy_over_checkpoints(evaluate_dir=evaluate_dir, train_dataset=name, eval_dataset=eval_name,
                                      algorithm_name=algorithm)


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # toggle these flags to control which parts run
    RUN_SAMPLING = True
    RUN_TRAINING = True
    RUN_EVALUATION = True

    # set this flag to control which algorithm to run (DQN, SQL)
    ALGORITHM = "SQL"

    if RUN_SAMPLING:
        run_sampling()
    if RUN_TRAINING:
        run_training(algorithm=ALGORITHM)
    if RUN_EVALUATION:
        run_evaluation(algorithm=ALGORITHM)
