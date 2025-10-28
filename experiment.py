import json
import hashlib
from pathlib import Path
from functools import partial
from sample_datasets import sample_dataset
from train_multi_agents import run_parallel_training
from rainbow_train_multi_agents import rainbow_run_parallel_training
from soft_train_multi_agents import soft_run_parallel_training
from evaluation import evaluate_checkpoints, plot_over_checkpoints, plot_final_checkpoint_tables


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
    # Sample size 7
    # "Size7Samples1Train100Test10": dict(sample_size=7, num_samples=1,
    #                                    num_rand_train_trees=100, num_rand_test_trees=10),
    # "Size7Samples20Train100Test10": dict(sample_size=7, num_samples=20,
    #                                     num_rand_train_trees=100, num_rand_test_trees=10),
    # "Size7Samples100Train100Test10": dict(sample_size=9, num_samples=100,
    #                                      num_rand_train_trees=100, num_rand_test_trees=10),
    # "Size7ValidationSet": dict(sample_size=7, num_samples=20, num_rand_train_trees=0, num_rand_test_trees=10),
    # Sample size 9
    # "Size9Samples1Train100Test10": dict(sample_size=9, num_samples=1,
    #                                    num_rand_train_trees=100, num_rand_test_trees=10),
    "Size9Samples20Train100Test10": dict(sample_size=9, num_samples=20,
                                         num_rand_train_trees=100, num_rand_test_trees=10),
}

# Set number of cores for parallel agent training and evaluation
n_cores = 5

# Training parameters (shared)
train_common = dict(
    episodes=6000,
    horizon=20,
    n_agents=5,
    checkpoint_freq=100,
    update_freq=1,
    batch_size=128,
    hidden_dim=256,
    replay_size=10_000,
    min_replay_start=1000,
    learning_rate=1e-4,
    gamma=0.9,
    tau=0.005
)

# DQN Boltzmann-based agent parameters
dqn_cfg = dict(
    temp=1.0,
)

# Rainbow DQN agent parameters
rainbow_cfg = dict(
    sigma_init=0.5,
    alpha=0.6,
    beta_start=0.4,
)

# Soft DQN agent parameters
soft_cfg = dict(
    replay_alpha=0.6,
    replay_beta_start=0.4,
    replay_beta_frames=100_000,
    temp_alpha_init=2.0,
    temp_alpha_frames=100_000
)

# Hash full parameters for file names
full_dqn_cfg = train_common | dqn_cfg
full_rainbow_cfg = train_common | rainbow_cfg
full_soft_cfg = train_common | soft_cfg


def stable_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


dqn_cfg_hash = stable_hash(full_dqn_cfg)
rainbow_cfg_hash = stable_hash(full_rainbow_cfg)
soft_cfg_hash = stable_hash(full_soft_cfg)

# Evaluation config
evaluate_cfg = dict(
    hidden_dim=train_common["hidden_dim"],
    raxmlng_path=raxmlng_path,
    horizon=train_common["horizon"],
    n_jobs=n_cores,
)


# === HELPER FUNCTIONS ===

def run_sampling():
    """Run dataset sampling for all experiment sets."""
    print("\n=== Sampling datasets ===")
    sample_fn = partial(sample_dataset, **sampling_cfg)
    for name, cfg in EXPERIMENTS.items():
        outdir = BASE_DIR / name
        print(f"Sampling → {name}")
        sample_fn(outdir=outdir, **cfg)


def run_training(train_dqn, train_rainbow, train_soft):
    """Run training for all experiments."""
    print("\n=== Training agents ===")

    dqn_fn = partial(run_parallel_training, raxmlng_path=raxmlng_path, n_cores=n_cores,
                     **train_common, **dqn_cfg)
    rainbow_fn = partial(rainbow_run_parallel_training, raxmlng_path=raxmlng_path, n_cores=n_cores,
                         **train_common, **rainbow_cfg)
    soft_fn = partial(soft_run_parallel_training, raxmlng_path=raxmlng_path, n_cores=n_cores,
                      **train_common, **soft_cfg)

    for name, cfg in EXPERIMENTS.items():
        # Don't train when the dataset has no training trees
        if cfg["num_rand_train_trees"] == 0:
            continue

        samples_dir = BASE_DIR / name

        if train_dqn:
            cfg_dir = samples_dir / f"dqn_{dqn_cfg_hash}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.json").write_text(json.dumps(full_dqn_cfg, indent=4))
            dqn_fn(samples_dir=samples_dir, checkpoint_dir=cfg_dir / "checkpoints")
        if train_rainbow:
            cfg_dir = samples_dir / f"rainbow_{rainbow_cfg_hash}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.json").write_text(json.dumps(full_rainbow_cfg, indent=4))
            rainbow_fn(samples_dir=samples_dir, checkpoint_dir=cfg_dir / "checkpoints")
        if train_soft:
            cfg_dir = samples_dir / f"soft_{soft_cfg_hash}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.json").write_text(json.dumps(full_soft_cfg, indent=4))
            soft_fn(samples_dir=samples_dir, checkpoint_dir=cfg_dir / "checkpoints")


def run_evaluation(eval_dqn, eval_rainbow, eval_soft, set_type="test"):
    """Evaluate trained checkpoints on train/test sets."""
    print("\n=== Evaluating agents ===")
    evaluate_fn = partial(evaluate_checkpoints, **evaluate_cfg)

    for name, cfg in EXPERIMENTS.items():
        # Datsets without training trees have no checkpoints to evaluate
        if cfg["num_rand_train_trees"] == 0:
            continue

        samples_dir = BASE_DIR / name

        # Evaluate checkpoints on test trees on self, and on validation datasets with only test trees
        evaluate_samples_dirs = {n: BASE_DIR / n for n,
                                 c in EXPERIMENTS.items() if n == name or c["num_rand_train_trees"] == 0}

        for eval_name, evaluate_samples_dir in evaluate_samples_dirs.items():
            for loops_suffix, forbid_loops in [("allow_loops", False), ("forbid_loops", True)]:
                if eval_dqn:
                    checkpoints_dir = samples_dir / f"dqn_{dqn_cfg_hash}" / "checkpoints"
                    evaluate_dir = samples_dir / f"dqn_{dqn_cfg_hash}" / f"evaluate_{eval_name}_{loops_suffix}"
                    evaluate_fn(
                        samples_dir=evaluate_samples_dir,
                        start_tree_set=set_type,
                        checkpoints_dir=checkpoints_dir,
                        forbid_loops=forbid_loops,
                        evaluate_dir=evaluate_dir
                    )
                    plot_over_checkpoints(evaluate_dir=evaluate_dir, dataset_name=name, algorithm_name="DQN",
                                          loops_suffix=loops_suffix)
                if eval_rainbow:
                    checkpoints_dir = samples_dir / f"rainbow_{rainbow_cfg_hash}" / "checkpoints"
                    evaluate_dir = samples_dir / f"rainbow_{rainbow_cfg_hash}" / f"evaluate_{eval_name}_{loops_suffix}"
                    evaluate_fn(
                        samples_dir=evaluate_samples_dir,
                        start_tree_set=set_type,
                        checkpoints_dir=checkpoints_dir,
                        forbid_loops=forbid_loops,
                        evaluate_dir=evaluate_dir
                    )
                    plot_over_checkpoints(evaluate_dir=evaluate_dir, dataset_name=name, algorithm_name="Rainbow",
                                          loops_suffix=loops_suffix)
                if eval_soft:
                    checkpoints_dir = samples_dir / f"soft_{soft_cfg_hash}" / "checkpoints"
                    evaluate_dir = samples_dir / f"soft_{soft_cfg_hash}" / f"evaluate_{eval_name}_{loops_suffix}"
                    evaluate_fn(
                        samples_dir=evaluate_samples_dir,
                        start_tree_set=set_type,
                        checkpoints_dir=checkpoints_dir,
                        forbid_loops=forbid_loops,
                        evaluate_dir=evaluate_dir
                    )
                    plot_over_checkpoints(evaluate_dir=evaluate_dir, dataset_name=name,
                                          algorithm_name="Soft Q-Learning", loops_suffix=loops_suffix)
                    plot_final_checkpoint_tables(evaluate_dir=evaluate_dir, dataset_name=name,
                                                 algorithm_name="Soft Q-Learning", loops_suffix=loops_suffix)


def run_plotting(plot_dqn, plot_rainbow, plot_soft):
    """Generate plots for evaluation results."""
    print("\n=== Plotting results ===")
    for name, cfg in EXPERIMENTS.items():
        samples_dir = BASE_DIR / name
        for loops_suffix in ["allow_loops", "forbid_loops"]:
            if plot_dqn:
                eval_dir = samples_dir / f"dqn_{dqn_cfg_hash}" / f"evaluate_{loops_suffix}"
                plot_over_checkpoints(evaluate_dir=eval_dir, dataset_name=name, algorithm_name="DQN",
                                      loops_suffix=loops_suffix)
            if plot_rainbow:
                eval_dir = samples_dir / f"rainbow_{rainbow_cfg_hash}" / f"evaluate_{loops_suffix}"
                plot_over_checkpoints(evaluate_dir=eval_dir, dataset_name=name,
                                      algorithm_name="Rainbow", loops_suffix=loops_suffix)
            if plot_soft:
                eval_dir = samples_dir / f"soft_{soft_cfg_hash}" / f"evaluate_{loops_suffix}"
                plot_over_checkpoints(evaluate_dir=eval_dir, dataset_name=name,
                                      algorithm_name="Soft Q-Network", loops_suffix=loops_suffix)


# === MAIN EXECUTION ===

if __name__ == "__main__":
    # toggle these flags to control which parts run
    RUN_SAMPLING = True
    RUN_TRAINING = True
    RUN_EVALUATION = True
    RUN_PLOTTING = True

    # toggle these flags to control which algorithms to run
    DQN = True
    RAINBOW = False
    SOFT = True

    if RUN_SAMPLING:
        run_sampling()
    if RUN_TRAINING:
        run_training(train_dqn=DQN, train_rainbow=RAINBOW, train_soft=SOFT)
    if RUN_EVALUATION:
        run_evaluation(eval_dqn=DQN, eval_rainbow=RAINBOW, eval_soft=SOFT)
    # if RUN_PLOTTING:
    #    run_plotting(plot_dqn=DQN, plot_rainbow=RAINBOW, plot_soft=SOFT)
