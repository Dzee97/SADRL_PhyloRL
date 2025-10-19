import json
import hashlib
from pathlib import Path
from functools import partial
from sample_datasets import sample_dataset
from train_multi_agents import run_parallel_training
from rainbow_train_multi_agents import rainbow_run_parallel_training
from soft_train_multi_agents import soft_run_parallel_training
from evaluation import evaluate_checkpoints, plot_over_checkpoints


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
    sample_size=7,
    num_pars_trees=10,
    num_bootstrap=10_000,
    evo_model="GTR+I+G",
    raxmlng_path=raxmlng_path,
    mafft_path=mafft_path
)

# Experiment sets
EXPERIMENTS = {
    "Samples1Train10Test10": dict(num_samples=1, num_rand_train_trees=10, num_rand_test_trees=10),
    "Samples10Train10Test10": dict(num_samples=10, num_rand_train_trees=10, num_rand_test_trees=10),
}

# Training parameters (shared)
train_common = dict(
    episodes=3000,
    horizon=20,
    n_agents=5,
    n_cores=2,
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

# DQN epsilon-based agent parameters
dqn_cfg = dict(
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=10_000,
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
    beta_start=0.4,
    alpha=0.2,
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

    dqn_fn = partial(run_parallel_training, raxmlng_path=raxmlng_path, **train_common, **dqn_cfg)
    rainbow_fn = partial(rainbow_run_parallel_training, raxmlng_path=raxmlng_path, **train_common, **rainbow_cfg)
    soft_fn = partial(soft_run_parallel_training, raxmlng_path=raxmlng_path, **train_common, **soft_cfg)

    for name in EXPERIMENTS.keys():
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


def run_evaluation(eval_dqn, eval_rainbow, eval_soft):
    """Evaluate trained checkpoints on train/test sets."""
    print("\n=== Evaluating agents ===")
    evaluate_fn = partial(evaluate_checkpoints, **evaluate_cfg)

    for name, cfg in EXPERIMENTS.items():
        samples_dir = BASE_DIR / name
        for set_type in ["train", "test"]:
            # skip test evaluation if not available
            if cfg["num_rand_test_trees"] == 0 and set_type == "test":
                continue
            if eval_dqn:
                checkpoints_dir = samples_dir / f"dqn_{dqn_cfg_hash}" / "checkpoints"
                evaluate_fn(
                    samples_dir=samples_dir,
                    start_tree_set=set_type,
                    checkpoints_dir=checkpoints_dir,
                    evaluate_dir=samples_dir / f"dqn_{dqn_cfg_hash}" / f"evaluate_{set_type}"
                )
            if eval_rainbow:
                checkpoints_dir = samples_dir / f"rainbow_{rainbow_cfg_hash}" / "checkpoints"
                evaluate_fn(
                    samples_dir=samples_dir,
                    start_tree_set=set_type,
                    checkpoints_dir=checkpoints_dir,
                    evaluate_dir=samples_dir / f"rainbow_{rainbow_cfg_hash}" / f"evaluate_{set_type}"
                )
            if eval_soft:
                checkpoints_dir = samples_dir / f"soft_{soft_cfg_hash}" / "checkpoints"
                evaluate_fn(
                    samples_dir=samples_dir,
                    start_tree_set=set_type,
                    checkpoints_dir=checkpoints_dir,
                    evaluate_dir=samples_dir / f"soft_{soft_cfg_hash}" / f"evaluate_{set_type}"
                )


def run_plotting(plot_dqn, plot_rainbow, plot_soft):
    """Generate plots for evaluation results."""
    print("\n=== Plotting results ===")
    for name, cfg in EXPERIMENTS.items():
        samples_dir = BASE_DIR / name
        for set_type in ["train", "test"]:
            if cfg["num_rand_test_trees"] == 0 and set_type == "test":
                continue
            if plot_dqn:
                eval_dir = samples_dir / f"dqn_{dqn_cfg_hash}" / f"evaluate_{set_type}"
                plot_over_checkpoints(evaluate_dir=eval_dir)
            if plot_rainbow:
                eval_dir = samples_dir / f"rainbow_{rainbow_cfg_hash}" / f"evaluate_{set_type}"
                plot_over_checkpoints(evaluate_dir=eval_dir)
            if plot_soft:
                eval_dir = samples_dir / f"soft_{soft_cfg_hash}" / f"evaluate_{set_type}"
                plot_over_checkpoints(evaluate_dir=eval_dir)


# === MAIN EXECUTION ===

if __name__ == "__main__":
    # toggle these flags to control which parts run
    RUN_SAMPLING = True
    RUN_TRAINING = True
    RUN_EVALUATION = True
    RUN_PLOTTING = True

    # toggle these flags to control which algorithms to run
    DQN = False
    RAINBOW = False
    SOFT = True

    if RUN_SAMPLING:
        run_sampling()
    if RUN_TRAINING:
        run_training(train_dqn=DQN, train_rainbow=RAINBOW, train_soft=SOFT)
    if RUN_EVALUATION:
        run_evaluation(eval_dqn=DQN, eval_rainbow=RAINBOW, eval_soft=SOFT)
    if RUN_PLOTTING:
        run_plotting(plot_dqn=DQN, plot_rainbow=RAINBOW, plot_soft=SOFT)
