from sample_datasets import sample_dataset
from train_multi_agents import run_parallel_training
from evaluation import evaluate_agents, evaluate_checkpoints, plot_over_checkpoints
from pathlib import Path
from functools import partial

# --- Sampling of datasets ---

output_dir = Path("output")
samples1_train1 = output_dir / "Samples1Train1"
samples1_train10 = output_dir / "Samples1Train10"

deps_dir = Path("dependencies")
raxmlng_path = deps_dir / "raxmlng" / "raxml-ng"
mafft_path = deps_dir / "mafft-linux64" / "mafft.bat"

datasets_dir = Path("datasets")
input_fasta = datasets_dir / "051_856_p__Basidiomycota_c__Agaricomycetes_o__Russulales.fasta"

sample_size = 7
num_pars_trees = 10
num_bootstrap = 10_000
evo_model = "GTR+I+G"

sample_dataset_partial = partial(sample_dataset,
                                 input_fasta=input_fasta,
                                 sample_size=sample_size,
                                 num_pars_trees=num_pars_trees,
                                 num_bootstrap=num_bootstrap,
                                 raxmlng_path=raxmlng_path,
                                 mafft_path=mafft_path,
                                 evo_model=evo_model)

print("--- Sampling of datasets ---")
sample_dataset_partial(outdir=samples1_train1,
                       num_samples=1,
                       num_rand_train_trees=1,
                       num_rand_test_trees=0)
sample_dataset_partial(outdir=samples1_train10,
                       num_samples=1,
                       num_rand_train_trees=10,
                       num_rand_test_trees=0)

# --- Training of agents ---

samples1_train1_checkpoints = samples1_train1 / "checkpoints"
samples1_train10_checkpoints = samples1_train10 / "checkpoints"

# training loop params
checkpoint_freq = 100
update_freq = 4
batch_size = 128
episodes = 2000
horizon = 20
n_agents = 5
n_cores = 2

# dqn agent params
hidden_dim = 256
replay_size = 10_000
learning_rate = 1e-5
gamma = 0.9
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 10_000
target_update = 1000

run_parallel_training_partial = partial(run_parallel_training,
                                        # dep paths
                                        raxml_path=raxmlng_path,
                                        # training loop params
                                        episodes=episodes,
                                        horizon=horizon,
                                        n_agents=n_agents,
                                        n_cores=n_cores,
                                        checkpoint_freq=checkpoint_freq,
                                        update_freq=update_freq,
                                        batch_size=batch_size,
                                        # dqn agent params
                                        hidden_dim=hidden_dim,
                                        replay_size=replay_size,
                                        learning_rate=learning_rate,
                                        gamma=gamma,
                                        epsilon_start=epsilon_start,
                                        epsilon_end=epsilon_end,
                                        epsilon_decay=epsilon_decay,
                                        target_update=target_update)

print("--- Training of agents ---")
run_parallel_training_partial(samples_dir=samples1_train1,
                              checkpoint_dir=samples1_train1_checkpoints)
run_parallel_training_partial(samples_dir=samples1_train10,
                              checkpoint_dir=samples1_train10_checkpoints)

# --- Evaluation of agents ---

samples1_train1_evaluate = samples1_train1 / "evaluate"
samples1_train10_evaluate = samples1_train10 / "evaluate"

evaluate_checkpoints_partial = partial(evaluate_checkpoints,
                                       hidden_dim=hidden_dim,
                                       raxmlng_path=raxmlng_path,
                                       horizon=horizon)

print("--- Evaluation of agents ---")
evaluate_checkpoints_partial(samples_dir=samples1_train1,
                             start_tree_set="train",
                             checkpoints_dir=samples1_train1_checkpoints,
                             evaluate_dir=samples1_train1_evaluate)
evaluate_checkpoints_partial(samples_dir=samples1_train10,
                             start_tree_set="train",
                             checkpoints_dir=samples1_train10_checkpoints,
                             evaluate_dir=samples1_train10_evaluate)

# --- Plottig for evaluation results ---
print("--- Plottig of evaluation results ---")
plot_over_checkpoints(evaluate_dir=samples1_train1_evaluate)
plot_over_checkpoints(evaluate_dir=samples1_train10_evaluate)
