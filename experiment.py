from sample_datasets import sample_dataset
from train_multi_agents import run_parallel_training
from evaluation import evaluate_agents, evaluate_checkpoints, plot_over_checkpoints
from pathlib import Path

samples1_trees1 = Path("Samples1Trees1")
samples1_trees10_1 = Path("Samples1Trees10Rep1")
samples1_trees10_2 = Path("Samples1Trees10Rep2")
samples10_trees10 = Path("Samples10Trees10")

samples_dirs = [samples1_trees1, samples1_trees10_1, samples1_trees10_2, samples10_trees10]

raxmlng_path = Path("raxmlng/raxml-ng")


def create_datasets():
    input_fasta = Path("datasets/051_856_p__Basidiomycota_c__Agaricomycetes_o__Russulales.fasta")
    sample_size = 7
    num_pars_trees = 10
    num_bootstrap = 1000
    evo_model = "GTR+I+G"

    sample_dataset(input_fasta=input_fasta, outdir=samples1_trees1, num_samples=1, sample_size=sample_size,
                   num_pars_trees=num_pars_trees, num_rand_trees=1, num_bootstrap=num_bootstrap,
                   raxmlng_path=raxmlng_path, evo_model=evo_model)

    sample_dataset(input_fasta=input_fasta, outdir=samples1_trees10_1, num_samples=1, sample_size=sample_size,
                   num_pars_trees=num_pars_trees, num_rand_trees=10, num_bootstrap=num_bootstrap,
                   raxmlng_path=raxmlng_path, evo_model=evo_model)

    sample_dataset(input_fasta=input_fasta, outdir=samples1_trees10_2, num_samples=1, sample_size=sample_size,
                   num_pars_trees=num_pars_trees, num_rand_trees=10, num_bootstrap=num_bootstrap,
                   raxmlng_path=raxmlng_path, evo_model=evo_model)

    sample_dataset(input_fasta=input_fasta, outdir=samples10_trees10, num_samples=10, sample_size=sample_size,
                   num_pars_trees=num_pars_trees, num_rand_trees=10, num_bootstrap=num_bootstrap,
                   raxmlng_path=raxmlng_path, evo_model=evo_model)

    return [samples1_trees1, samples1_trees10_1, samples1_trees10_2, samples10_trees10]


if __name__ == "__main__":
    # create_datasets()

    # for samples_dir in samples_dirs:
    #    print(f"Training on dataset {samples_dir}")
    #    run_parallel_training(
    #        samples_dir=samples_dir,
    #        save_dir=samples_dir / "checkpoints",
    #        raxml_path=raxmlng_path,
    #        episodes=2000,
    #        horizon=20,
    #        n_agents=5,
    #        n_cores=3
    #    )

    # for samples_data_dir in samples_dirs:
    #    for samples_model_dir in samples_dirs:
    #        print(f"Evaluating on dataset {samples_data_dir}, using agents trained on {samples_model_dir}")
    #        evaluate_agents(
    #            samples_dir=samples_data_dir,
    #            checkpoints_dir=samples_model_dir / "checkpoints",
    #            raxml_path=raxmlng_path,
    #            horizon=20,
    #            n_agents=5,
    #            plot_dir=samples_data_dir / "eval_plots" / samples_model_dir
    #        )

    results, pars_lls, episode_nums = evaluate_checkpoints(
        samples_dir=samples10_trees10,
        checkpoints_dir=samples10_trees10 / "checkpoints",
        raxml_path=raxmlng_path,
        horizon=20
    )

    plot_over_checkpoints(results, pars_lls, episode_nums)
