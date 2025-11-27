from joblib import Parallel, delayed
import re
import torch
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from environment import PhyloEnv
from agents import QNetwork


class EvalAgent:
    def __init__(self, feature_dim, hidden_dim, state_dict, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.q_net.load_state_dict(state_dict)
        self.q_net.eval()

    def select_best_action(self, feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(x)
            return int(torch.argmax(q_vals).item())

    def select_sorted_best_actions(self, feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(x)
            q_vals_sorted, indices_sorted = torch.sort(q_vals, descending=True)
            return indices_sorted.cpu().numpy()


def accuracy_over_checkpoints(evaluate_dir: Path, train_dataset: str, eval_dataset: str, algorithm_name: str):
    results = np.load(evaluate_dir / "results.npy")
    test_mls_all = np.load(evaluate_dir / "test_mls_all.npy")
    episode_nums = np.load(evaluate_dir / "episode_nums.npy")

    n_agents, n_samples, n_checkpoints, n_start_trees, n_steps = results.shape

    res_max = np.max(results, axis=4)
    test_mls_all_expended = test_mls_all[np.newaxis, :, np.newaxis, :]
    res_match_raxml = res_max >= test_mls_all_expended - 0.1

    res_match_raxml_count = np.sum(res_match_raxml, axis=3)

    res_match_raxml_count_sample_mean = np.mean(res_match_raxml_count, axis=1)
    res_match_raxml_count_agent_mean = np.mean(res_match_raxml_count, axis=0)
    res_match_raxml_count_agent_mean_sample_mean = np.mean(res_match_raxml_count_agent_mean, axis=0)
    # res_match_raxml_count_agent_mean_sample_std = np.std(res_match_raxml_count_agent_mean, axis=0)
    # res_match_raxml_count_agent_mean_sample_ci95 = 1.96 * \
    #    res_match_raxml_count_agent_mean_sample_std / np.sqrt(n_samples)

    res_diff_raxml = np.abs(res_max - test_mls_all_expended)
    res_diff_raxml_start_mean = np.mean(res_diff_raxml, axis=3)
    res_diff_raxml_start_mean_sample_mean = np.mean(res_diff_raxml_start_mean, axis=1)
    res_diff_raxml_start_mean_agent_mean = np.mean(res_diff_raxml_start_mean, axis=0)
    res_diff_raxml_start_mean_agent_mean_sample_mean = np.mean(res_diff_raxml_start_mean_agent_mean, axis=0)
    # res_diff_raxml_start_mean_agent_mean_sample_std = np.std(res_diff_raxml_start_mean_agent_mean, axis=0)
    # res_diff_raxml_start_mean_agent_mean_sample_ci95 = 1.96 * \
    #    res_diff_raxml_start_mean_agent_mean_sample_std / np.sqrt(n_samples)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = 'tab:red'
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Number of starting trees matching RAxML (out of {n_start_trees})", color=color)

    for a in range(n_agents):
        ax1.plot(episode_nums, res_match_raxml_count_sample_mean[a], color=color, alpha=0.4, linewidth=1.0,
                 label="_agent_trace" if a > 0 else "Agents")

    ax1.plot(episode_nums, res_match_raxml_count_agent_mean_sample_mean, color=color, linewidth=2.0,
             label="Mean")
    # ax1.fill_between(episode_nums,
    #                 res_match_raxml_count_agent_mean_sample_mean - res_match_raxml_count_agent_mean_sample_ci95,
    #                 res_match_raxml_count_agent_mean_sample_mean + res_match_raxml_count_agent_mean_sample_ci95,
    #                 alpha=0.2, color=color, label="95% CI")

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left', fontsize=9)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel("Absolute difference from RAxML optimum", color=color)

    for a in range(n_agents):
        ax2.plot(episode_nums, res_diff_raxml_start_mean_sample_mean[a], color=color, alpha=0.4, linewidth=1.0,
                 label="_agent_trace" if a > 0 else "Agents")

    ax2.plot(episode_nums, res_diff_raxml_start_mean_agent_mean_sample_mean, color=color, linewidth=2.0,
             label="Mean")
    # ax2.fill_between(episode_nums,
    #                 res_diff_raxml_start_mean_agent_mean_sample_mean - res_diff_raxml_start_mean_agent_mean_sample_ci95,
    #                 res_diff_raxml_start_mean_agent_mean_sample_mean + res_diff_raxml_start_mean_agent_mean_sample_ci95,
    #                 alpha=0.2, color=color, label="95% CI")

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right', fontsize=9)

    ax1.set_xticks(episode_nums)
    ax1.set_xticklabels(episode_nums, rotation=45)
    ax1.grid(alpha=0.3)
    ax1.set_title(
        f"{algorithm_name} - Train Dataset: {train_dataset} - Eval Dataset: {eval_dataset}\n"
        "Evaluation results across training checkpoints"
    )

    fig.tight_layout()
    plot_file = evaluate_dir / "accuracy_plot.png"
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_file}")

    for c in range(n_checkpoints):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Number of starting trees matching RAxML")
        ax1.set_ylabel("Count")

        ax1.hist(res_match_raxml_count_agent_mean[:, c], bins=n_start_trees+1)
        ax1.set_xticks(range(n_start_trees+1))
        ax1.set_xticklabels(range(n_start_trees+1))

        ax1.set_title(f"Accuracy distibution over evaluation samples - checkpoint: {episode_nums[c]}")
        fig.tight_layout()
        plot_file = evaluate_dir / f"accuracy_hist_cp{episode_nums[c]}.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Histgram saved to {plot_file}")


def plot_over_checkpoints(evaluate_dir: Path, dataset_name: str, algorithm_name: str):
    """
    Plot evaluation results across checkpoints for each sample.

    Shows the mean ± 95% CI over agents, with individual agent traces (thin lines),
    and the average step at which the highest LL was reached on the secondary axis.
    """
    results = np.load(evaluate_dir / "results.npy")
    # pars_lls = np.load(evaluate_dir / "pars_lls.npy")
    test_mls_best = np.load(evaluate_dir / "test_mls_best.npy")
    episode_nums = np.load(evaluate_dir / "episode_nums.npy")

    n_agents, n_samples, n_checkpoints, n_start_trees, n_steps = results.shape

    plot_dir = evaluate_dir / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    for sample_idx in range(n_samples):
        sample_results = results[:, sample_idx]  # (n_agents, n_checkpoints, n_start_trees, n_steps)

        # Max LL and step index per trajectory
        episode_max = np.max(sample_results, axis=3)       # (n_agents, n_checkpoints, n_start_trees)
        episode_argmax = np.argmax(sample_results, axis=3)  # same shape

        # Median over starting trees per agent
        episode_max_mean_per_agent = np.median(episode_max, axis=2)       # (n_agents, n_checkpoints)
        episode_argmax_mean_per_agent = np.median(episode_argmax, axis=2)  # (n_agents, n_checkpoints)

        # Mean and CI across agents
        episode_max_avg = np.mean(episode_max_mean_per_agent, axis=0)
        episode_max_std = np.std(episode_max_mean_per_agent, axis=0)
        episode_max_ci95 = 1.96 * episode_max_std / np.sqrt(n_agents)

        episode_argmax_avg = np.mean(episode_argmax_mean_per_agent, axis=0)
        episode_argmax_std = np.std(episode_argmax_mean_per_agent, axis=0)
        episode_argmax_ci95 = 1.96 * episode_argmax_std / np.sqrt(n_agents)

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(9, 5))

        # === Left axis: Highest LL ===
        color = 'tab:red'
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Highest LL (median over starting trees)", color=color)

        # Plot each agent as faint line
        for a in range(n_agents):
            ax1.plot(episode_nums, episode_max_mean_per_agent[a],
                     color=color, alpha=0.4, linewidth=1.0, label="_agent_trace" if a > 0 else "Agents")

        # Mean + CI
        ax1.plot(episode_nums, episode_max_avg, color=color, linewidth=2.0, label="Mean LL")
        ax1.fill_between(episode_nums,
                         episode_max_avg - episode_max_ci95,
                         episode_max_avg + episode_max_ci95,
                         alpha=0.2, color=color, label="95% CI")

        # Parsimony baseline
        ax1.axhline(y=test_mls_best[sample_idx], color='gray', linestyle="--", linewidth=1.5, label="RAxML-NG LL")

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='lower left', fontsize=9)

        # === Right axis: Step index of best LL ===
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel("Step of best LL", color=color)
        ax2.set_ylim((1, n_steps))

        # Plot each agent as faint line
        for a in range(n_agents):
            ax2.plot(episode_nums, episode_argmax_mean_per_agent[a],
                     color=color, alpha=0.4, linewidth=1.0, label="_agent_trace" if a > 0 else "Agents")

        ax2.plot(episode_nums, episode_argmax_avg, color=color, linewidth=2.0, label="Mean best step")
        ax2.fill_between(episode_nums,
                         episode_argmax_avg - episode_argmax_ci95,
                         episode_argmax_avg + episode_argmax_ci95,
                         alpha=0.2, color=color, label="95% CI (step)")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(loc="lower right", fontsize=9)

        # --- Layout ---
        ax1.set_xticks(episode_nums)
        ax1.set_xticklabels(episode_nums, rotation=45)
        ax1.grid(alpha=0.3)
        ax1.set_title(
            f"Dataset: {dataset_name} - {algorithm_name} - Sample {sample_idx+1}\n"
            "Highest LL per episode (mean ± 95% CI, with agent traces)"
        )

        fig.tight_layout()
        plot_file = plot_dir / f"sample{sample_idx+1}.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")


def plot_final_checkpoint_tables(evaluate_dir: Path, dataset_name: str, algorithm_name: str):
    """
    Create heatmap tables for each agent showing final checkpoint performance.
    Rows = samples, Columns = starting trees.
    Cells are colored based on how close the LL is to the parsimony LL.
    Each row has independent color scaling: green at/above pars LL, red at 10% below pars LL.
    """
    results = np.load(evaluate_dir / "results.npy")
    # pars_lls = np.load(evaluate_dir / "pars_lls.npy")
    test_mls_all = np.load(evaluate_dir / "test_mls_all.npy")
    test_mls_best = np.load(evaluate_dir / "test_mls_best.npy")
    episode_nums = np.load(evaluate_dir / "episode_nums.npy")

    n_agents, n_samples, n_checkpoints, n_start_trees, n_steps = results.shape

    plot_dir = evaluate_dir / "final_checkpoint_tables"
    os.makedirs(plot_dir, exist_ok=True)

    # Use the last checkpoint
    final_checkpoint_idx = -1

    for agent_idx in range(n_agents):
        # Get data for this agent's final checkpoint
        agent_results = results[agent_idx, :, final_checkpoint_idx, :, :]  # (n_samples, n_start_trees, n_steps)

        # Compute max LL and step index for each sample and start tree
        max_lls = np.nanmax(agent_results, axis=2)  # (n_samples, n_start_trees)
        argmax_steps = np.nanargmax(agent_results, axis=2)  # same shape

        # Find maximum number of valid trees across all samples
        max_valid_trees = 0
        for sample_idx in range(n_samples):
            valid_mask = ~np.isnan(max_lls[sample_idx])
            max_valid_trees = max(max_valid_trees, np.sum(valid_mask))

        # Compute color values for each sample independently
        color_values = np.full((n_samples, n_start_trees), np.nan)

        for sample_idx in range(n_samples):
            sample_lls = max_lls[sample_idx]

            for tree_idx in range(n_start_trees):
                ll_val = sample_lls[tree_idx]
                ml_val = test_mls_all[sample_idx, tree_idx]

                # Define color range: green at ml_ll, red at 10 below ml_ll
                red_threshold = ml_val - 10  # 10 below pars_ll

                if np.isnan(ll_val):
                    continue

                if ll_val >= ml_val:
                    # At or above parsimony: full green
                    color_values[sample_idx, tree_idx] = 1.0
                elif ll_val <= red_threshold:
                    # At or below 10% threshold: full red
                    color_values[sample_idx, tree_idx] = 0.0
                else:
                    # Linear interpolation between red (0) and green (1)
                    color_values[sample_idx, tree_idx] = (ll_val - red_threshold) / (ml_val - red_threshold)

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(max(10, n_start_trees), max(8, n_samples * 0.5)))

        im = ax.imshow(color_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set labels
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels([f'Sample {i+1} (RAxML-NG: {test_mls_best[i]:.1f})' for i in range(n_samples)])
        ax.set_xticks(range(n_start_trees))
        ax.set_xticklabels([f'Tree {i+1}' for i in range(n_start_trees)], rotation=45, ha='right')

        # Add text annotations: "LL\n(step)"
        for sample_idx in range(n_samples):
            for tree_idx in range(n_start_trees):
                ll_val = max_lls[sample_idx, tree_idx]
                if np.isnan(ll_val):
                    continue

                step_val = int(argmax_steps[sample_idx, tree_idx])
                color_val = color_values[sample_idx, tree_idx]
                text_color = 'white' if color_val < 0.5 else 'black'

                ax.text(tree_idx, sample_idx, f'{ll_val:.1f}\n(step {step_val})',
                        ha='center', va='center', color=text_color, fontsize=8)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Performance (per sample)', rotation=270, labelpad=20)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(['10 below RAxML-NG', '5 below RAxML-NG', '≥RAxML-NG LL'])

        # Title
        final_episode = episode_nums[final_checkpoint_idx]
        ax.set_title(
            f'{algorithm_name} - Agent {agent_idx} - Final Checkpoint (Episode {final_episode}) - '
            f'Dataset: {dataset_name}\n'
            f'Max LL and Steps Across Samples and Starting Trees',
            fontsize=11, pad=15
        )

        ax.set_xlabel('Starting Tree', fontsize=10)
        ax.set_ylabel('Sample', fontsize=10)

        plt.tight_layout()

        # Save
        plot_file = plot_dir / f"agent{agent_idx}_final.png"
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Final checkpoint tables saved to {plot_dir}")


def evaluate_checkpoints(samples_dir: Path, start_tree_set: str, checkpoints_dir: Path, hidden_dim: int,
                         evaluate_dir: Path, raxmlng_path: Path, horizon: int, add_new_features: bool,
                         top_k_reward: int, n_jobs: int):
    """
    Evaluate all agents across their checkpoints in parallel (one process per agent).
    Each agent process uses a single PhyloEnv instance to reuse cached data.
    """

    # ---- Safety check for existing output ----
    if evaluate_dir.exists():
        answer = input(f"Evaluation directory '{evaluate_dir}' already exists. Overwrite? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborting — existing output directory preserved.")
            return
        print(f"Removing existing directory: {evaluate_dir}")
        shutil.rmtree(evaluate_dir)
    os.makedirs(evaluate_dir, exist_ok=True)

    # ---- Base environment (just for metadata) ----
    base_env = PhyloEnv(samples_dir, raxmlng_path, horizon, add_new_features)
    tree_hash, feats = base_env.reset(start_tree_set="test")
    feature_dim = feats.shape[1]
    num_samples = len(base_env.samples)

    if start_tree_set == "train":
        num_start_trees = base_env.num_train_start_trees
    elif start_tree_set == "test":
        num_start_trees = base_env.num_test_start_trees
    else:
        raise ValueError('start_tree_set must either be "train" or "test"')

    max_num_start_trees = max(num_start_trees)

    # ---- Identify all agents & checkpoints ----
    checkpoints_files = list(checkpoints_dir.glob("agent_*_ep*.pt"))
    agent_nums, episode_nums = set(), set()
    for f in checkpoints_files:
        match = re.search(r"agent_(\d+)_ep(\d+)\.pt", str(f))
        if match:
            agent_nums.add(int(match.group(1)))
            episode_nums.add(int(match.group(2)))

    agent_nums = sorted(agent_nums)
    episode_nums = np.array(sorted(episode_nums))
    n_agents, n_checkpoints = len(agent_nums), len(episode_nums)

    results = np.full((n_agents, num_samples, n_checkpoints, max_num_start_trees, horizon + 1), np.nan)
    pars_lls = np.array([base_env.samples[i]["pars_ll"] for i in range(num_samples)])
    test_mls_all = np.array([base_env.samples[i]["rand_test_trees_ml_list"] for i in range(num_samples)])
    test_mls_best = np.array([base_env.samples[i]["rand_test_trees_ml_best"] for i in range(num_samples)])

    # ---- Worker function (evaluates one agent across all checkpoints) ----
    def eval_single_agent(agent_idx, agent_num):
        torch.set_num_threads(1)  # prevent oversubscription
        env = PhyloEnv(samples_dir, raxmlng_path, horizon, add_new_features)  # reuse within agent
        agent_results = np.full((num_samples, n_checkpoints, max_num_start_trees, horizon + 1), np.nan)

        print(f"[Agent {agent_num}] Starting evaluation with {n_checkpoints} checkpoints")

        for checkpoint_idx, episode_num in enumerate(episode_nums):
            checkpoint_file = checkpoints_dir / f"agent_{agent_num}_ep{episode_num}.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")

            print(f"[Agent {agent_num}] → Checkpoint {checkpoint_idx+1}/{n_checkpoints} (episode {episode_num})")

            state_dict = torch.load(checkpoint_file, map_location="cpu")
            agent = EvalAgent(feature_dim, hidden_dim, state_dict)

            for sample_idx in range(num_samples):
                for start_tree_idx in range(num_start_trees[sample_idx]):
                    tree_hash, feats = env.reset(sample_num=sample_idx, start_tree_set=start_tree_set,
                                                 start_tree_num=start_tree_idx)
                    ep_lls = [env.current_ll]
                    visited_trees = {tree_hash}
                    done = False
                    while not done:
                        action_idxs = agent.select_sorted_best_actions(feats)
                        highest_reward = -np.inf
                        selected_action_idx = None
                        actions_checked = 0

                        for action_idx in action_idxs:
                            preview_tree_hash, preview_reward = env.preview_step(action_idx, calc_reward=True)
                            if preview_tree_hash in visited_trees:
                                continue

                            if preview_reward > highest_reward:
                                highest_reward = preview_reward
                                selected_action_idx = action_idx

                            actions_checked += 1
                            if actions_checked == top_k_reward:
                                break

                        next_tree_hash, next_feats, reward, done = env.step(selected_action_idx)
                        visited_trees.add(next_tree_hash)
                        ep_lls.append(env.current_ll)
                        feats = next_feats
                    agent_results[sample_idx, checkpoint_idx, start_tree_idx, :len(ep_lls)] = ep_lls

        print(f"[Agent {agent_num}] Finished evaluation.")
        return agent_idx, agent_results

    # ---- Run in parallel (one job per agent) ----
    print(f"Starting evaluation of {n_agents} agents in parallel (n_jobs={n_jobs})...")
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(eval_single_agent)(agent_idx, agent_num)
        for agent_idx, agent_num in enumerate(agent_nums)
    )

    # ---- Merge results ----
    for agent_idx, agent_results in results_list:
        results[agent_idx, :, :, :, :] = agent_results

    # ---- Save outputs ----
    np.save(evaluate_dir / "results.npy", results)
    np.save(evaluate_dir / "pars_lls.npy", pars_lls)
    np.save(evaluate_dir / "test_mls_all.npy", test_mls_all)
    np.save(evaluate_dir / "test_mls_best.npy", test_mls_best)
    np.save(evaluate_dir / "episode_nums.npy", episode_nums)
    print(f"✅ Evaluation completed and saved to {evaluate_dir}")
