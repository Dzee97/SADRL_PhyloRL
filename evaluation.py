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
# from dqn_agent import QNetwork
from soft_dqn_agent import QNetwork


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


def plot_over_checkpoints(evaluate_dir: Path, dataset_name: str, algorithm_name: str, loops_suffix: str):
    """
    Plot evaluation results across checkpoints for each sample.

    Shows the mean ± 95% CI over agents, with individual agent traces (thin lines),
    and the average step at which the highest LL was reached on the secondary axis.
    """
    results = np.load(evaluate_dir / "results.npy")
    pars_lls = np.load(evaluate_dir / "pars_lls.npy")
    episode_nums = np.load(evaluate_dir / "episode_nums.npy")

    n_agents, n_samples, n_checkpoints, n_start_trees, n_steps = results.shape

    plot_dir = evaluate_dir / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    for sample_idx in range(n_samples):
        sample_results = results[:, sample_idx]  # (n_agents, n_checkpoints, n_start_trees, n_steps)

        # Max LL and step index per trajectory
        episode_max = np.max(sample_results, axis=3)       # (n_agents, n_checkpoints, n_start_trees)
        episode_argmax = np.argmax(sample_results, axis=3)  # same shape

        # Average over starting trees per agent
        episode_max_mean_per_agent = np.nanmean(episode_max, axis=2)       # (n_agents, n_checkpoints)
        episode_argmax_mean_per_agent = np.nanmean(episode_argmax, axis=2)  # (n_agents, n_checkpoints)

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
        ax1.set_ylabel("Highest LL (avg over starting trees)", color=color)

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
        ax1.axhline(y=pars_lls[sample_idx], color='gray', linestyle="--", linewidth=1.5, label="Parsimony LL")

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='lower left', fontsize=9)

        # === Right axis: Step index of best LL ===
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel("Step of best LL", color=color)

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
            f"Dataset: {dataset_name} - {algorithm_name} - Sample {sample_idx+1} - "
            f"{loops_suffix.replace('_', ' ').capitalize()}\n"
            "Highest LL per episode (mean ± 95% CI, with agent traces)"
        )

        fig.tight_layout()
        plot_file = plot_dir / f"sample{sample_idx+1}.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {plot_file}")


def plot_per_agent(evaluate_dir: Path, dataset_name: str, algorithm_name: str, loops_suffix: str):
    """
    Plot per-agent evaluation results across checkpoints.

    Each agent gets its own plot.
    For each episode:
      - The line shows the highest LL found in any step across all starting trees.
      - Marker color shows how many starting trees reached ≥ parsimony LL:
        0 = red, 1–10 = light→dark green.
    """
    results = np.load(evaluate_dir / "results.npy")
    pars_lls = np.load(evaluate_dir / "pars_lls.npy")
    episode_nums = np.load(evaluate_dir / "episode_nums.npy")

    n_agents, n_samples, n_checkpoints, n_start_trees, n_steps = results.shape
    plot_dir = evaluate_dir / "plots_per_agent"
    os.makedirs(plot_dir, exist_ok=True)

    # Create a discrete color map: 0 = red, 1–10 = greens
    from matplotlib.colors import ListedColormap, BoundaryNorm

    greens = plt.cm.Greens(np.linspace(0.3, 1.0, 10))  # 10 shades of green
    colors = np.vstack(([plt.cm.Reds(0.8)], greens))   # prepend red for 0
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 11.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    for sample_idx in range(n_samples):
        sample_results = results[:, sample_idx]  # (n_agents, n_checkpoints, n_start_trees, n_steps)
        pars_ll = pars_lls[sample_idx]

        for a in range(n_agents):
            agent_results = sample_results[a]  # (n_checkpoints, n_start_trees, n_steps)

            # Highest LL per episode across *all start trees and steps*
            episode_max = np.nanmax(agent_results, axis=(1, 2))  # (n_checkpoints,)

            # Count how many start trees reached ≥ parsimony LL
            reached_counts = np.sum(np.nanmax(agent_results, axis=2) >= pars_ll, axis=1)
            reached_counts = np.clip(reached_counts, 0, 10)  # cap at 10 for color scale

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(episode_nums, episode_max, color="tab:red", linewidth=2.0, label="Highest LL per episode")

            # Add discrete colored markers
            scatter = ax.scatter(
                episode_nums, episode_max, c=reached_counts, cmap=cmap, norm=norm,
                s=70, edgecolors="black", label="Start trees ≥ pars LL"
            )

            ax.axhline(y=pars_ll, color="gray", linestyle="--", linewidth=1.5, label="Parsimony LL")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Highest log-likelihood")
            ax.set_title(
                f"Dataset: {dataset_name} - {algorithm_name}\n"
                f"Sample {sample_idx+1}, Agent {a+1} ({loops_suffix.replace('_',' ')})"
            )
            ax.grid(alpha=0.3)
            ax.legend(loc="lower left", fontsize=9)

            # Add discrete colorbar
            cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(0, 11))
            cbar.ax.set_yticklabels([str(i) for i in range(0, 11)])
            cbar.set_label("Number of start trees ≥ pars LL")

            fig.tight_layout()
            plot_file = plot_dir / f"sample{sample_idx+1}_agent{a+1}.png"
            fig.savefig(plot_file, dpi=150)
            plt.close(fig)
            print(f"Plot saved to {plot_file}")


def evaluate_checkpoints(samples_dir: Path, start_tree_set: str, checkpoints_dir: Path, hidden_dim: int,
                         evaluate_dir: Path, raxmlng_path: Path, horizon: int, forbid_loops: bool, n_jobs: int):
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
    base_env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)
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

    # ---- Worker function (evaluates one agent across all checkpoints) ----
    def eval_single_agent(agent_idx, agent_num):
        torch.set_num_threads(1)  # prevent oversubscription
        env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)  # reuse within agent
        agent_results = np.full((num_samples, n_checkpoints, max_num_start_trees, horizon + 1), np.nan)

        print(f"[Agent {agent_num}] Starting evaluation with {n_checkpoints} checkpoints")

        for checkpoint_idx, episode_num in enumerate(episode_nums):
            checkpoint_file = checkpoints_dir / f"agent_{agent_num}_ep{episode_num}.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")

            print(f"[Agent {agent_num}] → Checkpoint {checkpoint_idx+1}/{n_checkpoints} (episode {episode_num})")

            state_dict = torch.load(checkpoint_file, map_location="cpu")
            agent = EvalAgent(feature_dim, hidden_dim, state_dict, device="cpu")

            for sample_idx in range(num_samples):
                for start_tree_idx in range(num_start_trees[sample_idx]):
                    tree_hash, feats = env.reset(sample_num=sample_idx, start_tree_set=start_tree_set,
                                                 start_tree_num=start_tree_idx)
                    ep_lls = [env.current_ll]
                    visited_trees = {tree_hash}
                    done = False
                    while not done:
                        if forbid_loops:
                            action_idxs = agent.select_sorted_best_actions(feats)
                            for action_idx in action_idxs:
                                preview_tree_hash = env.preview_step(action_idx)
                                if preview_tree_hash not in visited_trees:
                                    break
                        else:
                            action_idx = agent.select_best_action(feats)
                        next_tree_hash, next_feats, reward, done = env.step(action_idx)
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
    np.save(evaluate_dir / "episode_nums.npy", episode_nums)
    print(f"✅ Evaluation completed and saved to {evaluate_dir}")


# def evaluate_agents(samples_dir, checkpoints_dir, raxml_path, horizon, n_agents=5, plot_dir="plots"):
#     """
#     Evaluate multiple trained agents on each starting tree and plot average log-likelihoods.

#     Args:
#         samples_dir (str): Parent directory with sample datasets.
#         checkpoints_dir (str): Directory containing agent_i_final.pt files.
#         raxml_path (str): Path to RAxML-NG binary.
#         horizon (int): Max steps per episode.
#         n_agents (int): Number of agents to evaluate.
#         plot_dir (str): Directory where plots will be saved.
#     """
#     samples_dir = Path(samples_dir)
#     checkpoints_dir = Path(checkpoints_dir)
#     plot_dir = Path(plot_dir)
#     plot_dir.mkdir(exist_ok=True, parents=True)

#     # Initialize environment (shared)
#     env = PhyloEnv(samples_dir, Path(raxml_path), horizon=horizon)
#     feats = env.reset()
#     feature_dim = feats.shape[1]
#     num_samples = len(env.samples)

#     # Load all agents
#     agents = []
#     for i in range(n_agents):
#         ckpt_path = checkpoints_dir / f"agent_{i}_final.pt"
#         if not ckpt_path.exists():
#             raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
#         print(f"Loading agent {i} from {ckpt_path}")
#         state_dict = torch.load(ckpt_path, map_location="cpu")
#         agents.append(EvalAgent(feature_dim, state_dict))

#     # Evaluate all agents on each starting tree
#     for sample_idx in range(num_samples):
#         pars_ll = env.samples[sample_idx]["pars_ll"]
#         num_start_trees = len(env.samples[sample_idx]["rand_trees_list"])
#         print(f"Evaluating sample {sample_idx}, start tree {start_idx}")
#         for start_idx in range(num_start_trees):
#             all_agent_lls = []

#             # Run each agent
#             for agent in agents:
#                 feats = env.reset(sample_num=sample_idx, start_tree_num=start_idx)
#                 ep_lls = [env.current_ll]
#                 done = False

#                 while not done:
#                     action_idx = agent.select_best_action(feats)
#                     next_feats, reward, done = env.step(action_idx)
#                     ep_lls.append(env.current_ll)
#                     feats = next_feats

#                 all_agent_lls.append(ep_lls)

#             # --- Aggregate results across agents ---
#             ll_matrix = np.array(all_agent_lls)  # shape: (n_agents, n_steps)
#             mean_ll = np.mean(ll_matrix, axis=0)
#             std_ll = np.std(ll_matrix, axis=0)
#             ci95 = 1.96 * std_ll / np.sqrt(len(all_agent_lls))
#             max_ll = np.max(ll_matrix, axis=0)

#             # --- Plot results ---
#             steps = np.arange(len(mean_ll))
#             plt.figure(figsize=(8, 6))
#             plt.plot(steps, mean_ll, label="Mean log-likelihood", color="b")
#             plt.fill_between(steps, mean_ll - ci95, mean_ll + ci95, alpha=0.2, color="b", label="95% CI")
#             plt.plot(steps, max_ll, label="Max log-likelihood", color="g")
#             plt.axhline(y=pars_ll, color="r", linestyle="--", label="Parsimony LL")
#             plt.xlabel("Step")
#             plt.ylabel("Log-likelihood")
#             plt.title(f"Sample {sample_idx}, Start Tree {start_idx}")
#             plt.xticks(steps)  # integer ticks
#             plt.legend()
#             plt.tight_layout()
#             plt.grid()

#             plot_path = plot_dir / f"sample{sample_idx}_start{start_idx}.png"
#             plt.savefig(plot_path, dpi=150)
#             plt.close()
#             print(f"Saved plot: {plot_path}")
