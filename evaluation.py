import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from environment import PhyloEnv
from dqn_agent import QNetwork


class EvalAgent:
    def __init__(self, feature_dim, state_dict, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(feature_dim).to(self.device)
        self.q_net.load_state_dict(state_dict)

    def select_best_action(self, feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(x)
            return int(torch.argmax(q_vals).item())


def plot_over_checkpoints(results: np.ndarray, pars_lls, episode_nums):
    """
    Plot evaluation results across checkpoints for each sample.

    Shows the mean ± 95% CI over agents, with individual agent traces (thin lines),
    and the average step at which the highest LL was reached on the secondary axis.
    """
    n_agents, n_checkpoints, n_samples, n_start_trees, n_steps = results.shape

    for sample_idx in range(n_samples):
        sample_results = results[:, :, sample_idx]  # (n_agents, n_checkpoints, n_start_trees, n_steps)

        # Max LL and step index per trajectory
        episode_max = np.max(sample_results, axis=3)       # (n_agents, n_checkpoints, n_start_trees)
        episode_argmax = np.argmax(sample_results, axis=3)  # same shape

        # Average over starting trees per agent
        episode_max_mean_per_agent = np.mean(episode_max, axis=2)       # (n_agents, n_checkpoints)
        episode_argmax_mean_per_agent = np.mean(episode_argmax, axis=2)  # (n_agents, n_checkpoints)

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
                     color=color, alpha=0.3, linewidth=1.0, label="_agent_trace" if a > 0 else "Agents")

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

        ax2.plot(episode_nums, episode_argmax_avg, color=color, linewidth=2.0, label="Mean best step")
        # ax2.fill_between(episode_nums,
        #                 episode_argmax_avg - episode_argmax_ci95,
        #                 episode_argmax_avg + episode_argmax_ci95,
        #                 alpha=0.2, color=color, label="95% CI (step)")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc="lower right", fontsize=9)

        # --- Layout ---
        ax1.set_xticks(episode_nums)
        ax1.set_xticklabels(episode_nums, rotation=45)
        ax1.grid(alpha=0.3)
        ax1.set_title(
            f"Sample {sample_idx} — Highest LL per episode (mean ± 95% CI, with agent traces)"
        )

        fig.tight_layout()
        plt.savefig(f"sample{sample_idx}_LL_over_checkpoints.png", dpi=150)
        plt.show()


def evaluate_checkpoints(samples_dir, checkpoints_dir, raxml_path, horizon):
    samples_dir = Path(samples_dir)
    checkpoints_dir = Path(checkpoints_dir)
    env = PhyloEnv(samples_dir, Path(raxml_path), horizon=horizon)

    feats = env.reset()
    feature_dim = feats.shape[1]
    num_samples = len(env.samples)
    num_start_trees = len(env.current_sample["rand_trees_list"])

    checkpoints_files = checkpoints_dir.glob("agent_*_ep*.pt")
    agent_nums = set()
    episode_nums = set()

    for f in checkpoints_files:
        match = re.search(r"agent_(\d+)_ep(\d+)\.pt", str(f))
        if match:
            agent_nums.add(int(match.group(1)))
            episode_nums.add(int(match.group(2)))

    agent_nums = sorted(agent_nums)
    episode_nums = sorted(episode_nums)

    n_agents = len(agent_nums)
    n_checkpoints = len(episode_nums)

    results = np.zeros((n_agents, n_checkpoints, num_samples, num_start_trees, horizon+1))
    pars_lls = [env.samples[i]["pars_ll"] for i in range(num_samples)]

    for agent_idx in range(n_agents):
        for checkpoint_idx in range(n_checkpoints):
            agent_num = agent_nums[agent_idx]
            episode_num = episode_nums[checkpoint_idx]
            checkpoint_file = checkpoints_dir / f"agent_{agent_num}_ep{episode_num}.pt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_file}")
            print(f"Loading agent {agent_num} for episode checkpoint {episode_num}")
            state_dict = torch.load(checkpoint_file)

            agent = EvalAgent(feature_dim, state_dict)
            for sample_idx in range(num_samples):
                print(f"Evaluating sample {sample_idx}")
                for start_tree_idx in range(num_start_trees):
                    feats = env.reset(sample_num=sample_idx, start_tree_num=start_tree_idx)
                    ep_lls = [env.current_ll]
                    done = False

                    while not done:
                        action_idx = agent.select_best_action(feats)
                        next_feats, reward, done = env.step(action_idx)
                        ep_lls.append(env.current_ll)
                        feats = next_feats

                    results[agent_idx, checkpoint_idx, sample_idx, start_tree_idx] = ep_lls

    return results, pars_lls, episode_nums


def evaluate_agents(samples_dir, checkpoints_dir, raxml_path, horizon, n_agents=5, plot_dir="plots"):
    """
    Evaluate multiple trained agents on each starting tree and plot average log-likelihoods.

    Args:
        samples_dir (str): Parent directory with sample datasets.
        checkpoints_dir (str): Directory containing agent_i_final.pt files.
        raxml_path (str): Path to RAxML-NG binary.
        horizon (int): Max steps per episode.
        n_agents (int): Number of agents to evaluate.
        plot_dir (str): Directory where plots will be saved.
    """
    samples_dir = Path(samples_dir)
    checkpoints_dir = Path(checkpoints_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True, parents=True)

    # Initialize environment (shared)
    env = PhyloEnv(samples_dir, Path(raxml_path), horizon=horizon)
    feats = env.reset()
    feature_dim = feats.shape[1]
    num_samples = len(env.samples)

    # Load all agents
    agents = []
    for i in range(n_agents):
        ckpt_path = checkpoints_dir / f"agent_{i}_final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        print(f"Loading agent {i} from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        agents.append(EvalAgent(feature_dim, state_dict))

    # Evaluate all agents on each starting tree
    for sample_idx in range(num_samples):
        pars_ll = env.samples[sample_idx]["pars_ll"]
        num_start_trees = len(env.samples[sample_idx]["rand_trees_list"])
        print(f"Evaluating sample {sample_idx}, start tree {start_idx}")
        for start_idx in range(num_start_trees):
            all_agent_lls = []

            # Run each agent
            for agent in agents:
                feats = env.reset(sample_num=sample_idx, start_tree_num=start_idx)
                ep_lls = [env.current_ll]
                done = False

                while not done:
                    action_idx = agent.select_best_action(feats)
                    next_feats, reward, done = env.step(action_idx)
                    ep_lls.append(env.current_ll)
                    feats = next_feats

                all_agent_lls.append(ep_lls)

            # --- Aggregate results across agents ---
            ll_matrix = np.array(all_agent_lls)  # shape: (n_agents, n_steps)
            mean_ll = np.mean(ll_matrix, axis=0)
            std_ll = np.std(ll_matrix, axis=0)
            ci95 = 1.96 * std_ll / np.sqrt(len(all_agent_lls))
            max_ll = np.max(ll_matrix, axis=0)

            # --- Plot results ---
            steps = np.arange(len(mean_ll))
            plt.figure(figsize=(8, 6))
            plt.plot(steps, mean_ll, label="Mean log-likelihood", color="b")
            plt.fill_between(steps, mean_ll - ci95, mean_ll + ci95, alpha=0.2, color="b", label="95% CI")
            plt.plot(steps, max_ll, label="Max log-likelihood", color="g")
            plt.axhline(y=pars_ll, color="r", linestyle="--", label="Parsimony LL")
            plt.xlabel("Step")
            plt.ylabel("Log-likelihood")
            plt.title(f"Sample {sample_idx}, Start Tree {start_idx}")
            plt.xticks(steps)  # integer ticks
            plt.legend()
            plt.tight_layout()
            plt.grid()

            plot_path = plot_dir / f"sample{sample_idx}_start{start_idx}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    evaluate_agents(
        samples_dir="OUTTEST102",
        checkpoints_dir="OUTTEST1010/checkpoints",
        raxml_path="raxmlng/raxml-ng",
        horizon=20,
        n_agents=5,
        plot_dir="OUTTEST102/eval_plots1010"
    )
