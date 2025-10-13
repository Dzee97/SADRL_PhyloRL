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
        for start_idx in range(num_start_trees):
            print(f"Evaluating sample {sample_idx}, start tree {start_idx}")
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
