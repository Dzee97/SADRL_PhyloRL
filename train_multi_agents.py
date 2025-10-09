import os
import torch
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from environment import PhyloEnv
from dqn_agent import DQNAgent

CHECKPOINT_EVERY = 100


def train_agent_process(agent_id, samples_dir, raxml_path, episodes, horizon, save_dir):
    torch.set_num_threads(1)
    # Create environment for this process
    env = PhyloEnv(Path(samples_dir), Path(raxml_path), horizon=horizon)
    _, _, feats = env.reset()
    feature_dim = feats.shape[1]

    # Initialize DQN agent
    agent = DQNAgent(feature_dim)

    rewards = []
    for ep in range(episodes):
        annotated_tree, moves, feats = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action_idx, eps = agent.select_action(feats)
            move = moves[action_idx]
            feat_vec = feats[action_idx]
            (next_tree, next_moves, next_feats), reward, done = env.step(annotated_tree, move)
            agent.replay.push(feat_vec, reward, next_feats, done)
            agent.update()

            total_reward += reward
            annotated_tree, moves, feats = next_tree, next_moves, next_feats

        rewards.append(total_reward)
        print(f"[Agent {agent_id}] Episode {ep+1}/{episodes} | Reward: {total_reward:.4f} | Eps: {eps:.4f}")

        # Periodic saving
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = Path(save_dir) / f"agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)
            print(f"[Agent {agent_id}] Saved checkpoint {ckpt_path}")

    # Save final model
    agent.save(Path(save_dir) / f"agent_{agent_id}_final.pt")
    print(f"[Agent {agent_id}] Finished training.")
    return rewards


def run_parallel_training(samples_dir, raxml_path, episodes=2000, horizon=20,
                          n_agents=5, n_cores=None, save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)
    n_cores = n_cores or min(os.cpu_count(), n_agents)

    print(f"🚀 Launching {n_agents} agents across {n_cores} cores using Joblib...")

    results = Parallel(n_jobs=n_cores, backend="loky", verbose=10)(
        delayed(train_agent_process)(agent_id, samples_dir, raxml_path, episodes, horizon, save_dir)
        for agent_id in range(n_agents)
    )

    print("✅ All agents finished training.")
    return results


if __name__ == "__main__":
    run_parallel_training(
        samples_dir="OUTTEST",
        raxml_path="raxmlng/raxml-ng",
        n_cores=3)
