import os
import torch
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from environment import PhyloEnv
from dqn_agent import DQNAgent

CHECKPOINT_EVERY = 100
UPDATE_FREQUENCY = 4


def train_agent_process(agent_id, samples_dir, raxml_path, episodes, horizon, save_dir):
    torch.set_num_threads(1)

    # Create environment for this process
    env = PhyloEnv(Path(samples_dir), Path(raxml_path), horizon=horizon)
    feats = env.reset()
    feature_dim = feats.shape[1]

    # Initialize DQN agent
    agent = DQNAgent(feature_dim)

    rewards = []
    step_counter = 0

    for ep in range(episodes):
        feats = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action_idx, eps = agent.select_action(feats)
            feat_vec = feats[action_idx]
            next_feats, reward, done = env.step(action_idx)

            # Store transition with compressed next state
            agent.replay.push(feat_vec, reward, next_feats, done)

            # Update less frequently
            step_counter += 1
            if step_counter % UPDATE_FREQUENCY == 0:
                agent.update()

            total_reward += reward
            feats = next_feats

        rewards.append(total_reward)

        # Log less frequently
        if (ep + 1) % 10 == 0:
            recent_avg = np.mean(rewards[-10:])
            print(f"[Agent {agent_id}] Ep {ep+1}/{episodes} | "
                  f"Reward: {total_reward:.3f} | Avg10: {recent_avg:.3f} | "
                  f"Eps: {eps:.3f} | Cache hits: {env.cache_hits} | Cache size: {len(env.tree_cache)}")

        # Periodic saving
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = Path(save_dir) / f"agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)

    # Save final model
    agent.save(Path(save_dir) / f"agent_{agent_id}_final.pt")
    print(f"[Agent {agent_id}] Finished. Avg reward: {np.mean(rewards):.3f}")
    return rewards


def run_parallel_training(samples_dir, raxml_path, episodes=2000, horizon=20,
                          n_agents=5, n_cores=None, save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)
    n_cores = n_cores or min(os.cpu_count(), n_agents)

    print(f"🚀 Launching {n_agents} agents on {n_cores} cores")
    print(f"   Episodes: {episodes}, Horizon: {horizon}")
    print(f"   Update frequency: {UPDATE_FREQUENCY}")
    print(f"   Memory optimization: Single next-action storage\n")

    results = Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
        delayed(train_agent_process)(agent_id, samples_dir, raxml_path,
                                     episodes, horizon, save_dir)
        for agent_id in range(n_agents)
    )

    print("\n✅ All agents finished training.")

    # Summary statistics
    all_rewards = [r for agent_rewards in results for r in agent_rewards]
    print(f"\n📊 Summary:")
    print(f"   Mean reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"   Min: {np.min(all_rewards):.3f}, Max: {np.max(all_rewards):.3f}")

    return results


if __name__ == "__main__":
    run_parallel_training(
        samples_dir="OUTTEST1010",
        save_dir="OUTTEST1010/checkpoints",
        raxml_path="raxmlng/raxml-ng",
        episodes=2000,
        horizon=20,
        n_agents=5,
        n_cores=2
    )
