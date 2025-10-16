import os
import shutil
import torch
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from environment import PhyloEnv
from rainbow_dqn_agent import RainbowLiteDQNAgent


def train_agent_process(agent_id, samples_dir, raxmlng_path, episodes, horizon, checkpoint_dir,
                        checkpoint_freq, update_freq, hidden_dim, replay_size, learning_rate,
                        gamma, sigma_init, alpha, beta_start, tau, batch_size):
    torch.set_num_threads(1)

    # Create environment for this process
    env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)
    feats = env.reset()
    feature_dim = feats.shape[1]

    # Initialize Rainbow DQN agent
    agent = RainbowLiteDQNAgent(feature_dim=feature_dim,
                                hidden_dim=hidden_dim,
                                learning_rate=learning_rate,
                                gamma=gamma,
                                sigma_init=sigma_init,
                                alpha=alpha,
                                tau=tau,
                                replay_size=replay_size)

    rewards = []
    step_counter = 0
    beta_frames = episodes * horizon

    for ep in range(episodes):
        feats = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action_idx = agent.select_action(feats)
            feat_vec = feats[action_idx]
            next_feats, reward, done = env.step(action_idx)

            # Store transition with compressed next state
            agent.replay.push(feat_vec, reward, next_feats, done)

            # Update less frequently
            step_counter += 1
            beta = min(1.0, beta_start + step_counter * (1.0 - beta_start) / beta_frames)
            if step_counter % update_freq == 0:
                agent.update(batch_size, beta)

            total_reward += reward
            feats = next_feats

        rewards.append(total_reward)

        # Log less frequently
        if (ep + 1) % 10 == 0:
            recent_avg = np.mean(rewards[-10:])
            print(f"[Agent {agent_id}] Ep {ep+1}/{episodes} | "
                  f"Reward: {total_reward:.3f} | Avg10: {recent_avg:.3f} | "
                  f"Beta: {beta:.3f} | Avg Sigma: {agent.get_avg_sigma():.3f} | Cache hits: {env.cache_hits} | "
                  f"Cache size: {len(env.tree_cache)}")

        # Periodic saving
        if (ep + 1) % checkpoint_freq == 0:
            ckpt_path = Path(checkpoint_dir) / f"agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)

    # Save final model
    agent.save(Path(checkpoint_dir) / f"agent_{agent_id}_final.pt")
    print(f"[Agent {agent_id}] Finished. Avg reward: {np.mean(rewards):.3f}")
    return rewards


def rainbow_run_parallel_training(samples_dir, raxmlng_path, episodes, horizon, n_agents, n_cores, checkpoint_dir,
                                  checkpoint_freq, update_freq, hidden_dim, replay_size, learning_rate, gamma,
                                  sigma_init, alpha, beta_start, tau, batch_size):

    # ---- Check for existing checkpoint directory ----
    if checkpoint_dir.exists():
        answer = input(f"Checkpoint directory '{checkpoint_dir}' already exists. Overwrite? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborting — existing checkpoint directory preserved.")
            return
        print(f"Removing existing directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)
    n_cores = n_cores or min(os.cpu_count(), n_agents)

    print(f"🚀 Launching {n_agents} agents on {n_cores} cores")
    print(f"   Episodes: {episodes}, Horizon: {horizon}")

    results = Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
        delayed(train_agent_process)(agent_id=agent_id,
                                     samples_dir=samples_dir,
                                     checkpoint_dir=checkpoint_dir,
                                     # dep paths
                                     raxmlng_path=raxmlng_path,
                                     # training loop params
                                     episodes=episodes,
                                     horizon=horizon,
                                     checkpoint_freq=checkpoint_freq,
                                     update_freq=update_freq,
                                     batch_size=batch_size,
                                     # rainbow dqn agent params
                                     hidden_dim=hidden_dim,
                                     replay_size=replay_size,
                                     learning_rate=learning_rate,
                                     sigma_init=sigma_init,
                                     gamma=gamma,
                                     alpha=alpha,
                                     beta_start=beta_start,
                                     tau=tau)
        for agent_id in range(n_agents)
    )

    print("\n✅ All agents finished training.")

    # Summary statistics
    all_rewards = [r for agent_rewards in results for r in agent_rewards]
    print("\n📊 Summary:")
    print(f"   Mean reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"   Min: {np.min(all_rewards):.3f}, Max: {np.max(all_rewards):.3f}")

    return results
