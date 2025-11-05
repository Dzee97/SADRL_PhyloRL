import os
import shutil
import torch
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from environment import PhyloEnv
from soft_dqn_agent import SoftDQNAgent


def train_agent_process(agent_id, samples_dir, raxmlng_path, episodes, horizon, checkpoint_dir,
                        checkpoint_freq, update_freq, hidden_dim, replay_size, replay_alpha, min_replay_start,
                        learning_rate, weight_decay, gamma, temp_alpha_init, entropy_frames, entropy_start, entropy_end,
                        replay_beta_start, replay_beta_frames, tau, batch_size):
    torch.set_num_threads(1)

    # Create environment for this process
    env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)
    tree_hash, feats = env.reset()
    feature_dim = feats.shape[1]
    num_actions = feats.shape[0]

    # Initialize Soft DQN agent
    agent = SoftDQNAgent(feature_dim=feature_dim,
                         hidden_dim=hidden_dim,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         gamma=gamma,
                         tau=tau,
                         temp_alpha_init=temp_alpha_init,
                         replay_size=replay_size,
                         replay_alpha=replay_alpha)

    step_counter = 0

    # Target entropy schedule
    H_max = np.log(num_actions)
    H_start = entropy_start * H_max
    H_end = entropy_end * H_max

    for ep in range(episodes):
        tree_hash, feats = env.reset()
        raxml_return = env.current_sample["rand_test_trees_ml_best"] - env.current_ll
        current_return = 0.0
        highest_return = 0.0
        trees_visited = {tree_hash}
        q_loss = np.nan
        policy_entropy = np.nan
        done = False

        while not done:
            # Sample actions equal to number of current visited trees
            action_idxs = agent.select_actions(feats, num_actions=len(trees_visited))
            for action_idx in action_idxs:
                # Preview neighbor trees from actions and select first action giving an unvisited tree
                preview_tree_hash = env.preview_step(action_idx, calc_reward=False)
                if preview_tree_hash not in trees_visited:
                    break
            feat_vec = feats[action_idx]
            next_tree_hash, next_feats, reward, done = env.step(action_idx)

            # Store the new tree hash
            trees_visited.add(next_tree_hash)

            # Store transition with compressed next state
            agent.replay.push(feat_vec, reward, next_feats, done)

            # Update less frequently
            step_counter += 1
            beta = min(1.0, replay_beta_start + step_counter * (1.0 - replay_beta_start) / replay_beta_frames)
            target_entropy = max(H_end, H_start + step_counter * (H_end - H_start) / entropy_frames)

            if step_counter % update_freq == 0 and len(agent.replay) >= min_replay_start:
                q_loss, policy_entropy = agent.update(batch_size, beta, target_entropy)

            current_return += reward
            highest_return = max(highest_return, current_return)
            feats = next_feats

        # Log less frequently
        if (ep + 1) % 10 == 0:
            print(f"[Agent {agent_id}] Ep {ep+1}/{episodes} | Highest Return: {highest_return:.3f} | "
                  f"RAxML-NG Diff: {highest_return - raxml_return:.3f} | Beta: {beta:.3f} | Alpha: {agent.alpha:.3f} | "
                  f"Policy Ent: {policy_entropy:.3f} | Target Ent: {target_entropy:.3f} | Q loss: {q_loss:.3f} | "
                  f"Trees visited: {len(trees_visited)} | Cache hits: {env.cache_hits} | "
                  f"Cache size: {len(env.tree_cache)}")

        # Periodic saving
        if (ep + 1) % checkpoint_freq == 0:
            ckpt_path = Path(checkpoint_dir) / f"agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)

    # Save final model
    agent.save(Path(checkpoint_dir) / f"agent_{agent_id}_final.pt")
    print(f"[Agent {agent_id}] Finished]")


def soft_run_parallel_training(samples_dir, raxmlng_path, episodes, horizon, n_agents, n_cores, checkpoint_dir,
                               checkpoint_freq, update_freq, hidden_dim, replay_size, replay_alpha, min_replay_start,
                               learning_rate, weight_decay, gamma, temp_alpha_init, entropy_frames,
                               entropy_start, entropy_end, replay_beta_start, replay_beta_frames, tau, batch_size):

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

    Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
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
                                     min_replay_start=min_replay_start,
                                     batch_size=batch_size,
                                     # soft dqn agent params
                                     hidden_dim=hidden_dim,
                                     gamma=gamma,
                                     learning_rate=learning_rate,
                                     weight_decay=weight_decay,
                                     replay_size=replay_size,
                                     replay_alpha=replay_alpha,
                                     replay_beta_start=replay_beta_start,
                                     replay_beta_frames=replay_beta_frames,
                                     temp_alpha_init=temp_alpha_init,
                                     entropy_start=entropy_start,
                                     entropy_end=entropy_end,
                                     entropy_frames=entropy_frames,
                                     tau=tau,
                                     )
        for agent_id in range(n_agents)
    )

    print("\n✅ All agents finished training.")
