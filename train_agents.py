import os
import shutil
import torch
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
from environment import PhyloEnv
from agents import DQNAgent, SoftQAgent, GNNSoftQAgent


def train_dqn_agent_process(agent_id, samples_dir, raxmlng_path, episodes, horizon, checkpoint_dir, checkpoint_freq,
                            update_freq, hidden_dim, dropout_p, replay_size, replay_alpha, min_replay_start,
                            learning_rate, weight_decay, gamma, temp, replay_beta_start, replay_beta_frames, tau,
                            batch_size, double_q):
    torch.set_num_threads(1)

    # Create environment for this process
    env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)
    tree_hash, feats = env.reset()
    feature_dim = feats.shape[1]

    # Initialize DQN agent
    agent = DQNAgent(feature_dim=feature_dim,
                     hidden_dim=hidden_dim,
                     dropout_p=dropout_p,
                     learning_rate=learning_rate,
                     weight_decay=weight_decay,
                     gamma=gamma,
                     tau=tau,
                     replay_size=replay_size,
                     replay_alpha=replay_alpha,
                     double_q=double_q)

    step_counter = 0

    for ep in range(episodes):
        tree_hash, feats = env.reset()
        raxml_return = env.current_sample["rand_test_trees_ml_best"] - env.current_ll
        current_return = 0.0
        highest_return = 0.0
        trees_visited = {tree_hash}
        q_loss = np.nan
        done = False

        while not done:
            # Sample actions equal to number of current visited trees
            action_idxs = agent.select_actions(feats, temp, num_actions=len(trees_visited))
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

            if step_counter % update_freq == 0 and len(agent.replay) >= min_replay_start:
                q_loss = agent.update(batch_size, beta)

            current_return += reward
            highest_return = max(highest_return, current_return)
            feats = next_feats

        # Log less frequently
        if (ep + 1) % 10 == 0:
            print(f"[Agent {agent_id}] Ep {ep+1}/{episodes} | Highest Return: {highest_return:.3f} | "
                  f"RAxML-NG Diff: {highest_return - raxml_return:.3f} | ER Beta: {beta:.3f} | Q loss: {q_loss:.3f} | "
                  f"Cache hits: {env.cache_hits} | Cache size: {len(env.tree_cache)}")

        # Periodic saving
        if (ep + 1) % checkpoint_freq == 0:
            ckpt_path = Path(checkpoint_dir) / f"agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)

    # Save final model
    agent.save(Path(checkpoint_dir) / f"agent_{agent_id}_final.pt")
    print(f"[Agent {agent_id}] Finished")


def train_softq_agent_process(agent_id, samples_dir, raxmlng_path, episodes, horizon, checkpoint_dir, checkpoint_freq,
                              update_freq, hidden_dim, dropout_p, replay_size, replay_alpha, min_replay_start,
                              learning_rate, weight_decay, gamma, temp_alpha_init, entropy_frames, entropy_start,
                              entropy_end, replay_beta_start, replay_beta_frames, tau, batch_size):
    torch.set_num_threads(1)

    # Create environment for this process
    env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)
    tree_hash, feats = env.reset()
    feature_dim = feats.shape[1]
    num_actions = feats.shape[0]

    # Initialize Soft DQN agent
    agent = SoftQAgent(feature_dim=feature_dim,
                       hidden_dim=hidden_dim,
                       dropout_p=dropout_p,
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
                  f"RAxML-NG Diff: {highest_return - raxml_return:.3f} | ER Beta: {beta:.3f} | "
                  f"Alpha: {agent.alpha:.3f} | Policy Ent: {policy_entropy:.3f} | Target Ent: {target_entropy:.3f} | "
                  f"Q loss: {q_loss:.3f} | Cache hits: {env.cache_hits} | Cache size: {len(env.tree_cache)}")

        # Periodic saving
        if (ep + 1) % checkpoint_freq == 0:
            ckpt_path = Path(checkpoint_dir) / f"agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)

    # Save final model
    agent.save(Path(checkpoint_dir) / f"agent_{agent_id}_final.pt")
    print(f"[Agent {agent_id}] Finished")


def train_gnn_agent_process(agent_id, samples_dir, raxmlng_path, episodes, horizon, checkpoint_dir, checkpoint_freq,
                            update_freq, hidden_dim, dropout_p, replay_size, replay_alpha, min_replay_start,
                            learning_rate, weight_decay, gamma, temp_alpha_init, entropy_frames, entropy_start,
                            entropy_end, replay_beta_start, replay_beta_frames, tau, batch_size,
                            num_gat_layers, num_attention_heads):
    """Training function for GNN-based Soft Q agent."""
    import time
    torch.set_num_threads(1)

    # Timing metrics
    timing_stats = {
        'env_reset': 0.0,
        'action_selection': 0.0,
        'env_step': 0.0,
        'replay_push': 0.0,
        'agent_update': 0.0,
        'checkpoint_save': 0.0
    }
    timing_counts = {key: 0 for key in timing_stats}

    # Create environment in GNN mode
    t0 = time.time()
    env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon, use_gnn=True)
    tree_hash, (graph_data, action_embeddings) = env.reset()
    timing_stats['env_reset'] += time.time() - t0
    timing_counts['env_reset'] += 1
    
    print(f"[GNN Agent {agent_id}] Initialized environment (took {time.time() - t0:.2f}s)")
    
        # GNN uses atomic features: node_feat_dim=1, edge_feat_dim=1, action_dim=7
    t0 = time.time()
    agent = GNNSoftQAgent(
        node_feat_dim=1,  # [is_leaf]
        edge_feat_dim=1,  # [branch_length]
        action_dim=7,     # [4 node indices + 3 metadata]
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_attention_heads=num_attention_heads,
        dropout_p=dropout_p,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gamma=gamma,
        tau=tau,
        temp_alpha_init=temp_alpha_init,
        replay_size=replay_size,
        replay_alpha=replay_alpha
    )
    print(f"[GNN Agent {agent_id}] Initialized agent (took {time.time() - t0:.2f}s)")

    step_counter = 0
    num_actions = len(action_embeddings)

    # Target entropy schedule
    H_max = np.log(num_actions)
    H_start = entropy_start * H_max
    H_end = entropy_end * H_max

    print(f"[GNN Agent {agent_id}] Starting training: {episodes} episodes, {horizon} horizon")

    for ep in range(episodes):
        ep_start = time.time()
        
        t0 = time.time()
        tree_hash, (graph_data, action_embeddings) = env.reset()
        timing_stats['env_reset'] += time.time() - t0
        timing_counts['env_reset'] += 1
        
        raxml_return = env.current_sample["rand_test_trees_ml_best"] - env.current_ll
        current_return = 0.0
        highest_return = 0.0
        trees_visited = {tree_hash}
        q_loss = np.nan
        policy_entropy = np.nan
        done = False

        while not done:
            # GNN agent API: pass graph_data and action_embeddings
            t0 = time.time()
            action_idxs = agent.select_actions(graph_data, action_embeddings, num_actions=len(trees_visited))
            timing_stats['action_selection'] += time.time() - t0
            timing_counts['action_selection'] += 1
            
            for action_idx in action_idxs:
                # Preview neighbor trees from actions and select first action giving an unvisited tree
                preview_tree_hash = env.preview_step(action_idx, calc_reward=False)
                if preview_tree_hash not in trees_visited:
                    break
            
            # Get the selected action embedding
            # action_embeddings is now a tensor [num_actions, 7]
            action_embedding = action_embeddings[action_idx]
            
            t0 = time.time()
            next_tree_hash, (next_graph_data, next_action_embeddings), reward, done = env.step(action_idx)
            timing_stats['env_step'] += time.time() - t0
            timing_counts['env_step'] += 1

            # Store the new tree hash
            trees_visited.add(next_tree_hash)

            # GNN replay buffer API: push with graph data structures
            t0 = time.time()
            # Optimization: push() now converts next_action_embeddings to tensor
            # We can reuse this tensor in the next iteration if we returned it, but for now
            # we rely on the fact that push() is much faster than the old update loop.
            agent.replay.push(graph_data, action_embedding, action_idx, reward, 
                            next_graph_data, next_action_embeddings, done)
            timing_stats['replay_push'] += time.time() - t0
            timing_counts['replay_push'] += 1

            # Update less frequently
            step_counter += 1
            beta = min(1.0, replay_beta_start + step_counter * (1.0 - replay_beta_start) / replay_beta_frames)
            target_entropy = max(H_end, H_start + step_counter * (H_end - H_start) / entropy_frames)

            if step_counter % update_freq == 0 and len(agent.replay) >= min_replay_start:
                t0 = time.time()
                q_loss, policy_entropy = agent.update(batch_size, beta, target_entropy)
                timing_stats['agent_update'] += time.time() - t0
                timing_counts['agent_update'] += 1

            current_return += reward
            highest_return = max(highest_return, current_return)
            graph_data, action_embeddings = next_graph_data, next_action_embeddings

        ep_time = time.time() - ep_start

        # Log with timing info
        if (ep + 1) % 10 == 0:
            # Compute average times
            avg_times = {k: v / max(timing_counts[k], 1) for k, v in timing_stats.items()}
            
            print(f"[GNN Agent {agent_id}] Ep {ep+1}/{episodes} | Time: {ep_time:.2f}s | "
                  f"Return: {highest_return:.3f} | RAxML Diff: {highest_return - raxml_return:.3f}")
            print(f"  Performance: Beta={beta:.3f} | Alpha={agent.alpha:.3f} | "
                  f"Pol_Ent={policy_entropy:.3f} | Tgt_Ent={target_entropy:.3f} | Q_loss={q_loss:.3f}")
            print(f"  Timing (avg ms): Reset={avg_times['env_reset']*1000:.1f} | "
                  f"Action={avg_times['action_selection']*1000:.1f} | "
                  f"Step={avg_times['env_step']*1000:.1f} | "
                  f"Push={avg_times['replay_push']*1000:.1f} | "
                  f"Update={avg_times['agent_update']*1000:.1f}")
            print(f"  Cache: hits={env.cache_hits} | size={len(env.tree_cache)}")

        # Periodic saving
        if (ep + 1) % checkpoint_freq == 0:
            t0 = time.time()
            ckpt_path = Path(checkpoint_dir) / f"gnn_agent_{agent_id}_ep{ep+1}.pt"
            agent.save(ckpt_path)
            save_time = time.time() - t0
            timing_stats['checkpoint_save'] += save_time
            timing_counts['checkpoint_save'] += 1
            print(f"[GNN Agent {agent_id}] Saved checkpoint (took {save_time:.2f}s)")

    # Save final model
    t0 = time.time()
    agent.save(Path(checkpoint_dir) / f"gnn_agent_{agent_id}_final.pt")
    print(f"[GNN Agent {agent_id}] Saved final model (took {time.time() - t0:.2f}s)")
    
    # Print final timing summary
    print(f"\n[GNN Agent {agent_id}] Training Complete - Timing Summary:")
    print(f"  Total calls: Reset={timing_counts['env_reset']} | "
          f"Action={timing_counts['action_selection']} | Step={timing_counts['env_step']} | "
          f"Push={timing_counts['replay_push']} | Update={timing_counts['agent_update']}")
    print(f"  Total time (s): Reset={timing_stats['env_reset']:.1f} | "
          f"Action={timing_stats['action_selection']:.1f} | Step={timing_stats['env_step']:.1f} | "
          f"Push={timing_stats['replay_push']:.1f} | Update={timing_stats['agent_update']:.1f}")
    avg_times = {k: v / max(timing_counts[k], 1) for k, v in timing_stats.items()}
    print(f"  Avg time (ms): Reset={avg_times['env_reset']*1000:.1f} | "
          f"Action={avg_times['action_selection']*1000:.1f} | "
          f"Step={avg_times['env_step']*1000:.1f} | "
          f"Push={avg_times['replay_push']*1000:.1f} | "
          f"Update={avg_times['agent_update']*1000:.1f}")
    print(f"[GNN Agent {agent_id}] Finished")


def run_parallel_training(algorithm, samples_dir, raxmlng_path, n_agents, n_cores, checkpoint_dir, training_hps):
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

    if algorithm == "DQN":
        train_agent_process = train_dqn_agent_process
    elif algorithm == "SQL":
        train_agent_process = train_softq_agent_process
    elif algorithm == "GNN":
        train_agent_process = train_gnn_agent_process
    else:
        raise ValueError("Invalid algorithm name")

    print(f"🚀 Launching {n_agents} agents on {n_cores} cores")
    print(f"   Algorithm: {algorithm}")
    print("   Hyperparameters:\n")
    print("".join((f"   '{k}': {v}\n" for k, v in training_hps.items())))

    Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
        delayed(train_agent_process)(agent_id=agent_id,
                                     samples_dir=samples_dir,
                                     checkpoint_dir=checkpoint_dir,
                                     raxmlng_path=raxmlng_path,
                                     **training_hps)
        for agent_id in range(n_agents)
    )

    print("\n✅ All agents finished training.")
