from pathlib import Path
import numpy as np
import torch
from environment import PhyloEnv
from dqn_agent import DQNAgent


def train_dqn(
    samples_dir="OUTTEST",
    raxml_path="raxmlng/raxml-ng",
    episodes=2000,
    horizon=20,
    batch_size=128
):
    env = PhyloEnv(
        samples_parent_dir=Path(samples_dir),
        raxmlng_path=Path(raxml_path),
        horizon=horizon
    )

    # Initialize feature dimension
    _, _, feats = env.reset()
    feature_dim = feats.shape[1]
    agent = DQNAgent(feature_dim)

    all_rewards = []

    for ep in range(episodes):
        annotated_tree, moves, feats = env.reset()
        print(f"\nStart LL: {env.current_ll}")
        print(f"Pars LL: {env.current_sample['norm_ll']}")
        start_pars_diff = (env.current_sample['norm_ll'] - env.current_ll)  # / abs(env.current_sample['norm_ll'])
        print(f"Diff: {start_pars_diff}")
        total_reward = 0.0

        for t in range(horizon):
            action_idx, eps = agent.select_action(feats)
            move = moves[action_idx]
            feat_vec = feats[action_idx]

            (next_tree, next_moves, next_feats), reward, done = env.step(annotated_tree, move)

            # Store transition
            agent.replay.push(feat_vec, reward, next_feats, done)
            loss = agent.update(batch_size=batch_size)

            total_reward += reward
            if total_reward >= start_pars_diff:
                print(f"##### IMPROVEMENT EXCEEDS PARS TREE: {total_reward} (step: {t+1})")
            annotated_tree, moves, feats = next_tree, next_moves, next_feats
            if done:
                break

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.4f} | Eps: {eps:.4f}")

    return all_rewards


if __name__ == "__main__":
    train_dqn()
