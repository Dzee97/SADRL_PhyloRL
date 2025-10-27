import random
import numpy as np
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

# ------------------------- DQN NETWORK -------------------------


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ------------------------- REPLAY BUFFER -------------------------


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, feat, reward, next_feats, done):
        # feat: (F,), next_feats: (N_moves, F)
        self.buffer.append((feat, reward, next_feats, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        feats, rewards, next_feats, dones = zip(*batch)
        return np.stack(feats), np.array(rewards), np.stack(next_feats), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# ------------------------- DQN AGENT -------------------------


class DQNAgent:
    def __init__(self, feature_dim, hidden_dim, learning_rate, gamma, tau, replay_size, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.step_count = 0

        self.q_net = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer(replay_size)

    def select_action(self, state_action_feats, temp, eval_mode=False):
        with torch.no_grad():
            feats_t = torch.tensor(state_action_feats, dtype=torch.float32, device=self.device)
            q_values = self.q1(feats_t).squeeze(-1)  # shape [num_actions]

            if eval_mode:
                action = torch.argmax(q_values).item()
            else:
                probs = torch.softmax(q_values / temp, dim=0)
                action = torch.multinomial(probs, 1).item()

            return action

    def update(self, batch_size):
        if len(self.replay) < batch_size:
            return None

        feats, rewards, next_feats, dones = self.replay.sample(batch_size)

        # Convert to tensors
        feats = torch.tensor(feats, dtype=torch.float32, device=self.device)  # (B, F)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (B,)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)  # (B,)
        next_feats = torch.tensor(next_feats, dtype=torch.float32, device=self.device)  # (B, N_moves, F)

        # --- Q(s,a) for current features ---
        q_values = self.q_net(feats)  # (B,)

        # --- Compute target values ---
        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)
            next_qs_flat = self.target_net(next_feats_flat).view(B, N)
            next_q_max = next_qs_flat.max(dim=1)[0]  # (B,)

            q_targets = rewards + self.gamma * next_q_max * (1 - dones)

        # --- Loss and optimization ---
        loss = func.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- soft update target network ---
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), str(path))
