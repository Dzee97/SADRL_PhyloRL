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
    def __init__(self, input_dim, hidden_dim=256):
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
    def __init__(self, capacity=10_000):
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
    def __init__(
            self,
            feature_dim,
            lr=1e-5,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=10_000,
            target_update=1000,
            device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.step_count = 0

        self.q_net = QNetwork(feature_dim).to(self.device)
        self.target_net = QNetwork(feature_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()

    def select_action(self, feats):
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1.0 * self.step_count / self.epsilon_decay)
        self.step_count += 1

        if random.random() < eps:
            return random.randrange(len(feats)), eps
        else:
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32, device=self.device)
                q_vals = self.q_net(x)
                return int(torch.argmax(q_vals).item()), eps

    def update(self, batch_size=128):
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

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), str(path))
