import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

# ------------------------- NOISY DQN NETWORK -------------------------


class NoisyLinear(nn.Module):
    """Factorized Gaussian noise (Fortunato et al., 2018)"""

    def __init__(self, in_features, out_features, sigma_init):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register buffers for noise (non-trainable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialization as in the paper
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Factorized noise (same trick as in Fortunato et al.)"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Sample new noise for weights and bias"""
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return func.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, sigma_init):
        super().__init__()
        self.layers = nn.ModuleList([
            NoisyLinear(input_dim, hidden_dim, sigma_init),
            NoisyLinear(hidden_dim, hidden_dim, sigma_init),
            NoisyLinear(hidden_dim, hidden_dim, sigma_init),
            NoisyLinear(hidden_dim, 1, sigma_init)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layers[0](x))
        x = self.relu(self.layers[1](x))
        x = self.relu(self.layers[2](x))
        return self.layers[3](x).squeeze(-1)

    def reset_noise(self):
        for layer in self.layers:
            layer.reset_noise()

# ------------------------- PRIORITIZED REPLAY BUFFER -------------------------


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, device):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device
        self.pos = 0
        self.full = False

        self.feats = None
        self.rewards = None
        self.next_feats = None
        self.dones = None
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.max_prio = 1.0

    def push(self, feat, reward, next_feats, done):
        feat = torch.tensor(feat, dtype=torch.float32, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_feats = torch.tensor(next_feats, dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.float32, device=self.device)

        if self.feats is None:
            # allocate tensors
            self.feats = torch.zeros((self.capacity, *feat.shape), dtype=torch.float32, device=self.device)
            self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
            self.next_feats = torch.zeros((self.capacity, *next_feats.shape), dtype=torch.float32, device=self.device)
            self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)

        self.feats[self.pos] = feat
        self.rewards[self.pos] = reward
        self.next_feats[self.pos] = next_feats
        self.dones[self.pos] = done
        self.priorities[self.pos] = self.max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size, beta):
        n = self.capacity if self.full else self.pos
        prios = self.priorities[:n]
        probs = (prios ** self.alpha)
        probs /= probs.sum()

        indices = torch.multinomial(probs, batch_size, replacement=False)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            self.feats[indices],
            self.rewards[indices].squeeze(1),
            self.next_feats[indices],
            self.dones[indices].squeeze(1),
            indices,
            weights,
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities
        self.max_prio = max(self.max_prio, priorities.max().item())

    def __len__(self):
        return self.capacity if self.full else self.pos


# ------------------------- DQN AGENT -------------------------


class RainbowLiteDQNAgent:
    def __init__(self, feature_dim, hidden_dim, learning_rate, sigma_init, gamma, alpha,
                 tau, replay_size, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.step_count = 0

        self.q_net = NoisyQNetwork(feature_dim, hidden_dim, sigma_init).to(self.device)
        self.target_net = NoisyQNetwork(feature_dim, hidden_dim, sigma_init).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay = PrioritizedReplayBuffer(replay_size, alpha, self.device)

    def select_action(self, feats):
        with torch.no_grad():
            self.q_net.reset_noise()
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            q_vals = self.q_net(x)
            return int(torch.argmax(q_vals).item())

    def update(self, batch_size, beta):
        if len(self.replay) < batch_size:
            return None

        # Reset noise once for the whole batch
        self.q_net.reset_noise()

        feats, rewards, next_feats, dones, indices, weights = self.replay.sample(batch_size, beta)

        q_values = self.q_net(feats)
        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)
            next_q_online = self.q_net(next_feats_flat).view(B, N)
            best_next_actions = next_q_online.argmax(1)
            next_q_target = self.target_net(next_feats_flat).view(B, N)
            next_q_max = next_q_target[torch.arange(B), best_next_actions]
            q_targets = rewards + self.gamma * next_q_max * (1 - dones)

        td_errors = q_targets - q_values
        loss = (weights * func.smooth_l1_loss(q_values, q_targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        new_prios = td_errors.abs().detach() + 1e-6
        self.replay.update_priorities(indices, new_prios)

        # Soft target update
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.lerp_(param.data, self.tau)

        return loss.item()

    def get_avg_sigma(self):
        """Get average EFFECTIVE sigma (scaled by input dimension)"""
        sigmas = []
        for layer in self.q_net.layers:
            # Effective sigma includes the scaling factor
            effective_sigma = layer.weight_sigma.abs().mean().item() * math.sqrt(layer.in_features)
            sigmas.append(effective_sigma)
        return np.mean(sigmas)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), str(path))
