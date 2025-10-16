import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

# ------------------------- NOISY DQN NETWORK -------------------------


class NoisyLinear(nn.Module):
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
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.max_prio = 1.0

    def push(self, feat, reward, next_feats, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((feat, reward, next_feats, done))
        else:
            self.buffer[self.pos] = (feat, reward, next_feats, done)
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        feats, rewards, next_feats, dones = zip(*samples)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            np.stack(feats),
            np.array(rewards),
            np.stack(next_feats),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
        self.max_prio = self.priorities.max()

    def __len__(self):
        return len(self.buffer)


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
        self.replay = PrioritizedReplayBuffer(replay_size, alpha)

    def select_action(self, feats):
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32, device=self.device)
            self.q_net.reset_noise()
            q_vals = self.q_net(x)
            return int(torch.argmax(q_vals).item())

    def update(self, batch_size, beta):
        if len(self.replay) < batch_size:
            return None

        feats, rewards, next_feats, dones, indices, weights = self.replay.sample(batch_size, beta)

        # Convert to tensors
        feats = torch.tensor(feats, dtype=torch.float32, device=self.device)  # (B, F)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (B,)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)  # (B,)
        next_feats = torch.tensor(next_feats, dtype=torch.float32, device=self.device)  # (B, N_moves, F)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # --- Q(s,a) for current features ---
        self.q_net.reset_noise()
        q_values = self.q_net(feats)  # (B,)

        # --- Compute target values ---
        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)
            # Double DQN: action selection by online net, evaluation by target net
            next_q_online = self.q_net(next_feats_flat).view(B, N)
            best_next_actions = next_q_online.argmax(1)
            next_q_target = self.target_net(next_feats_flat).view(B, N)
            next_q_max = next_q_target[torch.arange(B), best_next_actions]

            q_targets = rewards + self.gamma * next_q_max * (1 - dones)

        # --- Loss and optimization ---
        td_errors = q_targets - q_values
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Update priorities ---
        new_prios = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, new_prios)

        # --- soft update target network ---
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()

    def get_avg_sigma(self):
        sigmas = []
        for layer in self.q_net.layers:
            sigmas.append(layer.weight_sigma.abs().mean().item())
        return np.mean(sigmas)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), str(path))
