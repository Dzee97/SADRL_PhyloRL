import random
import numpy as np
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from hyperparameters import (
    HIDDEN_DIM, REPLAY_BUFFER_CAPACITY, LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE, BATCH_SIZE,
    DOUBLE_Q, PRIORITIZED_REPLAY, PRIORITY_ALPHA, PRIORITY_BETA_START,
    PRIORITY_BETA_END, PRIORITY_BETA_ANNEAL_STEPS, DUELING, MULTI_STEP,
    N_STEPS, DISTRIBUTIONAL, V_MIN, V_MAX, NUM_ATOMS, NOISY_NETS, NOISY_SIGMA_INIT,
    INTRINSIC_CURIOSITY, INTRINSIC_REWARD_SCALE, CURIOSITY_LR
)

# ------------------------- NOISY LINEAR LAYER -------------------------

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA_INIT, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.bias = bias

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)

        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        if bias:
            self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        if self.bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        if self.bias:
            self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            return func.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                             self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias else None)
        else:
            return func.linear(input, self.weight_mu, self.bias_mu if self.bias else None)

# ------------------------- INTRINSIC CURIOSITY MODULE -------------------------

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=HIDDEN_DIM, eta=INTRINSIC_REWARD_SCALE):
        super().__init__()
        self.eta = eta
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Inverse model: predicts action from state and next state features
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Forward model: predicts next state features from state and action
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state_feat, action, next_state_feat):
        # Encode features
        phi = self.feature_encoder(state_feat)
        phi_next = self.feature_encoder(next_state_feat)

        # Inverse model loss
        inverse_input = torch.cat([phi, phi_next], dim=-1)
        pred_action = self.inverse_model(inverse_input)

        # Forward model loss
        action_onehot = torch.zeros(action.size(0), self.inverse_model[-1].out_features, device=action.device)
        action_onehot.scatter_(1, action.unsqueeze(-1), 1)
        forward_input = torch.cat([phi, action_onehot], dim=-1)
        pred_phi_next = self.forward_model(forward_input)

        # Intrinsic reward
        intrinsic_reward = self.eta * func.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=-1)

        return pred_action, pred_phi_next, phi_next, intrinsic_reward

# ------------------------- DQN NETWORK -------------------------


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, dueling=DUELING, distributional=DISTRIBUTIONAL,
                 noisy=NOISY_NETS, num_atoms=NUM_ATOMS, v_min=V_MIN, v_max=V_MAX):
        super().__init__()
        self.dueling = dueling
        self.distributional = distributional
        self.noisy = noisy
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1) if distributional else None
        self.support = torch.linspace(v_min, v_max, num_atoms) if distributional else None

        Linear = NoisyLinear if noisy else nn.Linear

        self.feature_layer = nn.Sequential(
            Linear(input_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if dueling:
            self.value_stream = Linear(hidden_dim, hidden_dim // 2)
            self.value_head = Linear(hidden_dim // 2, num_atoms if distributional else 1)
            self.advantage_stream = Linear(hidden_dim, hidden_dim // 2)
            self.advantage_head = Linear(hidden_dim // 2, num_atoms if distributional else 1)
        else:
            self.output_layer = Linear(hidden_dim, num_atoms if distributional else 1)

    def forward(self, x):
        features = self.feature_layer(x)

        if self.dueling:
            value = func.relu(self.value_stream(features))
            value = self.value_head(value)  # (batch, 1 or num_atoms)

            advantage = func.relu(self.advantage_stream(features))
            advantage = self.advantage_head(advantage)  # (batch, 1 or num_atoms)

            if self.distributional:
                q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            else:
                q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.output_layer(features)

        if self.distributional:
            # Apply softmax to get probabilities
            q_values = func.softmax(q_values, dim=-1)
        else:
            q_values = q_values.squeeze(-1)

        return q_values

    def reset_noise(self):
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

# ------------------------- REPLAY BUFFER -------------------------


# ------------------------- REPLAY BUFFER -------------------------

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY, prioritized=PRIORITIZED_REPLAY, alpha=PRIORITY_ALPHA):
        self.buffer = deque(maxlen=capacity)
        self.prioritized = prioritized
        self.alpha = alpha
        if prioritized:
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.max_priority = 1.0
        self.pos = 0

    def push(self, feat, action, reward, next_feats, done):
        # feat: (F,), action: int, next_feats: (N_moves, F)
        experience = (feat, action, reward, next_feats, done)
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        if self.prioritized:
            self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen

    def sample(self, batch_size, beta=PRIORITY_BETA_START):
        if self.prioritized:
            priorities = self.priorities[:len(self.buffer)] ** self.alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = (len(self.buffer) * probs[indices]) ** (-beta)
            weights /= weights.max()
            batch = [self.buffer[i] for i in indices]
            return np.stack([b[0] for b in batch]), np.array([b[1] for b in batch]), np.array([b[2] for b in batch]), np.stack([b[3] for b in batch]), np.array([b[4] for b in batch]), indices, weights
        else:
            batch = random.sample(self.buffer, batch_size)
            feats, actions, rewards, next_feats, dones = zip(*batch)
            return np.stack(feats), np.array(actions), np.array(rewards), np.stack(next_feats), np.array(dones), None, None

    def update_priorities(self, indices, priorities):
        if self.prioritized:
            for idx, prio in zip(indices, priorities):
                self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, priorities.max())

    def __len__(self):
        return len(self.buffer)

# ------------------------- DQN AGENT -------------------------


class DQNAgent:
    def __init__(
            self,
            feature_dim,
            lr=LEARNING_RATE,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            target_update=TARGET_UPDATE,
            device=None,
            double_q=DOUBLE_Q,
            prioritized_replay=PRIORITIZED_REPLAY,
            dueling=DUELING,
            multi_step=MULTI_STEP,
            n_steps=N_STEPS,
            distributional=DISTRIBUTIONAL,
            v_min=V_MIN,
            v_max=V_MAX,
            num_atoms=NUM_ATOMS,
            noisy_nets=NOISY_NETS,
            priority_alpha=PRIORITY_ALPHA,
            priority_beta_start=PRIORITY_BETA_START,
            priority_beta_end=PRIORITY_BETA_END,
            priority_beta_anneal_steps=PRIORITY_BETA_ANNEAL_STEPS,
            intrinsic_curiosity=INTRINSIC_CURIOSITY,
            intrinsic_reward_scale=INTRINSIC_REWARD_SCALE,
            curiosity_lr=CURIOSITY_LR
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.step_count = 0

        self.double_q = double_q
        self.prioritized_replay = prioritized_replay
        self.dueling = dueling
        self.multi_step = multi_step
        self.n_steps = n_steps
        self.distributional = distributional
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.noisy_nets = noisy_nets
        self.priority_alpha = priority_alpha
        self.priority_beta_start = priority_beta_start
        self.priority_beta_end = priority_beta_end
        self.priority_beta_anneal_steps = priority_beta_anneal_steps

        self.intrinsic_curiosity = intrinsic_curiosity
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.curiosity_lr = curiosity_lr

        self.q_net = QNetwork(feature_dim, dueling=dueling, distributional=distributional, noisy=noisy_nets,
                              num_atoms=num_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.target_net = QNetwork(feature_dim, dueling=dueling, distributional=distributional, noisy=noisy_nets,
                                   num_atoms=num_atoms, v_min=v_min, v_max=v_max).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(prioritized=prioritized_replay, alpha=priority_alpha)

        # Intrinsic Curiosity Module
        self.icm = None
        self.icm_optimizer = None
        if intrinsic_curiosity:
            # Action dimension: assume max 1000 possible actions
            max_actions = 1000
            self.icm = IntrinsicCuriosityModule(feature_dim, max_actions, eta=intrinsic_reward_scale).to(self.device)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=curiosity_lr)

    def select_action(self, feats):
        if self.noisy_nets:
            eps = 0.0  # No epsilon-greedy when using noisy nets
        else:
            eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                np.exp(-1.0 * self.step_count / self.epsilon_decay)
        self.step_count += 1

        if self.noisy_nets or random.random() > eps:
            with torch.no_grad():
                x = torch.tensor(feats, dtype=torch.float32, device=self.device)
                if self.noisy_nets:
                    self.q_net.reset_noise()
                q_vals = self.q_net(x)
                if self.distributional:
                    # For distributional, compute expected Q-values
                    support = self.q_net.support.to(self.device)
                    q_vals = (q_vals * support).sum(dim=-1)
                return int(torch.argmax(q_vals).item()), eps
        else:
            return random.randrange(len(feats)), eps

    def update(self, batch_size=BATCH_SIZE):
        if len(self.replay) < batch_size:
            return None

        # Compute current beta for prioritized replay
        beta = self.priority_beta_start + (self.priority_beta_end - self.priority_beta_start) * \
               min(1.0, self.step_count / self.priority_beta_anneal_steps)

        feats, actions, rewards, next_feats, dones, indices, weights = self.replay.sample(batch_size, beta)

        # Convert to tensors
        feats = torch.tensor(feats, dtype=torch.float32, device=self.device)  # (B, F)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)  # (B,)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (B,)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)  # (B,)
        next_feats = torch.tensor(next_feats, dtype=torch.float32, device=self.device)  # (B, N_moves, F)

        # Compute intrinsic rewards if enabled
        if self.intrinsic_curiosity and self.icm is not None:
            # For intrinsic reward, we need next_state_feat, which is the selected next feat
            # Since next_feats is (B, N, F), and action is the index, get next_state_feat
            next_state_feat = next_feats[torch.arange(batch_size), actions]  # (B, F)
            _, _, _, intrinsic_rewards = self.icm(feats, actions, next_state_feat)
        else:
            intrinsic_rewards = torch.zeros_like(rewards)

        total_rewards = rewards + intrinsic_rewards

        if self.noisy_nets:
            self.q_net.reset_noise()

        # Current Q-values or distributions
        current_q = self.q_net(feats)  # (B, 1 or num_atoms)

        # Compute targets
        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)

            if self.double_q:
                # Double Q: use online net to select action
                next_q_online = self.q_net(next_feats_flat)
                if self.distributional:
                    support = self.q_net.support.to(self.device)
                    next_q_online = (next_q_online * support).sum(dim=-1)
                next_actions = next_q_online.view(B, N).argmax(dim=1)  # (B,)
                next_actions_flat = next_actions + torch.arange(B, device=self.device) * N
                next_q_target = self.target_net(next_feats_flat)[next_actions_flat]  # (B, 1 or num_atoms)
                if self.distributional:
                    # For distributional, compute expected value for target computation
                    next_q_target = (next_q_target * self.target_net.support.to(self.device)).sum(dim=-1)
            else:
                next_q_flat = self.target_net(next_feats_flat)
                if self.distributional:
                    support = self.target_net.support.to(self.device)
                    next_q_flat = (next_q_flat * support).sum(dim=-1)
                next_q_target = next_q_flat.view(B, N).max(dim=1)[0]  # (B,)

            if self.multi_step:
                # Simple multi-step: approximate with gamma^n
                gamma_n = self.gamma ** self.n_steps
                targets = total_rewards + gamma_n * next_q_target * (1 - dones)
            else:
                targets = total_rewards + self.gamma * next_q_target * (1 - dones)

            if self.distributional:
                # Project targets to distribution
                targets = self._project_distribution(targets.unsqueeze(-1))  # (B, num_atoms)

        # Compute loss
        if self.distributional:
            loss = -(targets * torch.log(current_q + 1e-8)).sum(dim=-1).mean()
        else:
            loss = func.mse_loss(current_q, targets)

        if self.prioritized_replay and weights is not None:
            loss = (loss * torch.tensor(weights, device=self.device)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized_replay and indices is not None:
            # Update priorities with TD errors
            with torch.no_grad():
                if self.distributional:
                    # For distributional, compute TD error as difference in expected Q-values
                    support = self.q_net.support.to(self.device)
                    expected_current = (current_q * support).sum(dim=-1)
                    expected_targets = (targets * support).sum(dim=-1)
                    td_errors = (expected_current - expected_targets).abs().cpu().numpy()
                else:
                    td_errors = (current_q - targets).abs().cpu().numpy()
                self.replay.update_priorities(indices, td_errors + 1e-6)

        # Train Intrinsic Curiosity Module
        if self.intrinsic_curiosity and self.icm is not None:
            pred_action, pred_phi_next, phi_next, _ = self.icm(feats, actions, next_state_feat)
            inverse_loss = func.cross_entropy(pred_action, actions)
            forward_loss = func.mse_loss(pred_phi_next, phi_next)
            icm_loss = inverse_loss + forward_loss
            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            self.icm_optimizer.step()

        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def _project_distribution(self, targets):
        # Project scalar targets to categorical distribution
        batch_size = targets.size(0)
        support = self.q_net.support.to(self.device)
        delta_z = self.q_net.delta_z
        v_min = self.v_min
        v_max = self.v_max
        num_atoms = self.num_atoms

        targets = targets.clamp(v_min, v_max)
        b = (targets - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l = torch.clamp(l, 0, num_atoms - 1)
        u = torch.clamp(u, 0, num_atoms - 1)

        m_l = torch.zeros(batch_size, num_atoms, device=self.device)
        m_u = torch.zeros(batch_size, num_atoms, device=self.device)

        offset = torch.arange(batch_size, device=self.device).unsqueeze(1)
        m_l.scatter_add_(1, l, (u.float() - b).squeeze(-1).unsqueeze(-1))
        m_u.scatter_add_(1, u, (b - l.float()).squeeze(-1).unsqueeze(-1))

        return m_l + m_u

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), str(path))

    def load(self, path: Path):
        self.q_net.load_state_dict(torch.load(str(path), map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
