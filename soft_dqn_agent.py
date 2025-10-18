import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


# ------------------------- SIMPLE Q NETWORK -------------------------


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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

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

# ------------------------- SOFT DQN AGENT (CLIPPED DOUBLE Q) -------------------------


class SoftDQNAgent:
    def __init__(self, feature_dim, hidden_dim, learning_rate, gamma, tau,
                 alpha, replay_size, replay_alpha, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Clipped double Q networks
        self.q1 = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_q1 = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_q2 = QNetwork(feature_dim, hidden_dim).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()

        self.optimizer1 = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.optimizer2 = optim.Adam(self.q2.parameters(), lr=learning_rate)

        self.replay = PrioritizedReplayBuffer(replay_size, replay_alpha, self.device)

    def select_action(self, state_action_feats, eval_mode=False):
        """
        Select an action given all possible (state, action) feature vectors.
        Each row of `state_action_feats` corresponds to one action for the same state.
        During training: sample from Boltzmann (softmax) policy.
        During evaluation: choose greedy (argmax) action.
        """
        with torch.no_grad():
            feats_t = torch.tensor(state_action_feats, dtype=torch.float32, device=self.device)
            q_values = self.q1(feats_t).squeeze(-1)  # shape [num_actions]

            if eval_mode:
                action = torch.argmax(q_values).item()
            else:
                probs = torch.softmax(q_values / self.alpha, dim=0)
                action = torch.multinomial(probs, 1).item()

            return action

    def update(self, batch_size, beta):
        if len(self.replay) < batch_size:
            return None

        feats, rewards, next_feats, dones, indices, weights = self.replay.sample(batch_size, beta)

        q1_vals = self.q1(feats)
        q2_vals = self.q2(feats)

        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)
            q1_next = self.target_q1(next_feats_flat).view(B, N)
            q2_next = self.target_q2(next_feats_flat).view(B, N)

            q_min = torch.min(q1_next, q2_next)

            soft_value = self.alpha * torch.logsumexp(q_min / self.alpha, dim=1)
            q_target = rewards + self.gamma * (1 - dones) * soft_value

        td_error1 = q_target - q1_vals
        td_error2 = q_target - q2_vals

        loss1 = (weights * td_error1.pow(2)).mean()
        loss2 = (weights * td_error2.pow(2)).mean()

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        new_prios = 0.5 * (td_error1.abs() + td_error2.abs()).detach() + 1e-6
        self.replay.update_priorities(indices, new_prios)

        for t_p, p in zip(self.target_q1.parameters(), self.q1.parameters()):
            t_p.data.lerp_(p.data, self.tau)
        for t_p, p in zip(self.target_q2.parameters(), self.q2.parameters()):
            t_p.data.lerp_(p.data, self.tau)

        return (loss1.item() + loss2.item()) / 2

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q1.state_dict(), str(path))
