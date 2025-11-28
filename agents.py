import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. GNN agents will not work.")
    print("Install with: pip install torch-geometric")


# ------------------------- GNN DATA STRUCTURES -------------------------


@dataclass
class GraphData:
    """Stores tree graph structure with stable node naming.
    
    CRITICAL: Phylogenetic trees are fundamentally UNROOTED, but GATv2Conv processes
    DIRECTED edges. To handle this correctly:
    - edge_index MUST contain BIDIRECTIONAL edges for every tree edge
    - If tree has edge (A, B), edge_index must have both (A→B) and (B→A)
    - edge_features must be duplicated accordingly (same features for both directions)
    - This ensures GNN can propagate information in both directions along each branch
    
    The TreePreprocessor arbitrarily roots the tree during DFS and stores directed edges
    (child→parent). The conversion function MUST create bidirectional edges from these.
    
    Feature Design (atomic features only):
    - node_features: [num_nodes, 1] containing [is_leaf] (1.0=leaf, 0.0=internal)
    - edge_features: [num_edges, 1] containing [branch_length]
    - GNN learns all tree properties from topology + these two atomic features
    """
    node_features: torch.Tensor      # [num_nodes, node_feat_dim] - typically node_feat_dim=1
    edge_index: torch.Tensor         # [2, num_edges] - MUST be bidirectional for unrooted trees!
    edge_features: torch.Tensor      # [num_edges, edge_feat_dim] - typically edge_feat_dim=1
    node_name_to_idx: Dict[str, int] # Maps node names to indices (for stable reference)
    
    def to(self, device):
        """Move tensors to device."""
        return GraphData(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            node_name_to_idx=self.node_name_to_idx  # Dict stays on CPU
        )


def make_edges_bidirectional(edge_index: torch.Tensor, edge_features: torch.Tensor) -> tuple:
    """Convert directed edges to bidirectional for unrooted tree representation.
    
    Args:
        edge_index: [2, num_edges] directed edge indices (e.g., child→parent from DFS)
        edge_features: [num_edges, edge_feat_dim] features for directed edges
    
    Returns:
        bidirectional_edge_index: [2, 2*num_edges] with both (i→j) and (j→i)
        bidirectional_edge_features: [2*num_edges, edge_feat_dim] duplicated features
    
    Example:
        Input:  edge_index = [[0, 1], [1, 2]]  (0→1, 1→2)
        Output: edge_index = [[0, 1, 1, 2], [1, 0, 2, 1]]  (0→1, 1→0, 1→2, 2→1)
    """
    # Create reverse edges
    reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    
    # Concatenate original and reverse
    bidirectional_edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    bidirectional_edge_features = torch.cat([edge_features, edge_features], dim=0)
    
    return bidirectional_edge_index, bidirectional_edge_features


# ------------------------- SIMPLE Q NETWORK -------------------------


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
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

# ------------------------- DQN AGENT -------------------------


class DQNAgent:
    def __init__(self, feature_dim, hidden_dim, dropout_p, learning_rate, weight_decay, gamma, tau,
                 replay_size, replay_alpha, double_q, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.double_q = double_q

        # Q network and target network
        self.q_net = QNetwork(feature_dim, hidden_dim, dropout_p).to(self.device)
        self.target_net = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.replay = PrioritizedReplayBuffer(replay_size, replay_alpha, self.device)

    def select_actions(self, state_action_feats, temp, num_actions):
        with torch.no_grad():
            feats_t = torch.tensor(state_action_feats, dtype=torch.float32, device=self.device)
            q_values = self.q_net(feats_t).squeeze(-1)  # shape [num_actions]

            probs = torch.softmax(q_values / temp, dim=0)
            num_actions = min(num_actions, len(probs))
            actions = torch.multinomial(probs, num_actions, replacement=False).cpu().numpy()

            return actions

    def update(self, batch_size, beta):
        if len(self.replay) < batch_size:
            return None

        feats, rewards, next_feats, dones, indices, weights = self.replay.sample(batch_size, beta)

        # Current Q-values
        q_vals = self.q_net(feats)

        # Compute target values
        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)

            if self.double_q:
                # Double Q-learning: use online network to select actions
                q_next_online = self.q_net(next_feats_flat).view(B, N)
                best_actions = q_next_online.argmax(dim=1)  # (B,)

                # Use target network to evaluate selected actions
                q_next_target = self.target_net(next_feats_flat).view(B, N)
                q_next_max = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)  # (B,)
            else:
                # Standard DQN: use target network for both selection and evaluation
                q_next_flat = self.target_net(next_feats_flat).view(B, N)
                q_next_max = q_next_flat.max(dim=1)[0]  # (B,)

            q_target = rewards + self.gamma * (1 - dones) * q_next_max

        # TD loss
        td_error = q_target - q_vals
        loss = (weights * td_error.pow(2)).mean()

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update PER priorities
        new_prios = td_error.abs().detach() + 1e-6
        self.replay.update_priorities(indices, new_prios)

        # Update target network
        for t_p, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            t_p.data.lerp_(p.data, self.tau)

        return loss.item()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), str(path))


# ------------------------- SOFT Q-learning AGENT (CLIPPED DOUBLE Q) -------------------------


class SoftQAgent:
    def __init__(self, feature_dim, hidden_dim, dropout_p, learning_rate, weight_decay, gamma, tau,
                 temp_alpha_init, replay_size, replay_alpha, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau

        # Learnable temperature parameter (log-scale for stability)
        self.log_alpha = torch.tensor([torch.log(torch.tensor(temp_alpha_init))],
                                      dtype=torch.float32,
                                      device=self.device,
                                      requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # Clipped double Q networks
        self.q1 = QNetwork(feature_dim, hidden_dim, dropout_p).to(self.device)
        self.q2 = QNetwork(feature_dim, hidden_dim, dropout_p).to(self.device)
        self.target_q1 = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.target_q2 = QNetwork(feature_dim, hidden_dim).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()

        self.optimizer1 = optim.AdamW(self.q1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer2 = optim.AdamW(self.q2.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.replay = PrioritizedReplayBuffer(replay_size, replay_alpha, self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_actions(self, state_action_feats, num_actions):
        """
        Select an action given all possible (state, action) feature vectors.
        Each row of `state_action_feats` corresponds to one action for the same state.
        During training: sample from Boltzmann (softmax) policy.
        During evaluation: choose greedy (argmax) action.
        """
        with torch.no_grad():
            feats_t = torch.tensor(state_action_feats, dtype=torch.float32, device=self.device)
            q_values = self.q1(feats_t).squeeze(-1)  # shape [num_actions]

            alpha = self.log_alpha.exp()
            probs = torch.softmax(q_values / alpha, dim=0)
            num_actions = min(num_actions, len(probs))
            actions = torch.multinomial(probs, num_actions, replacement=False).cpu().numpy()

            return actions

    def update(self, batch_size, beta, target_entropy):
        if len(self.replay) < batch_size:
            return None

        feats, rewards, next_feats, dones, indices, weights = self.replay.sample(batch_size, beta)

        # Current Q-values
        q1_vals = self.q1(feats)
        q2_vals = self.q2(feats)

        # Target soft value
        with torch.no_grad():
            B, N, F = next_feats.shape
            next_feats_flat = next_feats.view(B * N, F)
            q1_next = self.target_q1(next_feats_flat).view(B, N)
            q2_next = self.target_q2(next_feats_flat).view(B, N)

            q_min = torch.min(q1_next, q2_next)

            alpha = self.log_alpha.exp()
            soft_value = alpha * torch.logsumexp(q_min / alpha, dim=1)
            q_target = rewards + self.gamma * (1 - dones) * soft_value

        # TD losses
        td_error1 = q_target - q1_vals
        td_error2 = q_target - q2_vals
        loss1 = (weights * td_error1.pow(2)).mean()
        loss2 = (weights * td_error2.pow(2)).mean()

        # Update Q-networks
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        # Update alpha (temperature)
        with torch.no_grad():
            q_next = self.q1(next_feats_flat).view(B, N)
            log_probs = torch.log_softmax(q_next / alpha, dim=1)
            probs = torch.softmax(q_next / alpha, dim=1)
            policy_entropy = -(probs * log_probs).sum(dim=1).mean()

        alpha_loss = self.log_alpha * (policy_entropy - target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # self.log_alpha.data.clamp_(min=-10.0, max=2.0)

        # Update PER priorities
        new_prios = 0.5 * (td_error1.abs() + td_error2.abs()).detach() + 1e-6
        self.replay.update_priorities(indices, new_prios)

        # Update target networks
        for t_p, p in zip(self.target_q1.parameters(), self.q1.parameters()):
            t_p.data.lerp_(p.data, self.tau)
        for t_p, p in zip(self.target_q2.parameters(), self.q2.parameters()):
            t_p.data.lerp_(p.data, self.tau)

        return (loss1.item() + loss2.item()) / 2, policy_entropy.item()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q1.state_dict(), str(path))


# ------------------------- GNN Q NETWORK -------------------------


class GNNQNetwork(nn.Module):
    """GATv2-based Q-Network for phylogenetic trees.
    
    Architecture:
    1. Encode tree structure with GATv2 layers
    2. Encode action embedding with MLP (using NODE EMBEDDINGS + metadata)
    3. Combine tree + action representations
    4. Output Q-value
    
    Typical usage with atomic features:
    - node_feat_dim=1: [is_leaf] 
    - edge_feat_dim=1: [branch_length]
    - action_dim=7: [4 node indices + 3 metadata floats]
    """
    
    def __init__(self, node_feat_dim, edge_feat_dim, action_dim, hidden_dim, 
                 num_gat_layers=3, num_attention_heads=4, dropout_p=0.0, num_action_node_indices=4):
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GNN agents")
        
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_node_indices = num_action_node_indices
        
        # Node feature projection
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        
        # GATv2 layers for tree encoding
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_attention_heads,
                    heads=num_attention_heads,
                    dropout=dropout_p,
                    edge_dim=edge_feat_dim,
                    concat=True
                )
            )
        
        # Action encoder (processes action embedding)
        # Input: (4 * hidden_dim for node embeddings) + (action_dim - 4 for metadata)
        mlp_input_dim = (num_action_node_indices * hidden_dim) + (action_dim - num_action_node_indices)
        
        self.action_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Final Q-value head (matches hand-crafted 3-layer MLP structure)
        # Input: tree_emb (3×hidden_dim from sum+mean+max) + action_emb (hidden_dim) = 4×hidden_dim
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _validate_bidirectional_edges(self, edge_index: torch.Tensor):
        """Validate that edges are bidirectional (for unrooted trees).
        
        Raises RuntimeError if any edge (i→j) doesn't have corresponding (j→i).
        """
        edges_set = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edges_set.add((src, dst))
        
        # Check every edge has its reverse
        missing_reverses = []
        for src, dst in edges_set:
            if (dst, src) not in edges_set:
                missing_reverses.append((src, dst))
        
        if missing_reverses:
            raise RuntimeError(
                f"Graph edges are NOT bidirectional! Missing reverse edges for: {missing_reverses[:5]}... "
                f"(showing first 5 of {len(missing_reverses)} missing). "
                f"Phylogenetic trees are unrooted - use make_edges_bidirectional() when creating GraphData!"
            )
    
    def encode_tree(self, graph_data: Data, batch_idx: Optional[torch.Tensor] = None):
        """Encode tree structure once. EFFICIENCY: Call this once per tree!
        
        CRITICAL: This assumes graph_data contains BIDIRECTIONAL edges for unrooted trees.
        Use make_edges_bidirectional() when creating GraphData from TreePreprocessor.
        
        Args:
            graph_data: PyG Data object with node_features, edge_index, edge_attr
                        edge_index MUST be bidirectional (both i→j and j→i for each edge)
            batch_idx: [num_nodes] batch assignment for each node (for batched graphs)
        
        Returns:
            tree_embedding: [3*hidden_dim] or [batch_size, 3*hidden_dim]
                           Concatenation of sum, mean, and max pooling
            node_embeddings: [num_nodes, hidden_dim] - Learned embeddings for each node
        """
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        
        # VALIDATION: Check edge bidirectionality (only in first call for efficiency)
        if not hasattr(self, '_validated_bidirectional'):
            self._validate_bidirectional_edges(edge_index)
            self._validated_bidirectional = True
        
        # Encode nodes
        x = self.node_encoder(x)  # [num_nodes, hidden_dim]
        x = torch.relu(x)
        
        # GATv2 message passing
        for gat_layer in self.gat_layers:
            x_new = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = torch.relu(x_new) + x  # Residual connection
        
        node_embeddings = x  # Keep node embeddings for action encoding
        
        # Global tree representation using multiple aggregations to preserve information
        # Concatenate sum, mean, and max to capture:
        #   - sum: total information (size-aware, preserves counts)
        #   - mean: average information (size-invariant, standard patterns)
        #   - max: extreme values (captures outliers and max characteristics)
        if batch_idx is not None:
            sum_pool = global_add_pool(x, batch_idx)    # [batch_size, hidden_dim]
            mean_pool = global_mean_pool(x, batch_idx)  # [batch_size, hidden_dim]
            # Use manual max pooling to avoid slow fallback when torch-scatter not installed
            batch_size = batch_idx.max().item() + 1
            max_pool = torch.zeros(batch_size, x.shape[-1], device=x.device)
            for i in range(batch_size):
                mask = batch_idx == i
                max_pool[i] = x[mask].max(dim=0)[0]
            tree_emb = torch.cat([sum_pool, mean_pool, max_pool], dim=-1)  # [batch_size, hidden_dim * 3]
        else:
            sum_pool = x.sum(dim=0)     # [hidden_dim]
            mean_pool = x.mean(dim=0)   # [hidden_dim]
            max_pool = x.max(dim=0)[0]  # [hidden_dim]
            tree_emb = torch.cat([sum_pool, mean_pool, max_pool], dim=-1)  # [hidden_dim * 3]
        
        return tree_emb, node_embeddings
    
    def forward(self, graph_data: Data, action_tensor: torch.Tensor, batch_idx: Optional[torch.Tensor] = None, 
                tree_embedding: Optional[torch.Tensor] = None, node_embeddings: Optional[torch.Tensor] = None,
                batch_ptr: Optional[torch.Tensor] = None):
        """Compute Q-value for (graph, action) pair.
        
        Args:
            graph_data: PyG Data object (ignored if tree_embedding provided)
            action_tensor: [..., action_dim] - Contains node indices + metadata
            batch_idx: [num_nodes] batch assignment (ignored if tree_embedding provided)
            tree_embedding: Pre-computed tree encoding [3*hidden_dim]
            node_embeddings: Pre-computed node embeddings [num_nodes, hidden_dim]
            batch_ptr: [batch_size + 1] - Start indices for each graph in batch (needed for batched actions)
        
        Returns:
            Q-value: scalar or [num_actions]
        """
        # Use pre-computed tree embedding if provided (EFFICIENCY!)
        if tree_embedding is None or node_embeddings is None:
            tree_emb, node_embs = self.encode_tree(graph_data, batch_idx)
        else:
            tree_emb = tree_embedding
            node_embs = node_embeddings
        
        # --- ENCODE ACTION USING NODE EMBEDDINGS ---
        
        # 1. Extract indices and metadata
        # action_tensor: [..., action_dim]
        node_indices = action_tensor[..., :self.num_action_node_indices].long() # [..., 4]
        metadata = action_tensor[..., self.num_action_node_indices:]            # [..., 3]
        
        # 2. Handle batch shifting if needed
        if batch_ptr is not None:
            # batch_ptr: [batch_size + 1]
            # We need shifts for each item in the batch.
            # Assuming action_tensor dim 0 is batch dimension.
            shifts = batch_ptr[:-1].to(node_indices.device)
            
            # Broadcast shifts to match node_indices shape
            # node_indices: [batch_size, ..., 4]
            # shifts: [batch_size] -> [batch_size, 1, ..., 1]
            # We need to append 1s for all dimensions after the batch dimension
            view_shape = [-1] + [1] * (node_indices.dim() - 1)
            shifts = shifts.view(view_shape)
            
            node_indices = node_indices + shifts
            
        # 3. Gather node embeddings
        # node_embs: [total_nodes, hidden_dim]
        # node_indices: [..., 4]
        
        # Flatten indices to gather
        flat_indices = node_indices.view(-1)
        gathered_embs = node_embs[flat_indices] # [total_indices, hidden_dim]
        
        # Reshape back: [..., 4, hidden_dim]
        gathered_embs = gathered_embs.view(*node_indices.shape, -1)
        
        # Flatten the 4 embeddings: [..., 4 * hidden_dim]
        gathered_embs_flat = gathered_embs.view(*node_indices.shape[:-1], -1)
        
        # 4. Concatenate with metadata
        action_input = torch.cat([gathered_embs_flat, metadata], dim=-1)
        
        # Encode action
        action_emb = self.action_encoder(action_input)  # [..., hidden_dim]
        
        # Combine tree and action representations
        # tree_emb is [3*hidden_dim] or [batch_size, 3*hidden_dim]
        # action_emb is [hidden_dim] or [batch_size, num_actions, hidden_dim]
        
        if action_emb.dim() == 1:
            # Single action, single tree
            combined = torch.cat([tree_emb, action_emb], dim=0)
        elif action_emb.dim() == 2 and tree_emb.dim() == 1:
             # Multiple actions, single tree
            tree_emb_expanded = tree_emb.unsqueeze(0).expand(action_emb.shape[0], -1)
            combined = torch.cat([tree_emb_expanded, action_emb], dim=1)
        elif action_emb.dim() == 2 and tree_emb.dim() == 2:
            # Batched trees, one action per tree (e.g. Q-value of chosen action)
            combined = torch.cat([tree_emb, action_emb], dim=1)
        elif action_emb.dim() == 3 and tree_emb.dim() == 2:
            # Batched trees, multiple actions per tree (e.g. target Q-values)
            # tree_emb: [batch_size, 3*hidden_dim] -> [batch_size, 1, 3*hidden_dim]
            tree_emb_expanded = tree_emb.unsqueeze(1).expand(-1, action_emb.shape[1], -1)
            combined = torch.cat([tree_emb_expanded, action_emb], dim=2)
        else:
             raise ValueError(f"Unexpected shapes: tree {tree_emb.shape}, action {action_emb.shape}")
        
        # Compute Q-value
        q_value = self.q_head(combined)  # [..., 1]
        
        return q_value.squeeze(-1)


# ------------------------- GNN PRIORITIZED REPLAY BUFFER -------------------------


class GNNPrioritizedReplayBuffer:
    """Replay buffer optimized for 80GB A100 GPU.
    
    EFFICIENCY: Stores ALL data on GPU for zero-latency sampling!
    - All GraphData objects kept on GPU
    - No CPU storage or caching
    - Instant sampling with zero transfers
    - With 80GB VRAM, can store ~10K graphs easily
    """
    
    def __init__(self, capacity, alpha, device):
        self.capacity = capacity
        self.alpha = alpha
        self.device = torch.device(device) if isinstance(device, str) else device
        self.pos = 0
        self.full = False
        
        # EFFICIENCY: 80GB A100 - Store EVERYTHING on GPU!
        # No CPU storage, no caching complexity - just raw GPU speed!
        self.tree_graphs = [None] * capacity          # GraphData objects (ON GPU)
        self.action_tensors = [None] * capacity       # Tensor [action_dim] (ON GPU)
        self.action_indices = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_tree_graphs = [None] * capacity     # GraphData objects (ON GPU)
        self.next_action_tensors = [None] * capacity  # Tensor [num_actions, action_dim] (ON GPU)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        self.priorities = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.max_prio = 1.0
    
    def push(self, tree_graph: GraphData, action_embedding: torch.Tensor, action_idx: int, reward: float, 
             next_tree_graph: GraphData, next_action_embeddings: torch.Tensor, done: bool):
        """Add transition to buffer - KEEP EVERYTHING ON GPU!"""
        # EFFICIENCY: Move to GPU if not already there, then store
        if tree_graph.node_features.device.type != self.device.type:
            tree_graph = tree_graph.to(self.device)
        if next_tree_graph.node_features.device.type != self.device.type:
            next_tree_graph = next_tree_graph.to(self.device)
            
        # Convert actions to tensors immediately -> already tensors!
        # CRITICAL: Detach to prevent memory leaks from computation graph!
        action_tensor = action_embedding.to(self.device).detach()
        next_action_tensor = next_action_embeddings.to(self.device).detach()
        
        # OPTIMIZATION: Store PyG Data objects directly to avoid recreation in update loop
        # We attach node_name_to_idx to the Data object so we can use it later if needed
        # CRITICAL FIX: Do NOT attach dictionary to Data object directly as it breaks PyG collation
        # PyG tries to collate all attributes, and it doesn't know how to collate dictionaries
        tree_data = Data(
            x=tree_graph.node_features,
            edge_index=tree_graph.edge_index,
            edge_attr=tree_graph.edge_features
        )
        # tree_data.node_name_to_idx = tree_graph.node_name_to_idx  <-- REMOVED
        
        next_tree_data = Data(
            x=next_tree_graph.node_features,
            edge_index=next_tree_graph.edge_index,
            edge_attr=next_tree_graph.edge_features
        )
        # next_tree_data.node_name_to_idx = next_tree_graph.node_name_to_idx <-- REMOVED

        # Store directly (already on GPU!)
        self.tree_graphs[self.pos] = tree_data
        self.action_tensors[self.pos] = action_tensor
        self.action_indices[self.pos] = action_idx
        self.rewards[self.pos] = reward
        self.next_tree_graphs[self.pos] = next_tree_data
        self.next_action_tensors[self.pos] = next_action_tensor
        self.dones[self.pos] = float(done)
        self.priorities[self.pos] = self.max_prio
        
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0
    
    def sample(self, batch_size, beta):
        """Sample batch - ALL DATA ALREADY ON GPU!"""
        n = self.capacity if self.full else self.pos
        prios = self.priorities[:n]
        probs = (prios ** self.alpha)
        probs /= probs.sum()
        
        indices = torch.multinomial(probs, batch_size, replacement=False)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # EFFICIENCY: Everything is already on GPU! Just index!
        batch_tree_graphs = [self.tree_graphs[i.item()] for i in indices]
        batch_action_tensors = [self.action_tensors[i.item()] for i in indices]
        batch_action_indices = self.action_indices[indices]
        batch_rewards = self.rewards[indices]
        batch_next_tree_graphs = [self.next_tree_graphs[i.item()] for i in indices]
        batch_next_action_tensors = [self.next_action_tensors[i.item()] for i in indices]
        batch_dones = self.dones[indices]
        
        return (
            batch_tree_graphs,
            batch_action_tensors,
            batch_action_indices,
            batch_rewards,
            batch_next_tree_graphs,
            batch_next_action_tensors,
            batch_dones,
            indices,
            weights
        )
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        self.priorities[indices] = priorities
        self.max_prio = max(self.max_prio, priorities.max().item())
    
    def __len__(self):
        return self.capacity if self.full else self.pos


# ------------------------- GNN SOFT Q AGENT -------------------------


class GNNSoftQAgent:
    """Soft Q-learning agent with GNN feature extraction.
    
    Uses GATv2Conv for tree encoding and learnable temperature parameter.
    Implements clipped double Q-learning for stability.
    
    Typical initialization with atomic features:
        GNNSoftQAgent(
            node_feat_dim=1,  # [is_leaf]
            edge_feat_dim=1,  # [branch_length]
            action_dim=7,     # [4 node indices + 3 metadata]
            hidden_dim=256,
            num_gat_layers=3,
            num_attention_heads=4,
            ...
        )
    """
    
    def __init__(self, node_feat_dim, edge_feat_dim, action_dim, hidden_dim, 
                 num_gat_layers, num_attention_heads, dropout_p, learning_rate, 
                 weight_decay, gamma, tau, temp_alpha_init, replay_size, replay_alpha, device=None):
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GNN agents")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.action_dim = action_dim
        
        # Learnable temperature parameter (log-scale for stability)
        self.log_alpha = torch.tensor([torch.log(torch.tensor(temp_alpha_init))],
                                      dtype=torch.float32,
                                      device=self.device,
                                      requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Clipped double Q networks
        self.q1 = GNNQNetwork(node_feat_dim, edge_feat_dim, action_dim, hidden_dim,
                              num_gat_layers, num_attention_heads, dropout_p).to(self.device)
        self.q2 = GNNQNetwork(node_feat_dim, edge_feat_dim, action_dim, hidden_dim,
                              num_gat_layers, num_attention_heads, dropout_p).to(self.device)
        
        self.target_q1 = GNNQNetwork(node_feat_dim, edge_feat_dim, action_dim, hidden_dim,
                                      num_gat_layers, num_attention_heads).to(self.device)
        self.target_q2 = GNNQNetwork(node_feat_dim, edge_feat_dim, action_dim, hidden_dim,
                                      num_gat_layers, num_attention_heads).to(self.device)
        
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()
        
        self.optimizer1 = optim.AdamW(self.q1.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer2 = optim.AdamW(self.q2.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # EFFICIENCY: 80GB A100 - store EVERYTHING on GPU!
        self.replay = GNNPrioritizedReplayBuffer(replay_size, replay_alpha, self.device)
    
    @property
    def alpha(self):
        return self.log_alpha.exp().item()
    
    def select_actions(self, tree_graph: GraphData, action_embeddings: torch.Tensor, num_actions: int):
        """Select actions using Boltzmann policy.
        
        EFFICIENCY: Encodes tree ONCE, then computes Q-values for all actions in batch!
        
        Args:
            tree_graph: Current tree structure
            action_embeddings: Tensor of all possible actions [num_actions, action_dim]
            num_actions: Number of actions to sample
        
        Returns:
            Array of selected action indices
        """
        with torch.no_grad():
            # Move graph to device
            tree_graph = tree_graph.to(self.device)
            
            # Convert to PyG Data format
            graph_data = Data(
                x=tree_graph.node_features,
                edge_index=tree_graph.edge_index,
                edge_attr=tree_graph.edge_features
            )
            
            # Encode tree through GAT layers
            tree_embedding, node_embeddings = self.q1.encode_tree(graph_data)
            
            # EFFICIENCY: Batch all action embeddings
            # action_embeddings is already a tensor [num_actions, action_dim]
            action_tensors = action_embeddings.to(self.device)
            
            # EFFICIENCY: Compute Q-values for all actions in ONE forward pass!
            q_values = self.q1(graph_data, action_tensors, tree_embedding=tree_embedding, node_embeddings=node_embeddings)  # [num_possible_actions]
            
            # Boltzmann policy
            alpha = self.log_alpha.exp()
            probs = torch.softmax(q_values / alpha, dim=0)
            num_actions = min(num_actions, len(probs))
            actions = torch.multinomial(probs, num_actions, replacement=False).cpu().numpy()
            
            return actions
    
    def update(self, batch_size, beta, target_entropy):
        """Perform one update step."""
        if len(self.replay) < batch_size:
            return None, None
        
        # Sample batch
        (batch_tree_graphs, batch_action_tensors, batch_action_indices, batch_rewards, 
         batch_next_tree_graphs, batch_next_action_tensors, batch_dones,
         indices, weights) = self.replay.sample(batch_size, beta)
        
        # Move weights to device
        weights = weights.to(self.device)
        
        # EFFICIENCY: Batch all graphs for parallel processing
        # batch_tree_graphs is now a list of Data objects (already on GPU)
        batched_graph = Batch.from_data_list(batch_tree_graphs)
        batched_actions = torch.stack(batch_action_tensors)  # [batch_size, action_dim]
        
        # Encode all trees at once with each network
        tree_embeddings_q1, node_embeddings_q1 = self.q1.encode_tree(batched_graph, batched_graph.batch)  # [batch_size, 3*hidden_dim]
        tree_embeddings_q2, node_embeddings_q2 = self.q2.encode_tree(batched_graph, batched_graph.batch)  # [batch_size, 3*hidden_dim]
        
        # Get batch pointers for index shifting
        batch_ptr = batched_graph.ptr
        
        # Batch compute Q-values
        q1_vals = self.q1(None, batched_actions, tree_embedding=tree_embeddings_q1, node_embeddings=node_embeddings_q1, batch_ptr=batch_ptr)
        q2_vals = self.q2(None, batched_actions, tree_embedding=tree_embeddings_q2, node_embeddings=node_embeddings_q2, batch_ptr=batch_ptr)
        
        # EFFICIENCY: Compute target Q-values with FULLY BATCHED operations
        with torch.no_grad():
            # Batch all next-state graphs (already Data objects)
            batched_next_graph = Batch.from_data_list(batch_next_tree_graphs)
            next_batch_ptr = batched_next_graph.ptr
            
            # Stack all actions: [batch_size, num_actions, action_dim]
            # Note: This assumes num_actions is constant across batch. 
            # If not, we need to handle variable sizes, but for now assuming constant for speed.
            all_next_actions = torch.stack(batch_next_action_tensors)
            
            # Encode all next-state trees in parallel
            next_tree_embeddings, next_node_embeddings = self.target_q1.encode_tree(batched_next_graph, batched_next_graph.batch)
            
            # Evaluate ALL moves with GNN (same for q1 and q2)
            q1_next = self.target_q1(None, all_next_actions, tree_embedding=next_tree_embeddings, node_embeddings=next_node_embeddings, batch_ptr=next_batch_ptr)
            q2_next = self.target_q2(None, all_next_actions, tree_embedding=next_tree_embeddings, node_embeddings=next_node_embeddings, batch_ptr=next_batch_ptr)
            
            # Min over both Q-networks
            q_min = torch.min(q1_next, q2_next)  # [batch_size, num_actions]
            
            # Soft value computation (vectorized over batch)
            alpha = self.log_alpha.exp()
            soft_values = alpha * torch.logsumexp(q_min / alpha, dim=1)  # [batch_size]
            
            # Compute targets
            q_targets = batch_rewards + self.gamma * (1 - batch_dones) * soft_values  # [batch_size]
        
        # TD losses
        td_error1 = q_targets - q1_vals
        td_error2 = q_targets - q2_vals
        
        # Ensure td_errors are 1D (squeeze any extra dimensions)
        td_error1 = td_error1.squeeze()
        td_error2 = td_error2.squeeze()
        
        loss1 = (weights * td_error1.pow(2)).mean()
        loss2 = (weights * td_error2.pow(2)).mean()
        
        # Update Q-networks
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        
        # Update alpha (temperature)
        with torch.no_grad():
            # EFFICIENCY: Graph already on GPU!
            tree_graph = batch_tree_graphs[0]
            action_tensors = batch_next_action_tensors[0]
            
            # EFFICIENCY: Encode tree once, batch all actions
            # tree_graph is now a Data object
            tree_embedding = self.q1.encode_tree(tree_graph)
            q_vals = self.q1(tree_graph, action_tensors, tree_embedding=tree_embedding)
            
            alpha = self.log_alpha.exp()
            log_probs = torch.log_softmax(q_vals / alpha, dim=0)
            probs = torch.softmax(q_vals / alpha, dim=0)
            policy_entropy = -(probs * log_probs).sum()
        
        alpha_loss = self.log_alpha * (policy_entropy - target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update PER priorities - ensure 1D tensor
        new_prios = 0.5 * (td_error1.abs() + td_error2.abs()).detach() + 1e-6
        new_prios = new_prios.flatten()  # Force to 1D
        self.replay.update_priorities(indices, new_prios)
        
        # Update target networks
        for t_p, p in zip(self.target_q1.parameters(), self.q1.parameters()):
            t_p.data.lerp_(p.data, self.tau)
        for t_p, p in zip(self.target_q2.parameters(), self.q2.parameters()):
            t_p.data.lerp_(p.data, self.tau)
        
        return (loss1.item() + loss2.item()) / 2, policy_entropy.item()
    
    def save(self, path: Path):
        """Save model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'node_feat_dim': self.node_feat_dim,
            'edge_feat_dim': self.edge_feat_dim,
            'action_dim': self.action_dim
        }, str(path))