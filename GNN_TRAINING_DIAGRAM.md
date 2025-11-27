# GNN Training Process Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHYLOGENETIC TREE SEARCH                      │
│                     WITH GNN Q-LEARNING                          │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Training Flow

```
EPISODE START
     │
     ├─► [1] ENVIRONMENT RESET (~500ms)
     │    ├─ Sample random tree from dataset
     │    ├─ Extract graph structure:
     │    │   • Node features: [is_leaf] (1-dim per node)
     │    │   • Edge features: [branch_length] (1-dim per edge)
     │    │   • Edge index: bidirectional connections
     │    └─ Extract available SPR moves → ActionEmbeddings
     │         [4 node indices + 3 metadata floats = 7-dim per action]
     │
     ├─► [2] EPISODE LOOP (20 steps per episode)
     │    │
     │    ├─► [2a] ACTION SELECTION (~45ms)
     │    │    │
     │    │    ├─ INPUT: GraphData + List[ActionEmbedding]
     │    │    │
     │    │    ├─ TREE EMBEDDING CACHE CHECK:
     │    │    │   • Cache key: (edge_index, node_features, edge_features)
     │    │    │   • If cached: Return embedding instantly (~1ms)
     │    │    │   • If miss: Compute GNN forward pass (~35ms)
     │    │    │   • Cache hit rate: ~80-90% after warmup
     │    │    │
     │    │    ├─ GNN FORWARD PASS (only on cache miss):
     │    │    │   ┌──────────────────────────────────────┐
     │    │    │   │ Tree Encoding (encode_tree):         │
     │    │    │   │   • Node encoder: [1] → [256]       │
     │    │    │   │   • GAT Layer 1: message passing    │
     │    │    │   │   • GAT Layer 2: message passing    │
     │    │    │   │   • GAT Layer 3: message passing    │
     │    │    │   │   • Aggregation:                     │
     │    │    │   │     - SUM pool   → [256]            │
     │    │    │   │     - MEAN pool  → [256]            │
     │    │    │   │     - MAX pool   → [256]            │
     │    │    │   │   • Concat → [768] tree embedding   │
     │    │    │   └──────────────────────────────────────┘
     │    │    │   
     │    │   For EACH action:
     │    │    │   ┌──────────────────────────────────────┐
     │    │    │   │ Action Tensor Cache:                 │
     │    │    │   │   • Convert ActionEmbedding → tensor │
     │    │    │   │   • Cached by node indices          │
     │    │    │   │   • Reused across episodes          │
     │    │    │   │ Action Encoding:                     │
     │    │    │   │   • Action encoder: [7] → [256]     │
     │    │    │   │ Q-Value Computation:                 │
     │    │    │   │   • Concat [tree:768 + action:256]  │
     │    │    │   │   • MLP: [1024] → [256] → [256] → [1]│
     │    │    │   │   • Output: Q-value (scalar)         │
     │    │    │   └──────────────────────────────────────┘
     │    │    │
     │    │    ├─ Compute Soft Q-values:
     │    │    │   Q_soft = α * log(Σ exp(Q_i / α))
     │    │    │
     │    │    └─ Sample action from softmax distribution
     │    │
     │    ├─► [2b] ENVIRONMENT STEP (~25ms)
     │    │    ├─ Apply SPR move to tree
     │    │    ├─ Compute log-likelihood (reward)
     │    │    └─ Extract next state (GraphData + actions)
     │    │
     │    ├─► [2c] REPLAY BUFFER PUSH (~1ms)
     │    │    └─ Store: (tree_graph, action, reward, 
     │    │              next_tree_graph, next_actions, done)
     │    │         All stored on GPU! (zero CPU transfers)
     │    │
     │    └─► [2d] AGENT UPDATE (~400ms) ✅ OPTIMIZED
     │         │   [Only if replay_buffer >= 1000 samples]
     │         │   [First update: ~700ms, after cache warmup: ~400ms]
     │         │
     │         ├─ SAMPLE BATCH (128 transitions):
     │         │   • 128 tree graphs
     │         │   • 128 actions
     │         │   • 128 rewards
     │         │   • 128 next-state graphs
     │         │   • 128 next-action sets
     │         │
     │         ├─ COMPUTE CURRENT Q-VALUES:
     │         │   ┌────────────────────────────────────────┐
     │         │   │ Batch Operations (PARALLEL):           │
     │         │   │   1. Create batch graph (128 trees)    │
     │         │   │   2. Q1 encode: 128 trees → [128, 768] │
     │         │   │   3. Q2 encode: 128 trees → [128, 768] │
     │         │   │   4. Q1 actions: [128, 7] → [128, 256] │
     │         │   │   5. Q2 actions: [128, 7] → [128, 256] │
     │         │   │   6. Q1 heads: [128, 1024] → [128]     │
     │         │   │   7. Q2 heads: [128, 1024] → [128]     │
     │         │   └────────────────────────────────────────┘
     │         │   Result: q1_vals[128], q2_vals[128]
     │         │
     │         ├─ COMPUTE TARGET Q-VALUES:
     │         │   ┌────────────────────────────────────────┐
     │         │   │ Next-State Processing (PARALLEL):      │
     │         │   │   1. Batch 128 next-state trees        │
     │         │   │   2. Target encode: [128, 768]         │
     │         │   │   3. Expand for actions:               │
     │         │   │      [128, N_actions, 768]             │
     │         │   │   4. Flatten: [128*N, 768]             │
     │         │   │   5. Encode all actions: [128*N, 256]  │
     │         │   │   6. Q-heads: [128*N] Q-values         │
     │         │   │   7. Reshape: [128, N_actions]         │
     │         │   │   8. Min(Q1, Q2) per state             │
     │         │   │   9. Soft-max value per state          │
     │         │   │  10. TD targets: r + γ * V_soft        │
     │         │   └────────────────────────────────────────┘
     │         │   Result: q_targets[128]
     │         │
     │         ├─ COMPUTE TD ERRORS:
     │         │   • td_error1 = q_targets - q1_vals
     │         │   • td_error2 = q_targets - q2_vals
     │         │
     │         ├─ UPDATE Q-NETWORKS:
     │         │   • loss1 = mean(weights * td_error1²)
     │         │   • loss2 = mean(weights * td_error2²)
     │         │   • Backward pass through GNN
     │         │   • Optimizer step (Adam)
     │         │
     │         ├─ UPDATE TEMPERATURE (α):
     │         │   • Compute policy entropy
     │         │   • Update log_α to match target entropy
     │         │
     │         ├─ UPDATE REPLAY PRIORITIES:
     │         │   • new_prio = |td_error1| + |td_error2|
     │         │
     │         └─ SOFT UPDATE TARGET NETWORKS:
     │             • target_θ ← τ*θ + (1-τ)*target_θ
     │
     └─► [3] EPISODE END
          ├─ Log metrics every 10 episodes
          └─ Save checkpoint every 1000 episodes
```

## GNN Architecture Detail

```
TREE → GRAPH NEURAL NETWORK → TREE EMBEDDING
  ↓
┌─────────────────────────────────────────────────┐
│ INPUT GRAPH                                      │
│   Nodes: [N, 1]  (is_leaf: 0.0 or 1.0)         │
│   Edges: [E, 1]  (branch_length: continuous)    │
│   Edge Index: [2, E]  (bidirectional pairs)     │
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ NODE ENCODER                                     │
│   Linear: [1] → [256]                           │
│   ReLU activation                                │
└─────────────────────────────────────────────────┘
  ↓ [N, 256]
┌─────────────────────────────────────────────────┐
│ GAT LAYER 1 (4 attention heads)                 │
│   For each node:                                 │
│     - Gather messages from neighbors             │
│     - Weight by attention + edge features        │
│     - Aggregate with multi-head attention        │
│   Residual: x = ReLU(GAT(x)) + x               │
└─────────────────────────────────────────────────┘
  ↓ [N, 256]
┌─────────────────────────────────────────────────┐
│ GAT LAYER 2 (4 attention heads)                 │
│   Same structure as Layer 1                      │
│   Residual: x = ReLU(GAT(x)) + x               │
└─────────────────────────────────────────────────┘
  ↓ [N, 256]
┌─────────────────────────────────────────────────┐
│ GAT LAYER 3 (4 attention heads)                 │
│   Same structure as Layer 1                      │
│   Residual: x = ReLU(GAT(x)) + x               │
└─────────────────────────────────────────────────┘
  ↓ [N, 256]
┌─────────────────────────────────────────────────┐
│ GLOBAL POOLING (Information Preservation)       │
│   ┌─────────────────────────────────┐           │
│   │ SUM:  Σ node_features → [256]  │ Size-aware│
│   └─────────────────────────────────┘           │
│   ┌─────────────────────────────────┐           │
│   │ MEAN: avg(node_features) → [256]│ Normalized│
│   └─────────────────────────────────┘           │
│   ┌─────────────────────────────────┐           │
│   │ MAX:  max(node_features) → [256]│ Extremes  │
│   └─────────────────────────────────┘           │
│   CONCATENATE → [768]                           │
└─────────────────────────────────────────────────┘
  ↓
TREE EMBEDDING [768]
```

## Q-Network Architecture

```
[TREE EMBEDDING: 768] + [ACTION EMBEDDING: 256]
           ↓
    CONCATENATE
           ↓
       [1024]
           ↓
┌──────────────────────┐
│  Q-HEAD (3-layer MLP)│
│  Linear: [1024]→[256]│
│  ReLU + Dropout      │
│  Linear: [256]→[256] │
│  ReLU               │
│  Linear: [256]→[1]   │
└──────────────────────┘
           ↓
    Q-VALUE (scalar)
```

## Memory Layout (80GB A100 GPU)

```
┌─────────────────────────────────────────────────┐
│ GPU MEMORY (80GB) - ACTUAL USAGE PER AGENT      │
├─────────────────────────────────────────────────┤
│ MODEL PARAMETERS (~50M params, ~800MB)          │
│   - GNN Q1 Network (4 GAT layers, hidden=256)   │
│   - GNN Q2 Network                               │
│   - Target Q1 Network                            │
│   - Target Q2 Network                            │
├─────────────────────────────────────────────────┤
│ REPLAY BUFFER (10,000 transitions, ~26MB)       │
│   - Tree graphs: 20K × 708 bytes = 14MB         │
│   - Action embeddings with cached tensors: 10MB │
│   - Next action lists: ~2MB                      │
│   - Scalars (rewards, dones, priorities): <1MB  │
├─────────────────────────────────────────────────┤
│ TREE EMBEDDING CACHE (~30MB, grows over time)   │
│   - Cached GNN outputs: [768] per tree          │
│   - ~10,000 unique trees × 3KB each             │
│   - Key: graph structure hash                    │
│   - 80-90% hit rate after warmup                │
├─────────────────────────────────────────────────┤
│ TRAINING BATCH (~100MB during update)           │
│   - 128 batched graphs                           │
│   - Tree embeddings [128, 768]                  │
│   - Action embeddings [128, 256]                │
│   - Gradients & optimizer states                │
├─────────────────────────────────────────────────┤
│ PYTORCH/CUDA OVERHEAD (~300MB)                  │
│   - CUDA context, allocator reserves            │
├─────────────────────────────────────────────────┤
│ TOTAL USED: ~1.2 GB per agent                   │
│ WITH 2 AGENTS: ~2.4 GB                          │
│ AVAILABLE: ~77.6 GB (97% free!)                 │
│                                                  │
│ 💡 Could easily support:                        │
│   - replay_size=50,000 (130MB vs 26MB)         │
│   - hidden_dim=512 (4x model size)             │
│   - batch_size=512 (4x batch size)             │
└─────────────────────────────────────────────────┘
```

## Performance Breakdown

```
TIMING PER EPISODE (after caches warm up ~100 episodes):
┌────────────────────────────────────┐
│ Environment Reset:    ~30ms  (0.3%) │
│ Action Selection:     ~27ms  (3%)   │ ✅ Tree embedding cache
│ Environment Step:     ~31ms  (3%)   │
│ Replay Push:          ~1ms   (0%)   │
│ Agent Update:         ~400ms (94%)  │ ✅ Action tensor cache + batching
├────────────────────────────────────┤
│ TOTAL PER STEP:       ~489ms        │
│ TOTAL PER EPISODE:    ~9.8s (20 steps with updates) │
├────────────────────────────────────┤
│ 30,000 EPISODES:      ~82 hours (~3.4 days) │
└────────────────────────────────────┘

OPTIMIZED UPDATE BREAKDOWN (400ms):
  - GNN tree encoding:       ~35% (140ms)
    * 4 GAT layers x 128 trees (batched)
    * Attention computation
    * Edge feature processing
    * Manual max pooling (torch-scatter avoided)
  
  - Action encoding/Q-heads:  ~25% (100ms)
    * Batch action encoding
    * Tensor cache hits: ~90%
    * MLP forward passes
  
  - Target computation:       ~25% (100ms)
    * Next-state processing
    * Soft-value computation
    * Fully batched operations
  
  - Backward pass/optimizer:  ~15% (60ms)
    * Gradient computation
    * Parameter updates

CACHE PERFORMANCE:
  - Tree embedding cache:
    * Warmup: First ~100 episodes
    * Hit rate: 80-90% after warmup
    * Saves: ~30ms per action selection
  
  - Action tensor cache:
    * Stored in ActionEmbedding objects
    * Hit rate: 90%+ from replay buffer reuse
    * Saves: ~2.5s per update initially
```

## Key Optimizations Applied

```
✅ BATCHED OPERATIONS:
   - All 128 tree graphs encoded in parallel
   - All actions processed simultaneously
   - Full GPU storage (zero CPU transfers)
   - Separate Q1/Q2 action encoders (prevents graph sharing bug)

✅ INFORMATION PRESERVATION:
   - Triple pooling (sum + mean + max)
   - No hand-crafted features
   - Bidirectional edges for unrooted trees
   - Residual connections in GAT layers

✅ TREE EMBEDDING CACHE:
   - Caches GNN forward pass outputs (4 GAT layers)
   - Key: (edge_index, node_features, edge_features)
   - Stored on GPU in agent.tree_embedding_cache
   - 80-90% hit rate after warmup
   - Saves ~30ms per action selection
   - Memory: ~30MB for 10K unique trees

✅ ACTION TENSOR CACHE:
   - Caches converted action tensors in ActionEmbedding objects
   - Key: (4 node indices, device)
   - Persists across episodes via replay buffer
   - 90%+ hit rate from replay reuse
   - Saves ~2.5s per update (2560 conversions → instant lookups)
   - Memory: ~10MB for 10K actions

✅ MANUAL MAX POOLING:
   - Replaced global_max_pool to avoid torch-scatter dependency
   - Prevents slow CPU fallback
   - Simple loop over batch dimension

📊 PERFORMANCE GAINS:
   - Action selection: 39ms → 27ms (30% faster)
   - Agent update: 3300ms → 400ms (8x faster!)
   - Episode time: ~66s → ~9.8s (6.7x faster!)
   - 30K training: 23 days → 3.4 days (feasible!)
```

## Comparison: Hand-Crafted vs GNN

```
HAND-CRAFTED FEATURES:
┌─────────────────────────┐
│ Tree → 50+ statistics   │
│ ├─ num_leaves           │
│ ├─ avg_branch_length    │
│ ├─ tree_depth           │
│ └─ ... 47 more          │
│ → Fixed [50] vector     │
└─────────────────────────┘
  ↓
Human-designed features
Structure information LOST

GNN FEATURES:
┌─────────────────────────┐
│ Tree → Raw graph        │
│ ├─ node: [is_leaf]      │
│ ├─ edge: [branch_len]   │
│ └─ topology preserved   │
│ → Learned [768] vector  │
└─────────────────────────┘
  ↓
Learned representation
Structure information PRESERVED
```
