import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from tqdm import tqdm
from environment import PhyloEnv
from soft_dqn_agent import QNetwork


class FeatureImportanceAnalyzer:
    def __init__(self, checkpoint_path: Path, feature_dim: int, hidden_dim: int, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(feature_dim, hidden_dim).to(self.device)
        self.q_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.q_net.eval()
        self.feature_dim = feature_dim

    # ==================== SHAP METHODS ====================

    def collect_feature_samples(self, env: PhyloEnv, n_samples: int = 1000,
                                sample_nums=None, start_tree_set="test"):
        """
        Collect feature samples from the environment for SHAP analysis.

        Args:
            env: PhyloEnv instance
            n_samples: number of state-action pairs to collect
            sample_nums: specific samples to use (optional)
            start_tree_set: "train" or "test"

        Returns:
            features: array of shape (n_samples, feature_dim)
            q_values: array of shape (n_samples,) with Q-values
        """
        print(f"Collecting {n_samples} feature samples...")
        features_list = []
        q_values_list = []

        samples_collected = 0
        with tqdm(total=n_samples) as pbar:
            while samples_collected < n_samples:
                sample_num = np.random.choice(sample_nums) if sample_nums is not None else None
                tree_hash, feats = env.reset(sample_num=sample_num, start_tree_set=start_tree_set)

                # Collect features from this episode
                done = False
                steps = 0
                max_steps = min(20, n_samples - samples_collected)

                while not done and steps < max_steps:
                    # Store all possible actions for this state
                    for action_feat in feats:
                        features_list.append(action_feat)

                        # Compute Q-value
                        with torch.no_grad():
                            feat_t = torch.tensor(action_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                            q_val = self.q_net(feat_t).item()
                            q_values_list.append(q_val)

                        samples_collected += 1
                        pbar.update(1)

                        if samples_collected >= n_samples:
                            break

                    if samples_collected >= n_samples:
                        break

                    # Take a step
                    with torch.no_grad():
                        feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
                        q_vals = self.q_net(feats_t)
                        action_idx = torch.argmax(q_vals).item()

                    tree_hash, next_feats, reward, done = env.step(action_idx)
                    feats = next_feats
                    steps += 1

        features = np.array(features_list[:n_samples])
        q_values = np.array(q_values_list[:n_samples])

        return features, q_values

    def compute_shap_values(self, background_samples: np.ndarray,
                            test_samples: np.ndarray,
                            method: str = "kernel",
                            **kwargs):
        """
        Compute SHAP values for test samples.

        Args:
            background_samples: array of shape (n_background, feature_dim)
            test_samples: array of shape (n_test, feature_dim)
            method: "kernel", "deep", "gradient", or "sampling"
            **kwargs: additional arguments for SHAP explainer

        Returns:
            shap_values: array of shape (n_test, feature_dim)
            explainer: the SHAP explainer object
        """
        print(f"Computing SHAP values using {method} method...")

        if method == "kernel":
            # Model-agnostic, works with any model but slower
            explainer = shap.KernelExplainer(
                self._predict_q_values,
                background_samples,
                **kwargs
            )
            shap_values = explainer.shap_values(test_samples)

        elif method == "deep":
            # Faster for deep learning, uses DeepLIFT
            # Need to wrap the model properly
            background_t = torch.tensor(background_samples, dtype=torch.float32, device=self.device)
            test_t = torch.tensor(test_samples, dtype=torch.float32, device=self.device)

            explainer = shap.DeepExplainer(self.q_net, background_t)
            shap_values = explainer.shap_values(test_t)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if torch.is_tensor(shap_values):
                shap_values = shap_values.cpu().numpy()

        elif method == "gradient":
            # Uses integrated gradients
            background_t = torch.tensor(background_samples, dtype=torch.float32, device=self.device)
            test_t = torch.tensor(test_samples, dtype=torch.float32, device=self.device)

            explainer = shap.GradientExplainer(self.q_net, background_t)
            shap_values = explainer.shap_values(test_t)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if torch.is_tensor(shap_values):
                shap_values = shap_values.cpu().numpy()

        elif method == "sampling":
            # Faster approximation of KernelExplainer
            explainer = shap.SamplingExplainer(
                self._predict_q_values,
                background_samples,
                **kwargs
            )
            shap_values = explainer.shap_values(test_samples)

        else:
            raise ValueError(f"Unknown method: {method}")

        return shap_values, explainer

    def _predict_q_values(self, features):
        """Wrapper for Q-network prediction (for SHAP)."""
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32, device=self.device)
            return self.q_net(features).cpu().numpy()

    # ==================== PERMUTATION METHOD ====================

    def permutation_importance(self, env: PhyloEnv, n_episodes: int = 50,
                               sample_nums=None, start_tree_set="test"):
        """
        Measure feature importance by permuting each feature and measuring performance drop.
        """
        print("Computing baseline performance...")
        baseline_returns = self._evaluate_agent(env, n_episodes, sample_nums, start_tree_set)
        baseline_mean = np.mean(baseline_returns)

        importance_scores = np.zeros(self.feature_dim)
        importance_std = np.zeros(self.feature_dim)

        print(f"Computing permutation importance for {self.feature_dim} features...")
        for feat_idx in tqdm(range(self.feature_dim)):
            permuted_returns = self._evaluate_agent(
                env, n_episodes, sample_nums, start_tree_set, permute_feature=feat_idx
            )
            importance_scores[feat_idx] = baseline_mean - np.mean(permuted_returns)
            importance_std[feat_idx] = np.std(permuted_returns)

        return importance_scores, importance_std, baseline_mean

    def _evaluate_agent(self, env: PhyloEnv, n_episodes: int, sample_nums,
                        start_tree_set: str, permute_feature: int = None):
        """Run episodes and optionally permute a specific feature."""
        returns = []

        for ep in range(n_episodes):
            sample_num = np.random.choice(sample_nums) if sample_nums is not None else None
            tree_hash, feats = env.reset(sample_num=sample_num, start_tree_set=start_tree_set)

            episode_return = 0.0
            done = False

            if permute_feature is not None:
                original_values = feats[:, permute_feature].copy()

            while not done:
                if permute_feature is not None:
                    np.random.shuffle(feats[:, permute_feature])

                with torch.no_grad():
                    feats_t = torch.tensor(feats, dtype=torch.float32, device=self.device)
                    q_vals = self.q_net(feats_t)
                    action_idx = torch.argmax(q_vals).item()

                if permute_feature is not None:
                    feats[:, permute_feature] = original_values

                tree_hash, next_feats, reward, done = env.step(action_idx)
                episode_return += reward

                feats = next_feats
                if permute_feature is not None:
                    original_values = feats[:, permute_feature].copy()

            returns.append(episode_return)

        return np.array(returns)


# ==================== PLOTTING FUNCTIONS ====================

def plot_shap_summary(shap_values, features, feature_names=None, output_path=None,
                      max_display=30):
    """
    Create SHAP summary plot showing feature importance.

    Args:
        shap_values: array of shape (n_samples, n_features)
        features: array of shape (n_samples, n_features)
        feature_names: list of feature names
        output_path: path to save plot
        max_display: maximum number of features to display
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"SHAP summary plot saved to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_shap_bar(shap_values, feature_names=None, output_path=None, max_display=30):
    """
    Create SHAP bar plot showing mean absolute SHAP values.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_names=feature_names,
                      plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"SHAP bar plot saved to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_shap_dependence(shap_values, features, feature_idx, feature_names=None,
                         output_path=None):
    """
    Create SHAP dependence plot for a specific feature.
    Shows how SHAP values for a feature vary with the feature's value.
    """
    plt.figure(figsize=(10, 6))

    feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"

    shap.dependence_plot(
        feature_idx,
        shap_values,
        features,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"SHAP dependence plot saved to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_shap_force(explainer, shap_values, features, feature_names=None,
                    sample_idx=0, output_path=None):
    """
    Create SHAP force plot for a single prediction.
    Shows how features push the prediction higher or lower.
    """
    if output_path:
        shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx],
            features[sample_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"SHAP force plot saved to {output_path}")
        plt.close()
    else:
        return shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx],
            features[sample_idx],
            feature_names=feature_names
        )


def compare_methods(shap_importance, permutation_importance, feature_names=None,
                    output_path=None):
    """
    Compare SHAP-based importance with permutation importance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Sort by SHAP importance
    sorted_indices = np.argsort(shap_importance)[::-1]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(shap_importance))]

    sorted_names = [feature_names[i] for i in sorted_indices]
    y_pos = np.arange(len(sorted_indices))

    # SHAP plot
    ax1.barh(y_pos, shap_importance[sorted_indices], alpha=0.7, color='steelblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_names)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('SHAP-based Feature Importance')
    ax1.grid(axis='x', alpha=0.3)

    # Permutation plot
    ax2.barh(y_pos, permutation_importance[sorted_indices], alpha=0.7, color='coral')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Performance Drop')
    ax2.set_title('Permutation Feature Importance')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")
        plt.close()
    else:
        plt.show()


# ==================== MAIN ANALYSIS PIPELINE ====================

def analyze_checkpoint_with_shap(checkpoint_path: Path, samples_dir: Path,
                                 raxmlng_path: Path, feature_names: list,
                                 horizon: int = 20, hidden_dim: int = 256,
                                 n_background: int = 100, n_test: int = 500,
                                 shap_method: str = "kernel",
                                 output_dir: Path = None,
                                 compare_with_permutation: bool = False):
    """
    Complete pipeline to analyze feature importance using SHAP.

    Args:
        checkpoint_path: path to checkpoint file
        samples_dir: path to samples directory
        raxmlng_path: path to RAxML-NG
        feature_names: list of feature names
        horizon: episode horizon
        hidden_dim: Q-network hidden dimension
        n_background: number of background samples for SHAP
        n_test: number of test samples for SHAP
        shap_method: "kernel", "deep", "gradient", or "sampling"
        output_dir: directory to save results
        compare_with_permutation: also compute permutation importance for comparison
    """
    # Initialize environment
    env = PhyloEnv(samples_dir, raxmlng_path, horizon=horizon)
    tree_hash, feats = env.reset(start_tree_set="test")
    feature_dim = feats.shape[1]

    if len(feature_names) != feature_dim:
        print(f"Warning: {len(feature_names)} feature names provided but {feature_dim} features detected")
        feature_names = feature_names[:feature_dim] if len(feature_names) > feature_dim else \
            feature_names + [f"Feature_{i}" for i in range(len(feature_names), feature_dim)]

    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(checkpoint_path, feature_dim, hidden_dim)

    # Collect samples
    print(f"\n{'='*60}")
    print("Step 1: Collecting background samples for SHAP")
    print(f"{'='*60}")
    background_features, background_q = analyzer.collect_feature_samples(
        env, n_samples=n_background, start_tree_set="test"
    )

    print(f"\n{'='*60}")
    print("Step 2: Collecting test samples for SHAP")
    print(f"{'='*60}")
    test_features, test_q = analyzer.collect_feature_samples(
        env, n_samples=n_test, start_tree_set="test"
    )

    print(f"\nBackground Q-values: mean={background_q.mean():.3f}, std={background_q.std():.3f}")
    print(f"Test Q-values: mean={test_q.mean():.3f}, std={test_q.std():.3f}")

    # Compute SHAP values
    print(f"\n{'='*60}")
    print(f"Step 3: Computing SHAP values ({shap_method} method)")
    print(f"{'='*60}")

    shap_values, explainer = analyzer.compute_shap_values(
        background_features,
        test_features,
        method=shap_method
    )

    # Compute mean absolute SHAP values as importance scores
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Optionally compute permutation importance
    perm_importance = None
    if compare_with_permutation:
        print(f"\n{'='*60}")
        print("Step 4: Computing permutation importance for comparison")
        print(f"{'='*60}")
        perm_importance, perm_std, baseline = analyzer.permutation_importance(
            env, n_episodes=30, start_tree_set="test"
        )

    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save numerical results
        print(f"\n{'='*60}")
        print("Step 5: Saving results")
        print(f"{'='*60}")

        np.savez(
            output_dir / "shap_results.npz",
            shap_values=shap_values,
            shap_importance=shap_importance,
            background_features=background_features,
            test_features=test_features,
            feature_names=feature_names,
            test_q_values=test_q
        )

        # Generate plots
        print("\nGenerating SHAP plots...")

        # 1. Summary plot (beeswarm)
        plot_shap_summary(
            shap_values, test_features, feature_names,
            output_path=output_dir / "shap_summary.png"
        )

        # 2. Bar plot
        plot_shap_bar(
            shap_values, feature_names,
            output_path=output_dir / "shap_bar.png"
        )

        # 3. Dependence plots for top 5 features
        top_5_features = np.argsort(shap_importance)[::-1][:5]
        for rank, feat_idx in enumerate(top_5_features, 1):
            plot_shap_dependence(
                shap_values, test_features, feat_idx, feature_names,
                output_path=output_dir / f"shap_dependence_top{rank}_{feature_names[feat_idx].replace('/', '_')}.png"
            )

        # 4. Force plot for a few examples
        for i in [0, len(test_features)//2, -1]:
            plot_shap_force(
                explainer, shap_values, test_features, feature_names,
                sample_idx=i,
                output_path=output_dir / f"shap_force_sample{i}.png"
            )

        # 5. Comparison plot if permutation importance computed
        if perm_importance is not None:
            compare_methods(
                shap_importance, perm_importance, feature_names,
                output_path=output_dir / "shap_vs_permutation.png"
            )

        # Save text report
        with open(output_dir / "shap_report.txt", 'w') as f:
            f.write(f"SHAP Feature Importance Analysis\n")
            f.write(f"{'='*60}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Method: {shap_method}\n")
            f.write(f"Background samples: {n_background}\n")
            f.write(f"Test samples: {n_test}\n")
            f.write(f"\nTop Features by Mean |SHAP value|:\n")
            f.write(f"{'-'*60}\n")

            sorted_indices = np.argsort(shap_importance)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                f.write(f"{rank:2d}. {feature_names[idx]:40s} {shap_importance[idx]:8.4f}\n")

            if perm_importance is not None:
                f.write(f"\n\nPermutation Importance (for comparison):\n")
                f.write(f"{'-'*60}\n")
                sorted_perm = np.argsort(perm_importance)[::-1]
                for rank, idx in enumerate(sorted_perm, 1):
                    f.write(f"{rank:2d}. {feature_names[idx]:40s} {perm_importance[idx]:8.4f}\n")

        print(f"\n✅ All results saved to {output_dir}")

    return shap_values, shap_importance, explainer


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Define your feature names (update based on your actual features)
    feature_names = [
        "Total branch length",
        "Longest branch",
        "Branch length prune",
        "Branch length regraft",
        "Topology distance",
        "Branch length distance",
        "New branch length",
        "Leaves prune subtree",
        "Leaves remain subtree",
        "Leaves regraft right",
        "Leaves regraft left",
        "Sum branch lengths prune subtree",
        "Sum branch lengths regraft subtree",
        "Sum branch lengths regraft right",
        "Sum branch lengths regraft left",
        "Longest branch prune subtree",
        "Longest branch regraft subtree",
        "Longest branch regraft right",
        "Longest branch regraft left",
        "UPGMA support prune split",
        "UPGMA support regraft split",
        "UPGMA support new prune split",
        "UPGMA support new regraft split",
        "NJ support prune split",
        "NJ support regraft split",
        "NJ support new prune split",
        "NJ support new regraft split"
    ]

    checkpoint_path = Path("output/Size9Samples1Train100Test10/soft_08e5601b/checkpoints/agent_0_ep6000.pt")
    samples_dir = Path("output/Size9Samples100Train100Test10")
    raxmlng_path = Path("dependencies/raxmlng/raxml-ng")
    output_dir = Path("output/shap_analysis")

    # Run analysis
    shap_values, importance, explainer = analyze_checkpoint_with_shap(
        checkpoint_path=checkpoint_path,
        samples_dir=samples_dir,
        raxmlng_path=raxmlng_path,
        feature_names=feature_names,
        hidden_dim=256,
<<<<<<< HEAD
        n_background=100,  # Start small, increase if needed
        n_test=1000,
        shap_method="deep",  # "kernel", "deep", "gradient", or "sampling"
=======
        n_background=500,  # Start small, increase if needed
        n_test=1000,
        shap_method="kernel",  # "kernel", "deep", "gradient", or "sampling"
>>>>>>> 92cda4702c50d77b65fd8fb5b60baf1e3c13fcc5
        output_dir=output_dir,
        compare_with_permutation=True  # Set to False to skip permutation (faster)
    )

    print("\n" + "="*60)
    print("Most Important Features:")
    print("="*60)
    top_10 = np.argsort(importance)[::-1]
    for rank, idx in enumerate(top_10, 1):
        print(f"{rank:2d}. {feature_names[idx]:40s} {importance[idx]:.4f}")
