import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import defaultdict
from hyperparameters import DEFAULT_EVALUATION_OUTPUT_DIR, CHECKPOINT_EVERY, EPISODES, N_AGENTS


def plot_evaluation_results(results, output_dir):
    """Generate plots for each evaluation result."""
    
    # Create plots directory
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for i, result in enumerate(results):
        checkpoint = result["checkpoint"]
        sample = result["sample"]
        tree_type = result["tree_type"]
        
        likelihoods = result["likelihoods"]
        steps = list(range(len(likelihoods)))
        
        raxml_ll = result["raxml_ll"]
        max_pars_ll = result["max_pars_ll"]
        starting_ll = result["starting_ll"]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot agent likelihood trajectory
        plt.plot(steps, likelihoods, 'b-', linewidth=2, label='Agent Likelihood')
        
        # Add starting point
        plt.plot(0, starting_ll, 'go', markersize=8, label=f'Starting LL: {starting_ll:.4f}')
        
        # Add RAxML reference line
        if raxml_ll is not None:
            plt.axhline(y=raxml_ll, color='r', linestyle='--', linewidth=2, 
                       label=f'RAxML LL: {raxml_ll:.4f}')
        
        # Add max parsimony reference line
        if max_pars_ll is not None:
            plt.axhline(y=max_pars_ll, color='g', linestyle='-.', linewidth=2,
                       label=f'Max Pars LL: {max_pars_ll:.4f}')
        
        # Labels and title
        plt.xlabel('Steps')
        plt.ylabel('Log Likelihood (Accuracy)')
        plt.title(f'Evaluation: {checkpoint} on {sample} ({tree_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = f"{checkpoint}_{sample}_{tree_type}.png"
        plot_path = plots_dir / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {plot_path}")
    
    print(f"\nAll individual plots saved to: {plots_dir}")


def plot_learning_progression(results, output_dir):
    """Generate a plot showing progression for each individual agent."""
    
    # Group results by agent and episode
    agent_episode_data = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        checkpoint_name = result["checkpoint"]
        max_ll = max(result["likelihoods"])  # highest likelihood achieved during evaluation
        
        # Parse checkpoint name: agent_{agent_id}_ep{episode}.pt
        parts = checkpoint_name.replace('.pt', '').split('_')
        if len(parts) >= 3 and parts[0] == 'agent' and parts[2].startswith('ep'):
            agent_id = int(parts[1])
            episode = int(parts[2][2:])  # remove 'ep'
            
            # Collect all max_ll for this agent at this episode
            agent_episode_data[agent_id][episode].append(max_ll)
    
    # Calculate average accuracy per episode for each agent
    agent_progression = {}
    
    for agent_id in range(N_AGENTS):
        episodes = []
        avg_accuracies = []
        
        for ep in range(CHECKPOINT_EVERY, EPISODES + 1, CHECKPOINT_EVERY):
            if ep in agent_episode_data[agent_id] and agent_episode_data[agent_id][ep]:
                avg_acc = sum(agent_episode_data[agent_id][ep]) / len(agent_episode_data[agent_id][ep])
                episodes.append(ep)
                avg_accuracies.append(avg_acc)
        
        agent_progression[agent_id] = (episodes, avg_accuracies)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for agent_id in range(N_AGENTS):
        if agent_id in agent_progression:
            episodes, accuracies = agent_progression[agent_id]
            plt.plot(episodes, accuracies, 'o-', color=colors[agent_id], 
                    linewidth=2, markersize=6, 
                    label=f'Agent {agent_id}')
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Maximum Log Likelihood')
    plt.title('Individual Agent Learning Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "learning_progression.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved learning progression plot: {plot_path}")


def main():
    results_file = Path(DEFAULT_EVALUATION_OUTPUT_DIR) / "evaluation_results.json"
    output_dir = DEFAULT_EVALUATION_OUTPUT_DIR
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run evaluate.py first to generate the results.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate individual evaluation plots
    plot_evaluation_results(results, output_dir)
    
    # Generate learning progression plot
    plot_learning_progression(results, output_dir)


if __name__ == "__main__":
    main()