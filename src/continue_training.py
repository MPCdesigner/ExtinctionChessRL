"""
Continue training from existing model to create version 2
"""

import os
import shutil
import json
from datetime import datetime

from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator
from self_play_trainer import Trainer


def save_version(version_num, win_rate, model_path="models/final_model.json"):
    """Save a model version with metadata"""
    os.makedirs("models/versions", exist_ok=True)
    
    # Create version filename with win rate
    version_name = f"model_v{version_num}_{int(win_rate*100)}pct.json"
    version_path = os.path.join("models/versions", version_name)
    
    # Copy model
    shutil.copy(model_path, version_path)
    
    # Save metadata
    metadata = {
        "version": version_num,
        "win_rate_vs_random": win_rate,
        "timestamp": datetime.now().isoformat(),
        "path": version_path
    }
    
    metadata_path = os.path.join("models/versions", f"metadata_v{version_num}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved version {version_num} with {win_rate:.1%} win rate to {version_path}")
    return version_path


def continue_training(start_version=1, target_version=2, additional_iterations=20):
    """Continue training from an existing model"""
    
    print(f"Loading version {start_version} model...")
    
    # Load the existing model
    config = NetworkConfig(
        input_size=39,
        hidden_sizes=[128, 64, 32],
        learning_rate=0.001
    )
    network = SimpleNeuralNetwork(config)
    
    # Try to load the latest model
    if os.path.exists("models/final_model.json"):
        network.load("models/final_model.json")
        print("Loaded existing model")
    else:
        print("No existing model found, starting fresh")
    
    # Create trainer with the loaded model
    trainer = Trainer(network, save_dir="models", log_file=f"training_log_v{target_version}.json")
    
    # Test initial performance
    print("\nTesting initial model performance...")
    initial_win_rate = trainer.test_against_random(num_games=50)
    print(f"Starting win rate: {initial_win_rate:.1%}")
    
    # Save as version 1 if this is the first continuation
    if start_version == 1:
        save_version(1, initial_win_rate)
    
    # Continue training with adjusted parameters
    print(f"\n{'='*50}")
    print(f"Training version {target_version}")
    print(f"Additional iterations: {additional_iterations}")
    print(f"{'='*50}")
    
    # Reduce exploration rate since model is already decent
    trainer.agent.exploration_rate = 0.15
    
    # Train with more games per iteration for better learning
    trainer.training_loop(
        iterations=additional_iterations,
        games_per_iteration=30  # More games for better training
    )
    
    # Test final performance
    print("\n" + "="*50)
    print("Testing improved model...")
    final_win_rate = trainer.test_against_random(num_games=100)  # More games for accurate measurement
    
    # Save as new version
    save_version(target_version, final_win_rate)
    
    # Show improvement
    improvement = final_win_rate - initial_win_rate
    print("\n" + "="*50)
    print(f"RESULTS:")
    print(f"Version {start_version} win rate: {initial_win_rate:.1%}")
    print(f"Version {target_version} win rate: {final_win_rate:.1%}")
    print(f"Improvement: {improvement:+.1%}")
    
    # Save comparison results
    comparison = {
        "version_1": {
            "win_rate": float(initial_win_rate),
            "version": start_version
        },
        "version_2": {
            "win_rate": float(final_win_rate),
            "version": target_version,
            "additional_training": additional_iterations
        },
        "improvement": float(improvement)
    }
    
    with open("models/version_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return final_win_rate


def train_multiple_versions(num_versions=3, iterations_per_version=20):
    """Train multiple versions progressively"""
    
    win_rates = []
    
    for version in range(1, num_versions + 1):
        print(f"\n{'#'*60}")
        print(f"TRAINING VERSION {version}")
        print(f"{'#'*60}")
        
        if version == 1:
            # First version - might already exist
            if os.path.exists("models/final_model.json"):
                config = NetworkConfig(input_size=39, hidden_sizes=[128, 64, 32])
                network = SimpleNeuralNetwork(config)
                network.load("models/final_model.json")
                trainer = Trainer(network)
                win_rate = trainer.test_against_random(num_games=50)
                save_version(1, win_rate)
            else:
                # Train from scratch
                win_rate = continue_training(0, 1, iterations_per_version)
        else:
            # Continue from previous version
            win_rate = continue_training(version-1, version, iterations_per_version)
        
        win_rates.append(win_rate)
        
        # Stop if we're not improving much
        if len(win_rates) > 1 and win_rates[-1] - win_rates[-2] < 0.02:
            print("\nStopping - improvement has plateaued")
            break
    
    # Plot progress if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(win_rates) + 1), [wr * 100 for wr in win_rates], 
                marker='o', linewidth=2, markersize=8)
        plt.xlabel('Version')
        plt.ylabel('Win Rate vs Random (%)')
        plt.title('Model Improvement Over Versions')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        for i, wr in enumerate(win_rates):
            plt.annotate(f'{wr*100:.1f}%', 
                        (i+1, wr*100), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        
        plt.savefig('models/version_progress.png')
        plt.show()
        print("\nProgress plot saved to models/version_progress.png")
    except ImportError:
        print("\nInstall matplotlib to see progress plots: pip install matplotlib")
    
    return win_rates


if __name__ == "__main__":
    import sys
    
    print("Extinction Chess - Continue Training")
    print("="*50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        # Train multiple versions
        print("Training multiple versions...")
        train_multiple_versions(num_versions=3, iterations_per_version=15)
    else:
        # Train single version 2
        print("Training version 2...")
        continue_training(
            start_version=1, 
            target_version=2, 
            additional_iterations=20
        )
    
    print("\nTraining complete!")
    print("Your models are saved in models/versions/")
    print("\nTo train more versions, run: python continue_training.py multi")