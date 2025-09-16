"""
Self-play training system for Extinction Chess
Generates games and trains the neural network
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json
import os

from extinction_chess import ExtinctionChess, Color, Move
from state_encoder import StateEncoder, MoveEncoder
from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator


@dataclass
class GameRecord:
    """Record of a single game for training"""
    positions: List[np.ndarray]  # Feature vectors for each position
    moves: List[int]  # Move indices played
    outcome: float  # 1 for white win, -1 for black win, 0 for draw
    
    def get_training_data(self):
        """Convert game record to training data"""
        training_data = []
        for i, position in enumerate(self.positions):
            # Adjust outcome based on whose turn it was
            # (positions from white's perspective get positive outcome for white win)
            perspective = 1 if i % 2 == 0 else -1
            adjusted_outcome = self.outcome * perspective
            training_data.append((position, adjusted_outcome))
        return training_data


class SelfPlayAgent:
    """Agent that plays games using the neural network"""
    
    def __init__(self, evaluator: PositionEvaluator, exploration_rate: float = 0.3):
        self.evaluator = evaluator
        self.exploration_rate = exploration_rate
        self.encoder = StateEncoder()
        self.move_encoder = MoveEncoder()
    
    def select_move(self, game: ExtinctionChess, temperature: float = 1.0) -> Move:
        """
        Select a move using the evaluator
        Temperature controls exploration vs exploitation
        """
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return None
        
        # Random move with some probability (exploration)
        if random.random() < self.exploration_rate:
            return random.choice(legal_moves)
        
        # Evaluate each move
        move_scores = []
        for move in legal_moves:
            # Make move on a copy
            game_copy = ExtinctionChess()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.game_over = game.game_over
            
            # Make the move and evaluate
            if game_copy.make_move(move):
                score = self.evaluator.evaluate(game_copy)
                # Negate score if black's turn (we want best move for current player)
                if game.current_player == Color.BLACK:
                    score = -score
                move_scores.append(score)
            else:
                move_scores.append(-2)  # Invalid move
        
        # Convert scores to probabilities
        scores = np.array(move_scores)
        
        if temperature == 0:
            # Deterministic: pick best move
            best_idx = np.argmax(scores)
            return legal_moves[best_idx]
        else:
            # Stochastic: sample based on scores
            # Convert scores to probabilities using softmax
            exp_scores = np.exp((scores - np.max(scores)) / temperature)
            probs = exp_scores / np.sum(exp_scores)
            
            # Sample move
            move_idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[move_idx]
    
    def play_game(self, max_moves: int = 200) -> GameRecord:
        """Play a complete self-play game"""
        game = ExtinctionChess()
        positions = []
        moves = []
        
        move_count = 0
        while not game.game_over and move_count < max_moves:
            # Record position
            features = self.encoder.get_simple_features(game)
            positions.append(features)
            
            # Select and play move
            # Use higher temperature early in the game for more exploration
            temperature = 1.0 if move_count < 20 else 0.5
            move = self.select_move(game, temperature)
            
            if move:
                game.make_move(move)
                moves.append(0)  # Simplified - just track that a move was made
                move_count += 1
            else:
                break
        
        # Determine outcome
        if game.winner == Color.WHITE:
            outcome = 1.0
        elif game.winner == Color.BLACK:
            outcome = -1.0
        else:
            outcome = 0.0
        
        return GameRecord(positions, moves, outcome)


class Trainer:
    """Manages the training process"""
    
    def __init__(self, network: SimpleNeuralNetwork, 
                 save_dir: str = "models",
                 log_file: str = "training_log.json"):
        self.network = network
        self.evaluator = PositionEvaluator(network)
        self.agent = SelfPlayAgent(self.evaluator)
        self.encoder = StateEncoder()
        self.save_dir = save_dir
        self.log_file = log_file
        self.training_history = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def generate_self_play_games(self, num_games: int = 10) -> List[GameRecord]:
        """Generate multiple self-play games"""
        games = []
        
        print(f"Generating {num_games} self-play games...")
        for i in range(num_games):
            if (i + 1) % 10 == 0:
                print(f"  Game {i + 1}/{num_games}")
            
            game_record = self.agent.play_game()
            games.append(game_record)
            
            # Print game outcome
            outcome_str = "White wins" if game_record.outcome > 0 else \
                         "Black wins" if game_record.outcome < 0 else "Draw"
            print(f"    Game {i + 1}: {len(game_record.positions)} moves, {outcome_str}")
        
        return games
    
    def train_on_games(self, games: List[GameRecord], epochs: int = 10) -> float:
        """Train network on game data"""
        # Collect all training data
        all_positions = []
        all_outcomes = []
        
        for game in games:
            training_data = game.get_training_data()
            for position, outcome in training_data:
                all_positions.append(position)
                all_outcomes.append(outcome)
        
        if not all_positions:
            return 0.0
        
        # Convert to numpy arrays
        X = np.array(all_positions)
        y = np.array(all_outcomes)
        
        print(f"Training on {len(X)} positions for {epochs} epochs...")
        
        # Train in mini-batches
        batch_size = 32
        total_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                loss = self.network.backward(batch_X, batch_y)
                epoch_loss += loss
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / max(1, len(X) // batch_size)
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        return total_loss / epochs if epochs > 0 else 0
    
    def training_loop(self, iterations: int = 10, games_per_iteration: int = 20):
        """Main training loop"""
        print(f"\nStarting training for {iterations} iterations")
        print(f"Games per iteration: {games_per_iteration}")
        print("-" * 50)
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            start_time = time.time()
            
            # Generate self-play games
            games = self.generate_self_play_games(games_per_iteration)
            
            # Calculate statistics
            wins_white = sum(1 for g in games if g.outcome > 0)
            wins_black = sum(1 for g in games if g.outcome < 0)
            draws = sum(1 for g in games if g.outcome == 0)
            avg_length = np.mean([len(g.positions) for g in games])
            
            print(f"\nGame statistics:")
            print(f"  White wins: {wins_white}/{games_per_iteration}")
            print(f"  Black wins: {wins_black}/{games_per_iteration}")
            print(f"  Draws: {draws}/{games_per_iteration}")
            print(f"  Avg game length: {avg_length:.1f} moves")
            
            # Train on games
            avg_loss = self.train_on_games(games, epochs=5)
            
            # Save model periodically
            if (iteration + 1) % 5 == 0:
                model_path = os.path.join(self.save_dir, f"model_iter_{iteration + 1}.json")
                self.network.save(model_path)
                print(f"Model saved to {model_path}")
            
            # Update agent with new network
            self.agent.evaluator = PositionEvaluator(self.network)
            
            # Reduce exploration over time
            self.agent.exploration_rate = max(0.1, 0.3 - 0.02 * iteration)
            
            # Log progress
            iteration_time = time.time() - start_time
            log_entry = {
                'iteration': iteration + 1,
                'avg_loss': float(avg_loss),
                'white_wins': wins_white,
                'black_wins': wins_black,
                'draws': draws,
                'avg_game_length': float(avg_length),
                'time': iteration_time,
                'exploration_rate': self.agent.exploration_rate
            }
            self.training_history.append(log_entry)
            
            print(f"Iteration time: {iteration_time:.1f} seconds")
            print(f"Current exploration rate: {self.agent.exploration_rate:.2f}")
        
        # Save final model and training history
        final_model_path = os.path.join(self.save_dir, "final_model.json")
        self.network.save(final_model_path)
        
        with open(self.log_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"Final model saved to {final_model_path}")
        print(f"Training log saved to {self.log_file}")
    
    def test_against_random(self, num_games: int = 100):
        """Test the trained network against a random player"""
        print(f"\nTesting against random player ({num_games} games)...")
        
        network_wins = 0
        random_wins = 0
        draws = 0
        
        for game_num in range(num_games):
            game = ExtinctionChess()
            network_plays_white = game_num % 2 == 0
            
            move_count = 0
            while not game.game_over and move_count < 200:
                is_network_turn = (game.current_player == Color.WHITE) == network_plays_white
                
                if is_network_turn:
                    # Network move
                    move = self.agent.select_move(game, temperature=0)  # Deterministic
                else:
                    # Random move
                    legal_moves = game.get_legal_moves()
                    move = random.choice(legal_moves) if legal_moves else None
                
                if move:
                    game.make_move(move)
                    move_count += 1
                else:
                    break
            
            # Check outcome
            if game.winner:
                if (game.winner == Color.WHITE) == network_plays_white:
                    network_wins += 1
                else:
                    random_wins += 1
            else:
                draws += 1
        
        print(f"Results:")
        print(f"  Network wins: {network_wins}/{num_games} ({100*network_wins/num_games:.1f}%)")
        print(f"  Random wins: {random_wins}/{num_games} ({100*random_wins/num_games:.1f}%)")
        print(f"  Draws: {draws}/{num_games} ({100*draws/num_games:.1f}%)")
        
        return network_wins / num_games


# Main training script
if __name__ == "__main__":
    print("Extinction Chess Self-Play Training")
    print("=" * 50)
    
    # Create network
    config = NetworkConfig(
        input_size=39,  # From StateEncoder.get_simple_features()
        hidden_sizes=[128, 64, 32],  # Small network for CPU
        learning_rate=0.001
    )
    network = SimpleNeuralNetwork(config)
    
    # Create trainer
    trainer = Trainer(network)
    
    # Run training
    trainer.training_loop(
        iterations=10,  # Number of training iterations
        games_per_iteration=20  # Games to generate each iteration
    )
    
    # Test against random player
    win_rate = trainer.test_against_random(num_games=50)
    
    print(f"\nFinal win rate vs random: {win_rate:.1%}")
    print("\nTraining complete! The model should now play better than random.")