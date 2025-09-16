"""
Enhanced training system with lookahead search and better data generation
"""

import numpy as np
import random
import time
import json
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

from extinction_chess import ExtinctionChess, Color, Move
from state_encoder import StateEncoder
from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator
from self_play_trainer import GameRecord, Trainer


class MinimaxAgent:
    """Agent that uses minimax search with neural network evaluation"""
    
    def __init__(self, evaluator: PositionEvaluator, max_depth: int = 2):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.encoder = StateEncoder()
        self.nodes_evaluated = 0
    
    def minimax(self, game: ExtinctionChess, depth: int, alpha: float, beta: float, 
                maximizing: bool) -> float:
        """Minimax with alpha-beta pruning"""
        self.nodes_evaluated += 1
        
        # Terminal node or depth limit
        if game.game_over or depth == 0:
            eval_score = self.evaluator.evaluate(game)
            # Return from current player's perspective
            return eval_score if game.current_player == Color.WHITE else -eval_score
        
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return 0  # Stalemate
        
        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                # Make move on copy
                game_copy = ExtinctionChess()
                game_copy.board = game.board.copy()
                game_copy.current_player = game.current_player
                game_copy.game_over = game.game_over
                
                if game_copy.make_move(move):
                    eval_score = self.minimax(game_copy, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                # Make move on copy
                game_copy = ExtinctionChess()
                game_copy.board = game.board.copy()
                game_copy.current_player = game.current_player
                game_copy.game_over = game.game_over
                
                if game_copy.make_move(move):
                    eval_score = self.minimax(game_copy, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
            return min_eval
    
    def select_move(self, game: ExtinctionChess, time_limit: float = 1.0) -> Move:
        """Select best move using minimax search"""
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        # For early game, add some randomness
        if game.board.fullmove_number <= 3 and random.random() < 0.3:
            return random.choice(legal_moves)
        
        self.nodes_evaluated = 0
        start_time = time.time()
        
        best_move = None
        best_score = -float('inf')
        
        # Try each move
        for move in legal_moves:
            # Time check
            if time.time() - start_time > time_limit:
                break
            
            # Evaluate move
            game_copy = ExtinctionChess()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.game_over = game.game_over
            
            if game_copy.make_move(move):
                # Evaluate position after move
                if game_copy.game_over:
                    # Immediate win/loss
                    if game_copy.winner == game.current_player:
                        return move  # Winning move!
                    else:
                        score = -1000  # Losing move
                else:
                    # Use minimax for evaluation
                    score = self.minimax(game_copy, self.max_depth - 1, 
                                       -float('inf'), float('inf'), False)
                
                # Negate score if black's turn
                if game.current_player == Color.BLACK:
                    score = -score
                
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move if best_move else legal_moves[0]
    
    def play_game(self, max_moves: int = 200) -> GameRecord:
        """Play a complete game using minimax"""
        game = ExtinctionChess()
        positions = []
        moves = []
        
        move_count = 0
        while not game.game_over and move_count < max_moves:
            # Record position
            features = self.encoder.get_simple_features(game)
            positions.append(features)
            
            # Select move with minimax
            move = self.select_move(game, time_limit=0.5)
            
            if move:
                game.make_move(move)
                moves.append(0)
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


class EnhancedTrainer(Trainer):
    """Enhanced trainer with better data generation and testing"""
    
    def __init__(self, network: SimpleNeuralNetwork, save_dir: str = "models"):
        super().__init__(network, save_dir)
        self.minimax_agent = MinimaxAgent(self.evaluator, max_depth=2)
    
    def generate_mixed_games(self, num_games: int = 30) -> List[GameRecord]:
        """Generate games using both random and minimax agents"""
        games = []
        
        print(f"Generating {num_games} training games...")
        
        # Mix of different game types
        for i in range(num_games):
            if i % 3 == 0:
                # Pure self-play with current network
                game_record = self.agent.play_game()
                game_type = "self-play"
            elif i % 3 == 1:
                # Minimax vs minimax (higher quality games)
                game_record = self.minimax_agent.play_game()
                game_type = "minimax"
            else:
                # Mixed random/evaluation
                self.agent.exploration_rate = 0.5  # More exploration
                game_record = self.agent.play_game()
                self.agent.exploration_rate = 0.2  # Reset
                game_type = "exploratory"
            
            games.append(game_record)
            
            # Print game info
            outcome_str = "White wins" if game_record.outcome > 0 else \
                         "Black wins" if game_record.outcome < 0 else "Draw"
            print(f"  Game {i+1} ({game_type}): {len(game_record.positions)} moves, {outcome_str}")
        
        return games
    
    def enhanced_training_loop(self, iterations: int = 30, 
                              games_per_iteration: int = 40,
                              test_games: int = 200):
        """Enhanced training with more data and better testing"""
        
        print(f"\nENHANCED TRAINING")
        print(f"Iterations: {iterations}")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Test games: {test_games}")
        print("-" * 60)
        
        best_win_rate = 0
        best_model_iter = 0
        
        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{iterations}")
            start_time = time.time()
            
            # Generate diverse training games
            games = self.generate_mixed_games(games_per_iteration)
            
            # Statistics
            wins_white = sum(1 for g in games if g.outcome > 0)
            wins_black = sum(1 for g in games if g.outcome < 0)
            draws = sum(1 for g in games if g.outcome == 0)
            avg_length = np.mean([len(g.positions) for g in games])
            
            print(f"\nGame statistics:")
            print(f"  White wins: {wins_white}/{games_per_iteration}")
            print(f"  Black wins: {wins_black}/{games_per_iteration}")
            print(f"  Draws: {draws}/{games_per_iteration}")
            print(f"  Avg game length: {avg_length:.1f} moves")
            
            # Train with more epochs on accumulated data
            print("\nTraining network...")
            avg_loss = self.train_on_games(games, epochs=10)
            print(f"Average loss: {avg_loss:.4f}")
            
            # Test periodically with more games
            if (iteration + 1) % 5 == 0:
                print(f"\nTesting performance ({test_games} games)...")
                win_rate = self.test_against_random(test_games)
                
                # Save if best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_model_iter = iteration + 1
                    model_path = os.path.join(self.save_dir, f"best_model_{int(win_rate*100)}pct.json")
                    self.network.save(model_path)
                    print(f"New best model! Saved to {model_path}")
                
                # Also test against minimax
                print("\nTesting against minimax (10 games)...")
                minimax_wins = self.test_against_minimax(num_games=10)
                print(f"Win rate vs minimax: {minimax_wins:.1%}")
            
            # Update minimax agent with new network
            self.minimax_agent.evaluator = PositionEvaluator(self.network)
            
            # Reduce exploration over time
            self.agent.exploration_rate = max(0.05, 0.3 - 0.01 * iteration)
            
            iteration_time = time.time() - start_time
            print(f"\nIteration time: {iteration_time:.1f} seconds")
        
        # Final comprehensive test
        print("\n" + "="*60)
        print("FINAL TESTING")
        print("="*60)
        
        print(f"\nFinal test against random ({test_games} games)...")
        final_win_rate = self.test_against_random(test_games)
        
        print(f"\nTraining complete!")
        print(f"Best model: Iteration {best_model_iter} with {best_win_rate:.1%} win rate")
        print(f"Final model: {final_win_rate:.1%} win rate")
    
    def test_against_minimax(self, num_games: int = 20) -> float:
        """Test current network against minimax player"""
        wins = 0
        
        for game_num in range(num_games):
            game = ExtinctionChess()
            network_plays_white = game_num % 2 == 0
            
            move_count = 0
            while not game.game_over and move_count < 200:
                is_network_turn = (game.current_player == Color.WHITE) == network_plays_white
                
                if is_network_turn:
                    # Network move (no lookahead)
                    move = self.agent.select_move(game, temperature=0)
                else:
                    # Minimax move (with lookahead)
                    move = self.minimax_agent.select_move(game, time_limit=0.5)
                
                if move:
                    game.make_move(move)
                    move_count += 1
                else:
                    break
            
            # Check outcome
            if game.winner:
                if (game.winner == Color.WHITE) == network_plays_white:
                    wins += 1
        
        return wins / num_games


def continue_enhanced_training():
    """Continue training with enhanced methods"""
    
    print("ENHANCED EXTINCTION CHESS TRAINING")
    print("="*60)
    
    # Load existing model or create new
    config = NetworkConfig(
        input_size=39,
        hidden_sizes=[128, 64, 32],
        learning_rate=0.0005  # Lower learning rate for fine-tuning
    )
    
    network = SimpleNeuralNetwork(config)
    
    # Try to load best existing model
    if os.path.exists("models/versions/model_v1_94pct.json"):
        network.load("models/versions/model_v1_94pct.json")
        print("Loaded existing model v1")
    elif os.path.exists("models/final_model.json"):
        network.load("models/final_model.json")
        print("Loaded final model")
    else:
        print("Starting with fresh network")
    
    # Create enhanced trainer
    trainer = EnhancedTrainer(network)
    
    # Run enhanced training
    trainer.enhanced_training_loop(
        iterations=20,           # More iterations
        games_per_iteration=50,  # More games per iteration
        test_games=200          # More thorough testing
    )
    
    # Save final enhanced model
    network.save("models/enhanced_model.json")
    print("\nEnhanced model saved to models/enhanced_model.json")
    
    # Save version
    import shutil
    win_rate = trainer.test_against_random(100)
    version_name = f"models/versions/model_v3_enhanced_{int(win_rate*100)}pct.json"
    shutil.copy("models/enhanced_model.json", version_name)
    print(f"Saved as {version_name}")


if __name__ == "__main__":
    continue_enhanced_training()