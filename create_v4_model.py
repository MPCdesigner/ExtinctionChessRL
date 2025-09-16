"""
Create v4 model by modifying evaluation weights
Keeps same neural network but changes how we combine heuristics
"""

import os
import shutil
import json
import numpy as np
from extinction_chess import ExtinctionChess, Color, PieceType, Position
from typing import Optional

from extinction_chess import ExtinctionChess, Color, PieceType
from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator
from state_encoder import StateEncoder
from self_play_trainer import SelfPlayAgent, Trainer


class ExtinctionFocusedEvaluator(PositionEvaluator):
    """
    Modified evaluator that properly weights extinction danger
    Uses same features but different combination
    """
    
    def __init__(self, network: SimpleNeuralNetwork):
        super().__init__(network)
        self.encoder = StateEncoder()
    
    def evaluate(self, game: ExtinctionChess, use_network: bool = True) -> float:
        """
        Evaluation that respects Extinction Chess rules
        Extinction danger >> Material advantage
        """
        # Terminal positions
        if game.game_over:
            if game.winner == Color.WHITE:
                return 1.0
            elif game.winner == Color.BLACK:
                return -1.0
            else:
                return 0.0
        
        # Get piece counts
        white_counts = game.board.get_piece_count(Color.WHITE)
        black_counts = game.board.get_piece_count(Color.BLACK)
        
        # CRITICAL: Check for immediate extinction threat
        white_endangered = [pt for pt in PieceType if white_counts[pt] == 1]
        black_endangered = [pt for pt in PieceType if black_counts[pt] == 1]
        
        # Extinction danger score (MOST IMPORTANT)
        extinction_score = 0.0
        
        # Heavy penalty for having endangered pieces
        extinction_score -= len(white_endangered) * 0.3
        extinction_score += len(black_endangered) * 0.3
        
        # Check if we can capture an endangered piece next move
        can_extinct_opponent = False
        for move in game.get_legal_moves()[:30]:  # Check first 30 moves
            target = game.board.get_piece(move.to_pos)
            if target:
                enemy_color = Color.BLACK if game.current_player == Color.WHITE else Color.WHITE
                enemy_counts = game.board.get_piece_count(enemy_color)
                if enemy_counts[target.piece_type] == 1:
                    can_extinct_opponent = True
                    break
        
        if can_extinct_opponent:
            extinction_score += 0.5 if game.current_player == Color.WHITE else -0.5
        
        # Material score (LESS IMPORTANT than extinction)
        material_score = 0.0
        if use_network:
            # Use neural network evaluation
            features = self.encoder.get_simple_features(game)
            network_eval = self.network.evaluate_position(features)
            material_score = network_eval * 0.3  # Reduce network's influence
        else:
            # Simple material count (but NO hardcoded values)
            white_total = sum(white_counts.values())
            black_total = sum(black_counts.values())
            material_score = (white_total - black_total) / 32.0
        
        # EXTINCTION-FOCUSED COMBINATION
        # 70% extinction danger, 30% other factors
        final_score = 0.7 * extinction_score + 0.3 * material_score
        
        # Clamp to valid range
        return np.clip(final_score, -1.0, 1.0)


def create_v4_model():
    """
    Load v3 enhanced model and test with better evaluation weights
    """
    print("Creating v4 Model - Modified Evaluation Weights")
    print("="*60)
    
    # Load the enhanced model (v3)
    if not os.path.exists("models/enhanced_model.json"):
        print("Enhanced model not found! Run enhanced_trainer.py first.")
        return
    
    # Load v3 model
    config = NetworkConfig(input_size=39, hidden_sizes=[128, 64, 32])
    network = SimpleNeuralNetwork(config)
    network.load("models/enhanced_model.json")
    print("Loaded v3 enhanced model")
    
    # Create evaluators for comparison
    old_evaluator = PositionEvaluator(network)  # Original 70% material, 30% extinction
    new_evaluator = ExtinctionFocusedEvaluator(network)  # New 30% material, 70% extinction
    
    print("\nTesting evaluation differences...")
    test_evaluation_differences(old_evaluator, new_evaluator)
    
    # Test both versions against random
    print("\nTesting v3 (material-focused) vs random...")
    old_agent = SelfPlayAgent(old_evaluator, exploration_rate=0)
    v3_win_rate = test_against_random(old_agent, num_games=100)
    
    print(f"\nTesting v4 (extinction-focused) vs random...")
    new_agent = SelfPlayAgent(new_evaluator, exploration_rate=0)
    v4_win_rate = test_against_random(new_agent, num_games=100)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"v3 (material-focused): {v3_win_rate:.1%} vs random")
    print(f"v4 (extinction-focused): {v4_win_rate:.1%} vs random")
    
    # The network is the same, but we save it with metadata about the evaluator
    metadata = {
        "version": 4,
        "base_model": "v3_enhanced",
        "evaluator": "extinction_focused",
        "weights": "70% extinction, 30% material",
        "expected_win_rate": float(v4_win_rate)
    }
    
    # Save v4 (same network, different evaluator metadata)
    v4_path = f"models/versions/model_v4_extinction_focused_{int(v4_win_rate*100)}pct.json"
    shutil.copy("models/enhanced_model.json", v4_path)
    
    metadata_path = "models/versions/metadata_v4.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nv4 model saved to {v4_path}")
    print("Note: v4 uses the same network as v3 but different evaluation weights")
    print("\nTo play against v4, use the ExtinctionFocusedEvaluator class")
    
    return new_evaluator


def test_evaluation_differences(old_eval, new_eval):
    """
    Show how the evaluators differ on key positions
    """
    game = ExtinctionChess()
    
    # Test 1: Starting position
    score_old = old_eval.evaluate(game)
    score_new = new_eval.evaluate(game)
    print(f"\nStarting position:")
    print(f"  v3 evaluation: {score_old:.3f}")
    print(f"  v4 evaluation: {score_new:.3f}")
    
    # Test 2: Create position with endangered piece
    # Remove most white knights
    game.board.set_piece(Position(0, 1), None)  # Remove one white knight
    
    score_old = old_eval.evaluate(game)
    score_new = new_eval.evaluate(game)
    print(f"\nWhite missing a knight (1 left):")
    print(f"  v3 evaluation: {score_old:.3f}")
    print(f"  v4 evaluation: {score_new:.3f}")
    print(f"  v4 should show more concern about the endangered knight")


def test_against_random(agent, num_games=100):
    """Test an agent against random player"""
    import random
    from extinction_chess import ExtinctionChess, Color
    
    wins = 0
    for i in range(num_games):
        game = ExtinctionChess()
        agent_plays_white = i % 2 == 0
        
        moves = 0
        while not game.game_over and moves < 200:
            if (game.current_player == Color.WHITE) == agent_plays_white:
                # Agent move
                move = agent.select_move(game, temperature=0)
            else:
                # Random move
                legal_moves = game.get_legal_moves()
                move = random.choice(legal_moves) if legal_moves else None
            
            if move:
                game.make_move(move)
                moves += 1
            else:
                break
        
        if game.winner:
            if (game.winner == Color.WHITE) == agent_plays_white:
                wins += 1
    
    return wins / num_games


def play_v3_vs_v4():
    """
    Have v3 and v4 play against each other
    """
    print("\nv3 vs v4 Head-to-Head (10 games)...")
    
    config = NetworkConfig(input_size=39, hidden_sizes=[128, 64, 32])
    network = SimpleNeuralNetwork(config)
    network.load("models/enhanced_model.json")
    
    v3_evaluator = PositionEvaluator(network)
    v4_evaluator = ExtinctionFocusedEvaluator(network)
    
    v3_agent = SelfPlayAgent(v3_evaluator, exploration_rate=0)
    v4_agent = SelfPlayAgent(v4_evaluator, exploration_rate=0)
    
    v3_wins = 0
    v4_wins = 0
    
    for game_num in range(10):
        game = ExtinctionChess()
        v3_plays_white = game_num % 2 == 0
        
        moves = 0
        while not game.game_over and moves < 200:
            if (game.current_player == Color.WHITE) == v3_plays_white:
                move = v3_agent.select_move(game, temperature=0)
            else:
                move = v4_agent.select_move(game, temperature=0)
            
            if move:
                game.make_move(move)
                moves += 1
            else:
                break
        
        if game.winner:
            if (game.winner == Color.WHITE) == v3_plays_white:
                v3_wins += 1
            else:
                v4_wins += 1
    
    print(f"Results: v3 won {v3_wins}, v4 won {v4_wins}")
    print("v4 should perform better due to extinction-focused evaluation")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        play_v3_vs_v4()
    else:
        create_v4_model()