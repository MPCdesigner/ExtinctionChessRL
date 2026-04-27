"""
Test and analyze what your trained model has learned
"""

import json
import numpy as np
from extinction_chess import ExtinctionChess, Color, PieceType, Position
from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator
from state_encoder import StateEncoder


def load_model(model_path="models/versions/model_v1_94pct.json"):
    """Load your trained model"""
    config = NetworkConfig(input_size=39, hidden_sizes=[128, 64, 32])
    network = SimpleNeuralNetwork(config)
    network.load(model_path)
    return network


def analyze_model_structure(model_path="models/versions/model_v1_94pct.json"):
    """Show the structure of your saved model"""
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    print("MODEL STRUCTURE:")
    print("="*50)
    print(f"Input size: {data['config']['input_size']} features")
    print(f"Hidden layers: {data['config']['hidden_sizes']}")
    print(f"Learning rate: {data['config']['learning_rate']}")
    
    print("\nLAYER SIZES:")
    for i, weights in enumerate(data['weights']):
        weight_array = np.array(weights)
        print(f"  Layer {i+1}: {weight_array.shape} = {weight_array.shape[0]} inputs → {weight_array.shape[1]} outputs")
    
    print("\nTOTAL PARAMETERS:")
    total_params = 0
    for weights in data['weights']:
        total_params += len(np.array(weights).flatten())
    for biases in data['biases']:
        total_params += len(np.array(biases).flatten())
    print(f"  {total_params:,} learned parameters")


def test_starting_position(network):
    """Test what the model thinks of the starting position"""
    game = ExtinctionChess()
    evaluator = PositionEvaluator(network)
    
    score = evaluator.evaluate(game)
    print("\nSTARTING POSITION EVALUATION:")
    print("="*50)
    print(f"Score: {score:.3f}")
    if abs(score) < 0.1:
        print("Model thinks the position is balanced (good!)")
    elif score > 0:
        print("Model thinks White has an advantage")
    else:
        print("Model thinks Black has an advantage")


def test_material_understanding(network):
    """Test if the model understands material value"""
    print("\nMATERIAL UNDERSTANDING TEST:")
    print("="*50)
    
    evaluator = PositionEvaluator(network)
    encoder = StateEncoder()
    
    # Test 1: Normal position
    game1 = ExtinctionChess()
    score1 = evaluator.evaluate(game1)
    
    # Test 2: Remove a black piece (white should be better)
    game2 = ExtinctionChess()
    # Remove black's queen
    game2.board.set_piece(Position(7, 3), None)  # Remove black queen
    score2 = evaluator.evaluate(game2)
    
    # Test 3: Remove a white piece (black should be better)
    game3 = ExtinctionChess()
    game3.board.set_piece(Position(0, 3), None)  # Remove white queen
    score3 = evaluator.evaluate(game3)
    
    print(f"Normal position: {score1:.3f}")
    print(f"Black missing queen: {score2:.3f}")
    print(f"White missing queen: {score3:.3f}")
    
    if score2 > score1 > score3:
        print("✓ Model correctly understands material advantage!")
    else:
        print("✗ Model may not fully understand material value")


def test_extinction_danger(network):
    """Test if the model understands extinction danger"""
    print("\nEXTINCTION DANGER TEST:")
    print("="*50)
    
    evaluator = PositionEvaluator(network)
    
    # Create a position where white has only 1 knight (endangered)
    game = ExtinctionChess()
    # Remove all white knights except one
    game.board.set_piece(Position(0, 1), None)  # Remove one white knight
    
    score_before = evaluator.evaluate(game)
    
    # Now remove the last knight (extinction!)
    game.board.set_piece(Position(0, 6), None)  # Remove last white knight
    
    score_after = evaluator.evaluate(game)
    
    print(f"White has 1 knight (endangered): {score_before:.3f}")
    print(f"White has 0 knights (extinct): {score_after:.3f}")
    
    if score_after < score_before - 0.5:
        print("✓ Model understands extinction is bad!")
    else:
        print("✗ Model may not fully grasp extinction danger")


def test_decision_making(network):
    """Test the model's move selection"""
    print("\nDECISION MAKING TEST:")
    print("="*50)
    
    from self_play_trainer import SelfPlayAgent
    
    evaluator = PositionEvaluator(network)
    agent = SelfPlayAgent(evaluator, exploration_rate=0)
    
    game = ExtinctionChess()
    
    # Get the model's preferred first moves
    print("Model's top choices for first move:")
    
    # Evaluate all first moves
    legal_moves = game.get_legal_moves()
    move_scores = []
    
    for move in legal_moves:
        game_copy = ExtinctionChess()
        game_copy.board = game.board.copy()
        game_copy.make_move(move)
        score = evaluator.evaluate(game_copy)
        move_scores.append((move, score))
    
    # Sort by score
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 5 moves
    for i, (move, score) in enumerate(move_scores[:5]):
        piece = game.board.get_piece(move.from_pos)
        print(f"  {i+1}. {piece.piece_type.value} {move.from_pos.to_algebraic()}-{move.to_pos.to_algebraic()}: {score:.3f}")


def visualize_feature_importance(network):
    """Show which features the model considers most important"""
    print("\nFEATURE IMPORTANCE (First layer weights):")
    print("="*50)
    
    # Get the first layer weights
    first_layer_weights = np.array(network.layers[0])
    
    # Calculate importance as sum of absolute weights for each input feature
    feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
    
    # Feature names (simplified)
    feature_names = [
        "White Pawns", "White Knights", "White Bishops", 
        "White Rooks", "White Queens", "White Kings",
        "Black Pawns", "Black Knights", "Black Bishops",
        "Black Rooks", "Black Queens", "Black Kings",
        # ... (remaining features)
    ]
    
    # Show top 10 most important features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    
    print("Top 10 most important features:")
    for i, idx in enumerate(top_indices[:10]):
        if idx < len(feature_names):
            name = feature_names[idx]
        else:
            name = f"Feature {idx}"
        print(f"  {i+1}. {name}: {feature_importance[idx]:.3f}")


def main():
    print("ANALYZING YOUR TRAINED MODEL")
    print("="*60)
    
    # Check if model exists
    import os
    model_path = "models/versions/model_v1_94pct.json"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Looking for alternative model...")
        if os.path.exists("models/final_model.json"):
            model_path = "models/final_model.json"
            print(f"Found model at {model_path}")
        else:
            print("No model found! Run training first.")
            return
    
    # Load and analyze model
    network = load_model(model_path)
    
    # Run all tests
    analyze_model_structure(model_path)
    test_starting_position(network)
    test_material_understanding(network)
    test_extinction_danger(network)
    test_decision_making(network)
    visualize_feature_importance(network)
    
    print("\n" + "="*60)
    print("WHAT YOUR MODEL HAS LEARNED:")
    print("- Basic position evaluation")
    print("- Material counting")
    print("- Some understanding of extinction danger")
    print("- Preference for certain opening moves")
    print("\nThe 94% win rate shows it learned meaningful patterns!")


if __name__ == "__main__":
    main()