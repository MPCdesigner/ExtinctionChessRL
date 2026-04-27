"""
Simple neural network evaluator for Extinction Chess
CPU-friendly architecture for initial experiments
"""

import numpy as np
import json
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Configuration for the neural network"""
    input_size: int = 39  # Size of feature vector
    hidden_sizes: List[int] = None  # Hidden layer sizes
    learning_rate: float = 0.001
    activation: str = 'relu'
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]  # Small network for CPU


class SimpleNeuralNetwork:
    """
    Simple feedforward neural network for position evaluation
    Implements forward pass and backpropagation from scratch
    (No PyTorch/TensorFlow needed for initial experiments)
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.layers = []
        self.biases = []
        
        # Initialize network layers
        layer_sizes = [config.input_size] + config.hidden_sizes + [1]  # Output is single value
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i+1]))
            self.layers.append(weight)
            self.biases.append(bias)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    def forward(self, x):
        """
        Forward pass through the network
        x: input features (batch_size, input_size)
        Returns: evaluation score between -1 and 1
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        self.activations = [x]
        
        # Hidden layers with ReLU
        for i in range(len(self.layers) - 1):
            z = np.dot(self.activations[-1], self.layers[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
        
        # Output layer with tanh (bounded between -1 and 1)
        z = np.dot(self.activations[-1], self.layers[-1]) + self.biases[-1]
        output = self.tanh(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, x, y_true, learning_rate=None):
        """
        Backpropagation to update weights
        x: input features
        y_true: true evaluation scores
        """
        if learning_rate is None:
            learning_rate = self.config.learning_rate
        
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        batch_size = x.shape[0]
        
        # Forward pass
        y_pred = self.forward(x)
        
        # Calculate loss (MSE)
        loss = np.mean((y_pred - y_true) ** 2)
        
        # Backward pass
        delta = 2 * (y_pred - y_true) / batch_size
        delta = delta * self.tanh_derivative(
            np.dot(self.activations[-2], self.layers[-1]) + self.biases[-1]
        )
        
        # Store gradients
        weight_gradients = []
        bias_gradients = []
        
        # Output layer gradients
        weight_gradients.append(np.dot(self.activations[-2].T, delta))
        bias_gradients.append(np.sum(delta, axis=0, keepdims=True))
        
        # Hidden layers gradients
        for i in range(len(self.layers) - 2, -1, -1):
            delta = np.dot(delta, self.layers[i+1].T)
            delta = delta * self.relu_derivative(
                np.dot(self.activations[i], self.layers[i]) + self.biases[i]
            )
            weight_gradients.append(np.dot(self.activations[i].T, delta))
            bias_gradients.append(np.sum(delta, axis=0, keepdims=True))
        
        # Update weights and biases
        weight_gradients.reverse()
        bias_gradients.reverse()
        
        for i in range(len(self.layers)):
            self.layers[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
        
        return loss
    
    def evaluate_position(self, features):
        """
        Evaluate a chess position
        features: feature vector from StateEncoder.get_simple_features()
        Returns: evaluation score between -1 (black winning) and 1 (white winning)
        """
        return self.forward(features).item()
    
    def train_batch(self, features_batch, outcomes_batch, epochs=10):
        """
        Train on a batch of positions
        features_batch: array of feature vectors
        outcomes_batch: array of game outcomes (1 for white win, -1 for black win, 0 for draw)
        """
        losses = []
        for epoch in range(epochs):
            loss = self.backward(features_batch, outcomes_batch)
            losses.append(loss)
        return losses
    
    def save(self, filepath):
        """Save network weights to file"""
        data = {
            'config': {
                'input_size': self.config.input_size,
                'hidden_sizes': self.config.hidden_sizes,
                'learning_rate': self.config.learning_rate,
                'activation': self.config.activation
            },
            'weights': [w.tolist() for w in self.layers],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load network weights from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = NetworkConfig(**data['config'])
        self.layers = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
        print(f"Model loaded from {filepath}")


class PositionEvaluator:
    """
    Combines neural network with simple heuristics for position evaluation
    """
    
    def __init__(self, network: Optional[SimpleNeuralNetwork] = None):
        self.network = network
    
    def material_balance(self, game):
        """Calculate material balance (simple heuristic)"""
        from extinction_chess import Color, PieceType
        
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 4  # Kings can be captured in extinction chess
        }
        
        white_material = 0
        black_material = 0
        
        white_counts = game.board.get_piece_count(Color.WHITE)
        black_counts = game.board.get_piece_count(Color.BLACK)
        
        for piece_type, value in piece_values.items():
            white_material += white_counts[piece_type] * value
            black_material += black_counts[piece_type] * value
        
        # Normalize to [-1, 1]
        total = white_material + black_material
        if total == 0:
            return 0
        return (white_material - black_material) / total
    
    def extinction_danger(self, game):
        """Evaluate danger of extinction"""
        from extinction_chess import Color
        
        white_endangered = len(game.get_endangered_pieces(Color.WHITE))
        black_endangered = len(game.get_endangered_pieces(Color.BLACK))
        
        # More endangered pieces is bad
        danger_score = (black_endangered - white_endangered) / 6.0
        return np.clip(danger_score, -1, 1)
    
    def evaluate(self, game, use_network=True):
        """
        Evaluate a position
        Returns: score from white's perspective (-1 to 1)
        """
        from extinction_chess import Color
        
        # Check for game over
        if game.game_over:
            if game.winner == Color.WHITE:
                return 1.0
            elif game.winner == Color.BLACK:
                return -1.0
            else:
                return 0.0  # Draw
        
        if use_network and self.network is not None:
            # Use neural network evaluation
            from state_encoder import StateEncoder
            encoder = StateEncoder()
            features = encoder.get_simple_features(game)
            return self.network.evaluate_position(features)
        else:
            # Use simple heuristics
            material = self.material_balance(game)
            danger = self.extinction_danger(game)
            
            # Weighted combination
            return 0.7 * material + 0.3 * danger


# Example usage and testing
if __name__ == "__main__":
    from extinction_chess import ExtinctionChess
    from state_encoder import StateEncoder
    
    # Create game and encoder
    game = ExtinctionChess()
    encoder = StateEncoder()
    
    # Create and test network
    config = NetworkConfig()
    network = SimpleNeuralNetwork(config)
    
    # Get features for current position
    features = encoder.get_simple_features(game)
    print(f"Feature vector shape: {features.shape}")
    
    # Evaluate position
    score = network.evaluate_position(features)
    print(f"Position evaluation: {score:.3f}")
    
    # Create evaluator with heuristics
    evaluator = PositionEvaluator(network)
    
    # Compare network vs heuristic evaluation
    network_eval = evaluator.evaluate(game, use_network=True)
    heuristic_eval = evaluator.evaluate(game, use_network=False)
    
    print(f"\nNetwork evaluation: {network_eval:.3f}")
    print(f"Heuristic evaluation: {heuristic_eval:.3f}")
    
    # Simulate training
    print("\nSimulating training...")
    
    # Generate fake training data (normally from self-play)
    batch_size = 32
    features_batch = np.random.randn(batch_size, config.input_size).astype(np.float32)
    outcomes_batch = np.random.choice([-1, 0, 1], size=batch_size).astype(np.float32)
    
    # Train for a few epochs
    losses = network.train_batch(features_batch, outcomes_batch, epochs=5)
    print(f"Training losses: {losses}")
    
    # Save model
    network.save("test_model.json")
    
    # Load model
    new_network = SimpleNeuralNetwork(config)
    new_network.load("test_model.json")
    
    print("\nModel saved and loaded successfully!")