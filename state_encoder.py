"""
State encoder for Extinction Chess
Converts board positions to tensors for neural network input
"""

import numpy as np
from typing import List, Tuple
from extinction_chess import (
    ExtinctionChess, Board, Position, Piece, 
    PieceType, Color
)


class StateEncoder:
    """Encodes chess board state into tensor format for neural network"""
    
    def __init__(self):
        # We'll use a simple encoding:
        # 12 channels for pieces (6 piece types × 2 colors)
        # 1 channel for current player
        # 1 channel for endangered pieces
        # Total: 14 channels × 8 × 8
        self.num_channels = 14
        self.board_size = 8
        
        # Piece type to channel index mapping
        self.piece_channels = {
            (PieceType.PAWN, Color.WHITE): 0,
            (PieceType.KNIGHT, Color.WHITE): 1,
            (PieceType.BISHOP, Color.WHITE): 2,
            (PieceType.ROOK, Color.WHITE): 3,
            (PieceType.QUEEN, Color.WHITE): 4,
            (PieceType.KING, Color.WHITE): 5,
            (PieceType.PAWN, Color.BLACK): 6,
            (PieceType.KNIGHT, Color.BLACK): 7,
            (PieceType.BISHOP, Color.BLACK): 8,
            (PieceType.ROOK, Color.BLACK): 9,
            (PieceType.QUEEN, Color.BLACK): 10,
            (PieceType.KING, Color.BLACK): 11,
        }
    
    def encode_board(self, game: ExtinctionChess) -> np.ndarray:
        """
        Encode the current board state as a tensor
        Returns: numpy array of shape (14, 8, 8)
        """
        tensor = np.zeros((self.num_channels, self.board_size, self.board_size), 
                         dtype=np.float32)
        
        # Encode pieces
        for rank in range(8):
            for file in range(8):
                pos = Position(rank, file)
                piece = game.board.get_piece(pos)
                if piece:
                    channel = self.piece_channels[(piece.piece_type, piece.color)]
                    tensor[channel, rank, file] = 1.0
        
        # Channel 12: Current player (1 for white's turn, 0 for black's turn)
        if game.current_player == Color.WHITE:
            tensor[12, :, :] = 1.0
        
        # Channel 13: Endangered pieces (pieces with only 1 left)
        white_endangered = game.get_endangered_pieces(Color.WHITE)
        black_endangered = game.get_endangered_pieces(Color.BLACK)
        
        for rank in range(8):
            for file in range(8):
                pos = Position(rank, file)
                piece = game.board.get_piece(pos)
                if piece:
                    if ((piece.color == Color.WHITE and piece.piece_type in white_endangered) or
                        (piece.color == Color.BLACK and piece.piece_type in black_endangered)):
                        tensor[13, rank, file] = 1.0
        
        return tensor
    
    def encode_move(self, from_pos: Position, to_pos: Position, 
                   promotion: PieceType = None) -> int:
        """
        Encode a move as an integer action
        8×8×73 possible moves (like AlphaZero):
        - 56 queen moves (8 directions × 7 squares)
        - 8 knight moves
        - 9 underpromotions (3 pieces × 3 directions)
        But for simplicity, we'll use: from_square × 64 + to_square
        """
        action = from_pos.rank * 64 + from_pos.file * 8 + to_pos.rank * 8 + to_pos.file
        if promotion:
            # Add offset for promotions
            action += 4096  # Base offset for promotions
            if promotion == PieceType.QUEEN:
                action += 0
            elif promotion == PieceType.ROOK:
                action += 1
            elif promotion == PieceType.BISHOP:
                action += 2
            elif promotion == PieceType.KNIGHT:
                action += 3
            elif promotion == PieceType.KING:
                action += 4
        return action
    
    def decode_move(self, action: int, game: ExtinctionChess):
        """Decode an action integer back to a move"""
        # This is complex - for now we'll use a simpler approach
        # Just map legal moves to indices
        legal_moves = game.get_legal_moves()
        if 0 <= action < len(legal_moves):
            return legal_moves[action]
        return None
    
    def get_simple_features(self, game: ExtinctionChess) -> np.ndarray:
        """
        Extract simple hand-crafted features for a basic evaluator
        This is easier to train than raw board representation
        """
        features = []
        
        # Material count for each piece type (normalized)
        for color in [Color.WHITE, Color.BLACK]:
            counts = game.board.get_piece_count(color)
            for piece_type in PieceType:
                features.append(counts[piece_type] / 8.0)  # Normalize by max reasonable count
        
        # Endangered pieces (binary features)
        white_endangered = game.get_endangered_pieces(Color.WHITE)
        black_endangered = game.get_endangered_pieces(Color.BLACK)
        
        for piece_type in PieceType:
            features.append(1.0 if piece_type in white_endangered else 0.0)
            features.append(1.0 if piece_type in black_endangered else 0.0)
        
        # Extinct pieces (binary features)
        white_extinct = game.get_extinct_pieces(Color.WHITE)
        black_extinct = game.get_extinct_pieces(Color.BLACK)
        
        for piece_type in PieceType:
            features.append(1.0 if piece_type in white_extinct else 0.0)
            features.append(1.0 if piece_type in black_extinct else 0.0)
        
        # Current player
        features.append(1.0 if game.current_player == Color.WHITE else 0.0)
        
        # Number of legal moves (mobility)
        features.append(len(game.get_legal_moves()) / 100.0)  # Normalize
        
        # Game phase (early/mid/late based on piece count)
        total_pieces = sum(counts[pt] for counts in 
                          [game.board.get_piece_count(Color.WHITE), 
                           game.board.get_piece_count(Color.BLACK)]
                          for pt in PieceType)
        features.append(total_pieces / 32.0)  # Normalize by starting pieces
        
        return np.array(features, dtype=np.float32)
    
    def create_training_batch(self, games_data: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training batches from game data
        games_data: List of (encoded_state, outcome) tuples
        Returns: (states_batch, outcomes_batch)
        """
        if not games_data:
            return np.array([]), np.array([])
        
        states = np.stack([state for state, _ in games_data])
        outcomes = np.array([outcome for _, outcome in games_data], dtype=np.float32)
        
        return states, outcomes


class MoveEncoder:
    """Simpler move encoding system for action selection"""
    
    @staticmethod
    def moves_to_policy(legal_moves, move_probs=None):
        """
        Convert legal moves to a policy vector
        If move_probs is None, returns uniform distribution
        """
        if move_probs is None:
            # Uniform distribution over legal moves
            return np.ones(len(legal_moves)) / len(legal_moves)
        return move_probs
    
    @staticmethod
    def select_move(legal_moves, policy, temperature=1.0):
        """
        Select a move based on policy probabilities
        Temperature controls exploration:
        - 0: deterministic (best move)
        - 1: sample from distribution
        - >1: more random
        """
        if temperature == 0:
            return legal_moves[np.argmax(policy)]
        
        # Apply temperature
        log_probs = np.log(policy + 1e-10)
        log_probs = log_probs / temperature
        probs = np.exp(log_probs - np.max(log_probs))
        probs = probs / np.sum(probs)
        
        # Sample move
        move_idx = np.random.choice(len(legal_moves), p=probs)
        return legal_moves[move_idx]


# Example usage
if __name__ == "__main__":
    from extinction_chess import ExtinctionChess
    
    # Create game and encoder
    game = ExtinctionChess()
    encoder = StateEncoder()
    
    # Encode current position
    tensor = encoder.encode_board(game)
    print(f"Board tensor shape: {tensor.shape}")
    print(f"Non-zero elements: {np.count_nonzero(tensor)}")
    
    # Get simple features
    features = encoder.get_simple_features(game)
    print(f"\nFeature vector shape: {features.shape}")
    print(f"Features: {features[:10]}...")  # Show first 10 features
    
    # Test move encoding
    legal_moves = game.get_legal_moves()
    print(f"\nNumber of legal moves: {len(legal_moves)}")
    
    # Test move selection
    move_encoder = MoveEncoder()
    uniform_policy = move_encoder.moves_to_policy(legal_moves)
    selected_move = move_encoder.select_move(legal_moves, uniform_policy, temperature=1.0)
    print(f"Selected move: {selected_move}")