from enum import Enum
from typing import List, Optional, Tuple, Set, Dict
from dataclasses import dataclass
from copy import deepcopy
from collections import Counter

class PieceType(Enum):
    PAWN = 'P'
    KNIGHT = 'N'
    BISHOP = 'B'
    ROOK = 'R'
    QUEEN = 'Q'
    KING = 'K'

class Color(Enum):
    WHITE = True
    BLACK = False

@dataclass
class Position:
    rank: int  # 0-7 (0 is white's back rank)
    file: int  # 0-7 (0 is a-file)
    
    def __eq__(self, other):
        return self.rank == other.rank and self.file == other.file
    
    def __hash__(self):
        return hash((self.rank, self.file))
    
    def to_algebraic(self) -> str:
        return f"{chr(97 + self.file)}{self.rank + 1}"

class Piece:
    def __init__(self, piece_type: PieceType, color: Color, position: Position):
        self.piece_type = piece_type
        self.color = color
        self.position = position
        self.has_moved = False
    
    def __repr__(self):
        color_str = 'W' if self.color == Color.WHITE else 'B'
        return f"{color_str}{self.piece_type.value}"

class Move:
    def __init__(self, from_pos: Position, to_pos: Position, 
                 promotion: Optional[PieceType] = None,
                 is_en_passant: bool = False,
                 is_castling: bool = False,
                 castling_rook_from: Optional[Position] = None,
                 castling_rook_to: Optional[Position] = None):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.promotion = promotion
        self.is_en_passant = is_en_passant
        self.is_castling = is_castling
        self.castling_rook_from = castling_rook_from
        self.castling_rook_to = castling_rook_to
        self.captured_piece = None
    
    def __repr__(self):
        move_str = f"{self.from_pos.to_algebraic()}-{self.to_pos.to_algebraic()}"
        if self.promotion:
            move_str += f"={self.promotion.value}"
        if self.is_castling:
            move_str += " (castle)"
        return move_str

class Board:
    def __init__(self):
        self.grid = [[None for _ in range(8)] for _ in range(8)]
        self.en_passant_target = None  # Position where en passant capture can occur
        self.halfmove_clock = 0  # For 50-move rule
        self.fullmove_number = 1
        self.position_history = []  # For threefold repetition
        self.setup_initial_position()
    
    def setup_initial_position(self):
        # Set up white pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, 
                      PieceType.QUEEN, PieceType.KING, PieceType.BISHOP, 
                      PieceType.KNIGHT, PieceType.ROOK]
        
        for file, piece_type in enumerate(piece_order):
            self.grid[0][file] = Piece(piece_type, Color.WHITE, Position(0, file))
            self.grid[1][file] = Piece(PieceType.PAWN, Color.WHITE, Position(1, file))
            self.grid[7][file] = Piece(piece_type, Color.BLACK, Position(7, file))
            self.grid[6][file] = Piece(PieceType.PAWN, Color.BLACK, Position(6, file))
    
    def get_piece(self, pos: Position) -> Optional[Piece]:
        if self.is_valid_position(pos):
            return self.grid[pos.rank][pos.file]
        return None
    
    def set_piece(self, pos: Position, piece: Optional[Piece]):
        if self.is_valid_position(pos):
            self.grid[pos.rank][pos.file] = piece
            if piece:
                piece.position = pos
    
    def is_valid_position(self, pos: Position) -> bool:
        return 0 <= pos.rank < 8 and 0 <= pos.file < 8
    
    def get_piece_count(self, color: Color) -> Dict[PieceType, int]:
        """Count pieces by type for extinction checking"""
        counts = {pt: 0 for pt in PieceType}
        for rank in self.grid:
            for piece in rank:
                if piece and piece.color == color:
                    counts[piece.piece_type] += 1
        return counts
    
    def get_position_key(self, current_player: Color) -> str:
        """Generate a unique key for position (for threefold repetition)"""
        key_parts = []
        for rank in self.grid:
            for piece in rank:
                if piece:
                    key_parts.append(f"{piece.color.value}{piece.piece_type.value}")
                else:
                    key_parts.append("--")
        key_parts.append(str(current_player.value))
        key_parts.append(str(self.en_passant_target))
        # Add castling rights
        for color in [Color.WHITE, Color.BLACK]:
            king = self.find_king(color)
            if king and not king.has_moved:
                for file in [0, 7]:
                    rook = self.get_piece(Position(0 if color == Color.WHITE else 7, file))
                    if rook and rook.piece_type == PieceType.ROOK and not rook.has_moved:
                        key_parts.append(f"castle{color.value}{file}")
        return "|".join(key_parts)
    
    def find_king(self, color: Color) -> Optional[Piece]:
        """Find a king of given color (there might be multiple in extinction chess!)"""
        for rank in self.grid:
            for piece in rank:
                if piece and piece.piece_type == PieceType.KING and piece.color == color:
                    return piece
        return None
    
    def copy(self):
        """Create a deep copy of the board"""
        new_board = Board()
        new_board.grid = [[None for _ in range(8)] for _ in range(8)]
        for rank in range(8):
            for file in range(8):
                if self.grid[rank][file]:
                    old_piece = self.grid[rank][file]
                    new_piece = Piece(old_piece.piece_type, old_piece.color, 
                                    Position(rank, file))
                    new_piece.has_moved = old_piece.has_moved
                    new_board.grid[rank][file] = new_piece
        new_board.en_passant_target = self.en_passant_target
        new_board.halfmove_clock = self.halfmove_clock
        new_board.fullmove_number = self.fullmove_number
        new_board.position_history = self.position_history.copy()
        return new_board

class ExtinctionChess:
    def __init__(self):
        self.board = Board()
        self.current_player = Color.WHITE
        self.game_over = False
        self.winner = None
        self.draw_reason = None
    
    def get_legal_moves(self, from_pos: Optional[Position] = None) -> List[Move]:
        """Get all legal moves for current player or for a specific piece"""
        moves = []
        for rank in range(8):
            for file in range(8):
                pos = Position(rank, file)
                piece = self.board.get_piece(pos)
                if piece and piece.color == self.current_player:
                    if from_pos is None or pos == from_pos:
                        moves.extend(self.get_piece_moves(piece))
        return moves
    
    def get_piece_moves(self, piece: Piece) -> List[Move]:
        """Get all possible moves for a piece"""
        moves = []
        
        if piece.piece_type == PieceType.PAWN:
            moves = self.get_pawn_moves(piece)
        elif piece.piece_type == PieceType.KNIGHT:
            moves = self.get_knight_moves(piece)
        elif piece.piece_type == PieceType.BISHOP:
            moves = self.get_bishop_moves(piece)
        elif piece.piece_type == PieceType.ROOK:
            moves = self.get_rook_moves(piece)
        elif piece.piece_type == PieceType.QUEEN:
            moves = self.get_queen_moves(piece)
        elif piece.piece_type == PieceType.KING:
            moves = self.get_king_moves(piece)
        
        return moves
    
    def get_pawn_moves(self, pawn: Piece) -> List[Move]:
        moves = []
        direction = 1 if pawn.color == Color.WHITE else -1
        start_rank = 1 if pawn.color == Color.WHITE else 6
        promotion_rank = 7 if pawn.color == Color.WHITE else 0
        
        # One square forward
        front_pos = Position(pawn.position.rank + direction, pawn.position.file)
        if self.board.is_valid_position(front_pos) and not self.board.get_piece(front_pos):
            if front_pos.rank == promotion_rank:
                # Must promote (to anything except pawn)
                for promo_type in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, 
                                  PieceType.KNIGHT, PieceType.KING]:
                    moves.append(Move(pawn.position, front_pos, promotion=promo_type))
            else:
                moves.append(Move(pawn.position, front_pos))
        
        # Two squares forward from starting position
        if pawn.position.rank == start_rank:
            front2_pos = Position(pawn.position.rank + 2 * direction, pawn.position.file)
            if (not self.board.get_piece(front_pos) and 
                not self.board.get_piece(front2_pos)):
                moves.append(Move(pawn.position, front2_pos))
        
        # Captures (diagonal)
        for file_offset in [-1, 1]:
            capture_pos = Position(pawn.position.rank + direction, 
                                 pawn.position.file + file_offset)
            if self.board.is_valid_position(capture_pos):
                target = self.board.get_piece(capture_pos)
                if target and target.color != pawn.color:
                    if capture_pos.rank == promotion_rank:
                        for promo_type in [PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, 
                                          PieceType.KNIGHT, PieceType.KING]:
                            moves.append(Move(pawn.position, capture_pos, promotion=promo_type))
                    else:
                        moves.append(Move(pawn.position, capture_pos))
        
        # En passant
        if self.board.en_passant_target:
            for file_offset in [-1, 1]:
                if (pawn.position.file + file_offset == self.board.en_passant_target.file and
                    pawn.position.rank + direction == self.board.en_passant_target.rank):
                    moves.append(Move(pawn.position, self.board.en_passant_target, 
                                    is_en_passant=True))
        
        return moves
    
    def get_knight_moves(self, knight: Piece) -> List[Move]:
        moves = []
        offsets = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
        
        for rank_offset, file_offset in offsets:
            new_pos = Position(knight.position.rank + rank_offset, 
                             knight.position.file + file_offset)
            if self.board.is_valid_position(new_pos):
                target = self.board.get_piece(new_pos)
                if not target or target.color != knight.color:
                    moves.append(Move(knight.position, new_pos))
        
        return moves
    
    def get_sliding_moves(self, piece: Piece, directions: List[Tuple[int, int]]) -> List[Move]:
        """Get moves for sliding pieces (bishop, rook, queen)"""
        moves = []
        
        for rank_dir, file_dir in directions:
            for distance in range(1, 8):
                new_pos = Position(piece.position.rank + rank_dir * distance,
                                 piece.position.file + file_dir * distance)
                
                if not self.board.is_valid_position(new_pos):
                    break
                
                target = self.board.get_piece(new_pos)
                if not target:
                    moves.append(Move(piece.position, new_pos))
                elif target.color != piece.color:
                    moves.append(Move(piece.position, new_pos))
                    break
                else:
                    break
        
        return moves
    
    def get_bishop_moves(self, bishop: Piece) -> List[Move]:
        return self.get_sliding_moves(bishop, [(1,1), (1,-1), (-1,1), (-1,-1)])
    
    def get_rook_moves(self, rook: Piece) -> List[Move]:
        return self.get_sliding_moves(rook, [(1,0), (-1,0), (0,1), (0,-1)])
    
    def get_queen_moves(self, queen: Piece) -> List[Move]:
        return self.get_sliding_moves(queen, [(1,0), (-1,0), (0,1), (0,-1),
                                              (1,1), (1,-1), (-1,1), (-1,-1)])
    
    def get_king_moves(self, king: Piece) -> List[Move]:
        moves = []
        
        # Regular moves (one square in any direction)
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue
                new_pos = Position(king.position.rank + rank_offset,
                                 king.position.file + file_offset)
                if self.board.is_valid_position(new_pos):
                    target = self.board.get_piece(new_pos)
                    if not target or target.color != king.color:
                        moves.append(Move(king.position, new_pos))
        
        # Castling (allowed even in/through/out of "check" in extinction chess)
        if not king.has_moved:
            rank = 0 if king.color == Color.WHITE else 7
            
            # Kingside castling
            kingside_rook = self.board.get_piece(Position(rank, 7))
            if (kingside_rook and kingside_rook.piece_type == PieceType.ROOK and
                not kingside_rook.has_moved and kingside_rook.color == king.color):
                # Check if squares between are empty
                if (not self.board.get_piece(Position(rank, 5)) and
                    not self.board.get_piece(Position(rank, 6))):
                    moves.append(Move(king.position, Position(rank, 6), is_castling=True,
                                     castling_rook_from=Position(rank, 7),
                                     castling_rook_to=Position(rank, 5)))
            
            # Queenside castling
            queenside_rook = self.board.get_piece(Position(rank, 0))
            if (queenside_rook and queenside_rook.piece_type == PieceType.ROOK and
                not queenside_rook.has_moved and queenside_rook.color == king.color):
                # Check if squares between are empty
                if (not self.board.get_piece(Position(rank, 1)) and
                    not self.board.get_piece(Position(rank, 2)) and
                    not self.board.get_piece(Position(rank, 3))):
                    moves.append(Move(king.position, Position(rank, 2), is_castling=True,
                                     castling_rook_from=Position(rank, 0),
                                     castling_rook_to=Position(rank, 3)))
        
        return moves
    
    def make_move(self, move: Move) -> bool:
        """Execute a move and check for game ending conditions"""
        if self.game_over:
            return False
        
        piece = self.board.get_piece(move.from_pos)
        if not piece or piece.color != self.current_player:
            return False
        
        # Track position for threefold repetition before making move
        position_key = self.board.get_position_key(self.current_player)
        self.board.position_history.append(position_key)
        
        # Handle captures
        captured = self.board.get_piece(move.to_pos)
        if captured:
            move.captured_piece = captured
            self.board.halfmove_clock = 0
        else:
            self.board.halfmove_clock += 1
        
        # Special moves
        if move.is_castling:
            # Move king
            self.board.set_piece(move.from_pos, None)
            self.board.set_piece(move.to_pos, piece)
            piece.has_moved = True
            
            # Move rook
            rook = self.board.get_piece(move.castling_rook_from)
            self.board.set_piece(move.castling_rook_from, None)
            self.board.set_piece(move.castling_rook_to, rook)
            rook.has_moved = True
            
        elif move.is_en_passant:
            # Remove the captured pawn
            captured_pawn_pos = Position(move.from_pos.rank, move.to_pos.file)
            captured = self.board.get_piece(captured_pawn_pos)
            move.captured_piece = captured
            self.board.set_piece(captured_pawn_pos, None)
            
            # Move the capturing pawn
            self.board.set_piece(move.from_pos, None)
            self.board.set_piece(move.to_pos, piece)
            
            self.board.halfmove_clock = 0
            
        else:
            # Regular move
            self.board.set_piece(move.from_pos, None)
            
            # Handle promotion
            if move.promotion:
                promoted_piece = Piece(move.promotion, piece.color, move.to_pos)
                promoted_piece.has_moved = True
                self.board.set_piece(move.to_pos, promoted_piece)
                self.board.halfmove_clock = 0
            else:
                self.board.set_piece(move.to_pos, piece)
                piece.has_moved = True
                
                if piece.piece_type == PieceType.PAWN:
                    self.board.halfmove_clock = 0
        
        # Update en passant target
        self.board.en_passant_target = None
        if piece.piece_type == PieceType.PAWN and abs(move.to_pos.rank - move.from_pos.rank) == 2:
            self.board.en_passant_target = Position(
                (move.from_pos.rank + move.to_pos.rank) // 2,
                move.from_pos.file
            )
        
        # Check for extinction
        if self.check_extinction(move):
            return True
        
        # Check for draws
        if self.check_draws():
            return True
        
        # Switch players
        self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        
        if self.current_player == Color.WHITE:
            self.board.fullmove_number += 1
        
        return True
    
    def check_extinction(self, last_move: Move) -> bool:
        """Check if either player has lost all pieces of a type"""
        # Get piece counts for both players
        white_counts = self.board.get_piece_count(Color.WHITE)
        black_counts = self.board.get_piece_count(Color.BLACK)
        
        # Check for extinctions
        white_extinct = any(count == 0 for count in white_counts.values())
        black_extinct = any(count == 0 for count in black_counts.values())
        
        # Handle promotion causing pawn extinction
        if last_move.promotion and self.current_player == Color.WHITE:
            if white_counts[PieceType.PAWN] == 0:
                white_extinct = True
        elif last_move.promotion and self.current_player == Color.BLACK:
            if black_counts[PieceType.PAWN] == 0:
                black_extinct = True
        
        # Rule 9: If both sides have extinction, the player who made the move wins
        if white_extinct and black_extinct:
            self.game_over = True
            self.winner = self.current_player
            return True
        elif white_extinct:
            self.game_over = True
            self.winner = Color.BLACK
            return True
        elif black_extinct:
            self.game_over = True
            self.winner = Color.WHITE
            return True
        
        return False
    
    def check_draws(self) -> bool:
        """Check for draw conditions"""
        # 50-move rule
        if self.board.halfmove_clock >= 100:  # 50 moves = 100 half-moves
            self.game_over = True
            self.draw_reason = "50-move rule"
            return True
        
        # Threefold repetition
        position_counts = Counter(self.board.position_history)
        if any(count >= 3 for count in position_counts.values()):
            self.game_over = True
            self.draw_reason = "Threefold repetition"
            return True
        
        # Stalemate (no legal moves)
        if not self.get_legal_moves():
            self.game_over = True
            self.draw_reason = "Stalemate"
            return True
        
        return False
    
    def get_game_state(self) -> dict:
        """Return complete game state for external use (e.g., display, AI evaluation)"""
        return {
            'board': self.board,
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'draw_reason': self.draw_reason,
            'white_counts': self.board.get_piece_count(Color.WHITE),
            'black_counts': self.board.get_piece_count(Color.BLACK),
            'legal_moves': self.get_legal_moves() if not self.game_over else [],
            'en_passant_target': self.board.en_passant_target,
            'halfmove_clock': self.board.halfmove_clock,
            'fullmove_number': self.board.fullmove_number
        }
    
    def get_endangered_pieces(self, color: Color) -> List[PieceType]:
        """Return list of piece types that are endangered (only 1 left) for given color"""
        counts = self.board.get_piece_count(color)
        return [piece_type for piece_type, count in counts.items() if count == 1]
    
    def get_extinct_pieces(self, color: Color) -> List[PieceType]:
        """Return list of piece types that are extinct (0 left) for given color"""
        counts = self.board.get_piece_count(color)
        return [piece_type for piece_type, count in counts.items() if count == 0]