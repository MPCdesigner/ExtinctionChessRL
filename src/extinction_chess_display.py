"""
Display module for Extinction Chess
Handles all visualization and UI-related functionality
"""

from typing import Optional, List, Dict
from extinction_chess import ExtinctionChess, Board, Position, Piece, Move, Color, PieceType


class ChessDisplay:
    """Base class for chess display/visualization"""
    
    def __init__(self, game: ExtinctionChess):
        self.game = game
    
    def display(self):
        """Display the current game state"""
        raise NotImplementedError


class ConsoleDisplay(ChessDisplay):
    """Terminal/console display for Extinction Chess"""
    
    def __init__(self, game: ExtinctionChess):
        super().__init__(game)
        self.unicode_pieces = {
            (PieceType.KING, Color.WHITE): '♔',
            (PieceType.QUEEN, Color.WHITE): '♕',
            (PieceType.ROOK, Color.WHITE): '♖',
            (PieceType.BISHOP, Color.WHITE): '♗',
            (PieceType.KNIGHT, Color.WHITE): '♘',
            (PieceType.PAWN, Color.WHITE): '♙',
            (PieceType.KING, Color.BLACK): '♚',
            (PieceType.QUEEN, Color.BLACK): '♛',
            (PieceType.ROOK, Color.BLACK): '♜',
            (PieceType.BISHOP, Color.BLACK): '♝',
            (PieceType.KNIGHT, Color.BLACK): '♞',
            (PieceType.PAWN, Color.BLACK): '♟'
        }
        self.use_unicode = True
    
    def get_piece_symbol(self, piece: Optional[Piece]) -> str:
        """Get display symbol for a piece"""
        if not piece:
            return '.'
        
        if self.use_unicode:
            return self.unicode_pieces.get((piece.piece_type, piece.color), '?')
        else:
            # ASCII representation
            symbol = piece.piece_type.value
            return symbol.lower() if piece.color == Color.BLACK else symbol
    
    def display(self):
        """Display the current board state in the console"""
        self.display_board()
        self.display_piece_counts()
        self.display_game_status()
    
    def display_board(self):
        """Display the chess board"""
        print("\n  a b c d e f g h")
        print("  ---------------")
        
        for rank in range(7, -1, -1):
            print(f"{rank+1}|", end="")
            for file in range(8):
                piece = self.game.board.get_piece(Position(rank, file))
                symbol = self.get_piece_symbol(piece)
                print(f"{symbol} ", end="")
            print(f"|{rank+1}")
        
        print("  ---------------")
        print("  a b c d e f g h\n")
    
    def display_piece_counts(self):
        """Display piece counts and endangered status"""
        state = self.game.get_game_state()
        
        for color in [Color.WHITE, Color.BLACK]:
            color_name = color.name.capitalize()
            counts = state[f'{color.name.lower()}_counts']
            endangered = self.game.get_endangered_pieces(color)
            
            print(f"{color_name} pieces:", end=" ")
            for piece_type, count in counts.items():
                if count > 0:
                    status = " (endangered)" if piece_type in endangered else ""
                    print(f"{piece_type.value}:{count}{status}", end=" ")
            print()
    
    def display_game_status(self):
        """Display current game status"""
        state = self.game.get_game_state()
        
        if state['game_over']:
            if state['winner']:
                print(f"\nGame Over! {state['winner'].name} wins!")
            else:
                print(f"\nGame Drawn! ({state['draw_reason']})")
        else:
            print(f"\n{state['current_player'].name} to move")
            print(f"Move {state['fullmove_number']}, Half-moves since capture/pawn: {state['halfmove_clock']}")
    
    def display_legal_moves(self, from_position: Optional[Position] = None):
        """Display all legal moves, optionally filtered by piece position"""
        moves = self.game.get_legal_moves(from_position)
        
        if not moves:
            print("No legal moves available")
            return
        
        print(f"\nLegal moves ({len(moves)} total):")
        
        # Group moves by piece
        moves_by_piece = {}
        for move in moves:
            piece = self.game.board.get_piece(move.from_pos)
            key = f"{piece.piece_type.value} at {move.from_pos.to_algebraic()}"
            if key not in moves_by_piece:
                moves_by_piece[key] = []
            moves_by_piece[key].append(move)
        
        for piece_desc, piece_moves in moves_by_piece.items():
            print(f"  {piece_desc}:")
            for move in piece_moves[:5]:  # Show first 5 moves for each piece
                move_desc = f"    → {move.to_pos.to_algebraic()}"
                if move.promotion:
                    move_desc += f" (promote to {move.promotion.value})"
                if move.is_castling:
                    move_desc += " (castle)"
                if move.is_en_passant:
                    move_desc += " (en passant)"
                print(move_desc)
            if len(piece_moves) > 5:
                print(f"    ... and {len(piece_moves) - 5} more")
    
    def display_endangered_warning(self):
        """Display warnings about endangered pieces"""
        for color in [Color.WHITE, Color.BLACK]:
            endangered = self.game.get_endangered_pieces(color)
            if endangered:
                color_name = color.name.capitalize()
                pieces_str = ", ".join([pt.value for pt in endangered])
                print(f"⚠️  {color_name} endangered pieces: {pieces_str}")


class ASCIIDisplay(ConsoleDisplay):
    """ASCII-only display (no Unicode characters)"""
    
    def __init__(self, game: ExtinctionChess):
        super().__init__(game)
        self.use_unicode = False


class CompactDisplay(ChessDisplay):
    """Compact single-line display for logging/training"""
    
    def display(self):
        """Display game state in a compact format"""
        state = self.game.get_game_state()
        
        # Build compact representation
        if state['game_over']:
            if state['winner']:
                status = f"WIN:{state['winner'].name}"
            else:
                status = f"DRAW:{state['draw_reason']}"
        else:
            status = f"PLAY:{state['current_player'].name}"
        
        # Count total pieces
        white_total = sum(state['white_counts'].values())
        black_total = sum(state['black_counts'].values())
        
        print(f"[Move {state['fullmove_number']}] {status} | W:{white_total} B:{black_total} | Moves:{len(state['legal_moves'])}")


class FENDisplay(ChessDisplay):
    """Display board in FEN-like notation (modified for Extinction Chess)"""
    
    def get_fen_string(self) -> str:
        """Generate a FEN-like string for the current position"""
        fen_parts = []
        
        # 1. Piece placement
        for rank in range(7, -1, -1):
            empty_count = 0
            rank_str = ""
            
            for file in range(8):
                piece = self.game.board.get_piece(Position(rank, file))
                if piece:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    
                    symbol = piece.piece_type.value
                    rank_str += symbol.lower() if piece.color == Color.BLACK else symbol
                else:
                    empty_count += 1
            
            if empty_count > 0:
                rank_str += str(empty_count)
            
            fen_parts.append(rank_str)
        
        board_fen = "/".join(fen_parts)
        
        # 2. Active color
        active = 'w' if self.game.current_player == Color.WHITE else 'b'
        
        # 3. Castling availability (modified for extinction chess)
        castling = self.get_castling_rights()
        
        # 4. En passant
        ep = self.game.board.en_passant_target.to_algebraic() if self.game.board.en_passant_target else "-"
        
        # 5. Halfmove clock
        halfmove = str(self.game.board.halfmove_clock)
        
        # 6. Fullmove number
        fullmove = str(self.game.board.fullmove_number)
        
        return f"{board_fen} {active} {castling} {ep} {halfmove} {fullmove}"
    
    def get_castling_rights(self) -> str:
        """Get castling rights string"""
        rights = ""
        
        # Check white castling
        white_king = self.game.board.find_king(Color.WHITE)
        if white_king and not white_king.has_moved:
            # Kingside
            kingside_rook = self.game.board.get_piece(Position(0, 7))
            if kingside_rook and kingside_rook.piece_type == PieceType.ROOK and not kingside_rook.has_moved:
                rights += "K"
            # Queenside
            queenside_rook = self.game.board.get_piece(Position(0, 0))
            if queenside_rook and queenside_rook.piece_type == PieceType.ROOK and not queenside_rook.has_moved:
                rights += "Q"
        
        # Check black castling
        black_king = self.game.board.find_king(Color.BLACK)
        if black_king and not black_king.has_moved:
            # Kingside
            kingside_rook = self.game.board.get_piece(Position(7, 7))
            if kingside_rook and kingside_rook.piece_type == PieceType.ROOK and not kingside_rook.has_moved:
                rights += "k"
            # Queenside
            queenside_rook = self.game.board.get_piece(Position(7, 0))
            if queenside_rook and queenside_rook.piece_type == PieceType.ROOK and not queenside_rook.has_moved:
                rights += "q"
        
        return rights if rights else "-"
    
    def display(self):
        """Display the FEN string"""
        print(f"FEN: {self.get_fen_string()}")


# Example usage
if __name__ == "__main__":
    from extinction_chess import ExtinctionChess
    
    # Create game
    game = ExtinctionChess()
    
    # Create different display types
    console_display = ConsoleDisplay(game)
    ascii_display = ASCIIDisplay(game)
    compact_display = CompactDisplay(game)
    fen_display = FENDisplay(game)
    
    # Display using different formats
    print("=== Console Display (Unicode) ===")
    console_display.display()
    
    print("\n=== ASCII Display ===")
    ascii_display.display_board()
    
    print("\n=== Compact Display ===")
    compact_display.display()
    
    print("\n=== FEN Display ===")
    fen_display.display()
    
    print("\n=== Legal Moves ===")
    console_display.display_legal_moves()
    
    # Make a move and display again
    moves = game.get_legal_moves()
    if moves:
        game.make_move(moves[0])
        print("\n=== After First Move ===")
        console_display.display()
        compact_display.display()