"""
Interactive 2-player mode for Extinction Chess
Allows manual play for testing game logic
"""

import os
import sys
from typing import Optional, List, Tuple
from extinction_chess import (
    ExtinctionChess, Position, Move, Color, PieceType, 
    Board, Piece
)
from extinction_chess_display import ConsoleDisplay, ASCIIDisplay


class InteractiveGame:
    """Interactive 2-player mode for testing Extinction Chess"""
    
    def __init__(self, use_unicode: bool = True):
        self.game = ExtinctionChess()
        self.display = ConsoleDisplay(self.game) if use_unicode else ASCIIDisplay(self.game)
        self.move_history = []
        self.command_history = []
        
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def parse_position(self, pos_str: str) -> Optional[Position]:
        """Parse algebraic notation (e.g., 'e2') to Position"""
        pos_str = pos_str.strip().lower()
        if len(pos_str) != 2:
            return None
        
        file_char = pos_str[0]
        rank_char = pos_str[1]
        
        if file_char < 'a' or file_char > 'h':
            return None
        if rank_char < '1' or rank_char > '8':
            return None
        
        file = ord(file_char) - ord('a')
        rank = int(rank_char) - 1
        
        return Position(rank, file)
    
    def parse_piece_type(self, type_str: str) -> Optional[PieceType]:
        """Parse piece type for promotion"""
        type_str = type_str.upper().strip()
        mapping = {
            'Q': PieceType.QUEEN,
            'QUEEN': PieceType.QUEEN,
            'R': PieceType.ROOK,
            'ROOK': PieceType.ROOK,
            'B': PieceType.BISHOP,
            'BISHOP': PieceType.BISHOP,
            'N': PieceType.KNIGHT,
            'KNIGHT': PieceType.KNIGHT,
            'K': PieceType.KING,
            'KING': PieceType.KING
        }
        return mapping.get(type_str)
    
    def get_moves_from_position(self, from_pos: Position) -> List[Move]:
        """Get all legal moves from a specific position"""
        return self.game.get_legal_moves(from_pos)
    
    def select_promotion(self) -> Optional[PieceType]:
        """Prompt user to select promotion piece"""
        print("\nPawn promotion! Choose piece:")
        print("  Q/QUEEN - Queen")
        print("  R/ROOK  - Rook")
        print("  B/BISHOP - Bishop")
        print("  N/KNIGHT - Knight")
        print("  K/KING  - King (yes, you can promote to King in Extinction Chess!)")
        
        while True:
            choice = input("Enter piece type: ").strip().upper()
            piece_type = self.parse_piece_type(choice)
            if piece_type and piece_type != PieceType.PAWN:
                return piece_type
            print("Invalid choice. Please enter Q, R, B, N, or K")
    
    def format_move_list(self, moves: List[Move]) -> List[str]:
        """Format moves for display"""
        move_strs = []
        
        for i, move in enumerate(moves):
            move_str = f"{i+1}. {move.from_pos.to_algebraic()}-{move.to_pos.to_algebraic()}"
            
            # Add annotations
            target = self.game.board.get_piece(move.to_pos)
            if target:
                move_str += f" (captures {target.piece_type.value})"
            if move.is_castling:
                move_str += " (castle)"
            if move.is_en_passant:
                move_str += " (en passant)"
            if move.promotion:
                move_str += f" (promote to {move.promotion.value})"
            
            move_strs.append(move_str)
        return move_strs
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("EXTINCTION CHESS - COMMANDS")
        print("="*50)
        print("\nMOVE FORMATS:")
        print("  e2 e4     - Move from e2 to e4")
        print("  e2-e4     - Alternative format")
        print("  e2        - Show all moves for piece at e2")
        print()
        print("SPECIAL COMMANDS:")
        print("  help      - Show this help")
        print("  board     - Redraw the board")
        print("  moves     - Show all legal moves")
        print("  history   - Show move history")
        print("  undo      - Take back last move (both players)")
        print("  resign    - Resign the game")
        print("  draw      - Offer/accept draw")
        print("  clear     - Clear screen and redraw")
        print("  counts    - Show detailed piece counts")
        print("  fen       - Show FEN-like position")
        print("  rules     - Show Extinction Chess rules")
        print("  quit/exit - Exit the game")
        print()
        print("EXTINCTION CHESS SPECIAL RULES:")
        print("  • No check/checkmate - King can be captured")
        print("  • Win by making any opponent piece type extinct")
        print("  • Pawns MUST promote (can promote to King!)")
        print("  • Can castle through/into/out of 'check'")
        print("="*50)
    
    def show_rules(self):
        """Display Extinction Chess rules"""
        print("\n" + "="*50)
        print("EXTINCTION CHESS RULES")
        print("="*50)
        print("""
1. OBJECTIVE: Eliminate ALL of any one type of opponent's pieces

2. NO CHECK/CHECKMATE: 
   - The king can be captured like any piece
   - No need to announce check or protect your king

3. CASTLING:
   - Normal castling rules apply (king and rook haven't moved, 
     squares between are empty)
   - You CAN castle through/into/out of "check" since check 
     doesn't exist in this variant

4. WINNING:
   - Making any piece type extinct wins immediately
   - If both players lose a type simultaneously, the player
     who made the move wins

5. PAWNS:
   - MUST promote at the end (cannot remain as pawns)
   - Can promote to ANY piece including KING
   - Cannot promote to pawn (would cause extinction)

6. ENDANGERED PIECES:
   - When you have only 1 of a piece type, it's "endangered"
   - Losing it means extinction and immediate loss

7. DRAWS:
   - 50 moves without capture/pawn move
   - Threefold repetition
   - No legal moves (stalemate)
   - Mutual agreement
        """)
        print("="*50)
    
    def show_piece_counts(self):
        """Show detailed piece counts"""
        print("\n" + "="*30)
        print("PIECE COUNTS")
        print("="*30)
        
        for color in [Color.WHITE, Color.BLACK]:
            counts = self.game.board.get_piece_count(color)
            endangered = self.game.get_endangered_pieces(color)
            
            print(f"\n{color.name}:")
            for piece_type in PieceType:
                count = counts[piece_type]
                status = ""
                if count == 0:
                    status = " [EXTINCT]"
                elif count == 1:
                    status = " [ENDANGERED!]"
                
                symbol = "  "
                if count > 0:
                    # Get a piece of this type to show symbol
                    for rank in range(8):
                        for file in range(8):
                            p = self.game.board.get_piece(Position(rank, file))
                            if p and p.piece_type == piece_type and p.color == color:
                                symbol = self.display.get_piece_symbol(p) + " "
                                break
                        if symbol != "  ":
                            break
                
                print(f"  {symbol}{piece_type.value}: {count}{status}")
    
    def show_move_history(self):
        """Display the move history"""
        print("\n" + "="*30)
        print("MOVE HISTORY")
        print("="*30)
        
        if not self.move_history:
            print("No moves yet")
            return
        
        for i, (move_num, color, move_str) in enumerate(self.move_history):
            if i % 2 == 0:
                print(f"{move_num}. ", end="")
            
            print(f"{move_str}", end="  ")
            
            if i % 2 == 1:
                print()
        
        if len(self.move_history) % 2 == 1:
            print()
    
    def handle_move_input(self, input_str: str) -> bool:
        """Handle move input from user. Returns True if move was made."""
        parts = input_str.replace('-', ' ').split()
        
        if len(parts) == 1:
            # User entered just a position - show moves from that square
            from_pos = self.parse_position(parts[0])
            if not from_pos:
                print(f"Invalid position: {parts[0]}")
                return False
            
            piece = self.game.board.get_piece(from_pos)
            if not piece:
                print(f"No piece at {parts[0]}")
                return False
            
            if piece.color != self.game.current_player:
                print(f"That's not your piece! It's {self.game.current_player.name}'s turn")
                return False
            
            moves = self.get_moves_from_position(from_pos)
            if not moves:
                print(f"No legal moves for {piece.piece_type.value} at {from_pos.to_algebraic()}")
                return False
            
            print(f"\nLegal moves for {piece.piece_type.value} at {from_pos.to_algebraic()}:")
            move_strs = self.format_move_list(moves)
            for move_str in move_strs:
                print(f"  {move_str}")
            
            # Ask user to select a move
            print("\nEnter move number or full move (e.g., 'e4' or '1'):")
            choice = input("> ").strip()
            
            if choice.isdigit():
                move_idx = int(choice) - 1
                if 0 <= move_idx < len(moves):
                    selected_move = moves[move_idx]
                else:
                    print("Invalid move number")
                    return False
            else:
                to_pos = self.parse_position(choice)
                if not to_pos:
                    print("Invalid destination")
                    return False
                
                # Find the move in the list
                selected_move = None
                for move in moves:
                    if move.to_pos == to_pos:
                        selected_move = move
                        break
                
                if not selected_move:
                    print(f"Illegal move: {from_pos.to_algebraic()}-{to_pos.to_algebraic()}")
                    return False
            
            # Handle promotion if needed
            if selected_move.promotion is None and piece.piece_type == PieceType.PAWN:
                promotion_rank = 7 if piece.color == Color.WHITE else 0
                if selected_move.to_pos.rank == promotion_rank:
                    promo_type = self.select_promotion()
                    if not promo_type:
                        return False
                    # Create new move with promotion
                    selected_move = Move(
                        selected_move.from_pos,
                        selected_move.to_pos,
                        promotion=promo_type
                    )
            
            # Make the move
            move_notation = f"{selected_move.from_pos.to_algebraic()}-{selected_move.to_pos.to_algebraic()}"
            if selected_move.promotion:
                move_notation += f"={selected_move.promotion.value}"
            
            if self.game.make_move(selected_move):
                self.move_history.append((
                    self.game.board.fullmove_number,
                    Color.WHITE if self.game.current_player == Color.BLACK else Color.BLACK,
                    move_notation
                ))
                return True
            else:
                print("Failed to make move")
                return False
        
        elif len(parts) == 2:
            # User entered from and to positions
            from_pos = self.parse_position(parts[0])
            to_pos = self.parse_position(parts[1])
            
            if not from_pos or not to_pos:
                print(f"Invalid position format")
                return False
            
            piece = self.game.board.get_piece(from_pos)
            if not piece:
                print(f"No piece at {from_pos.to_algebraic()}")
                return False
            
            if piece.color != self.game.current_player:
                print(f"That's not your piece! It's {self.game.current_player.name}'s turn")
                return False
            
            # Find the matching move
            moves = self.get_moves_from_position(from_pos)
            selected_move = None
            for move in moves:
                if move.to_pos == to_pos:
                    selected_move = move
                    break
            
            if not selected_move:
                print(f"Illegal move: {from_pos.to_algebraic()}-{to_pos.to_algebraic()}")
                # Show legal moves from that square
                if moves:
                    print("Legal moves from that square:")
                    for move_str in self.format_move_list(moves)[:5]:
                        print(f"  {move_str}")
                    if len(moves) > 5:
                        print(f"  ... and {len(moves)-5} more")
                return False
            
            # Handle promotion if needed
            if selected_move.promotion is None and piece.piece_type == PieceType.PAWN:
                promotion_rank = 7 if piece.color == Color.WHITE else 0
                if to_pos.rank == promotion_rank:
                    promo_type = self.select_promotion()
                    if not promo_type:
                        return False
                    # Find the move with this promotion
                    for move in moves:
                        if move.to_pos == to_pos and move.promotion == promo_type:
                            selected_move = move
                            break
            
            # Make the move
            move_notation = f"{selected_move.from_pos.to_algebraic()}-{selected_move.to_pos.to_algebraic()}"
            if selected_move.promotion:
                move_notation += f"={selected_move.promotion.value}"
            
            if self.game.make_move(selected_move):
                self.move_history.append((
                    self.game.board.fullmove_number,
                    Color.WHITE if self.game.current_player == Color.BLACK else Color.BLACK,
                    move_notation
                ))
                return True
            else:
                print("Failed to make move")
                return False
        
        else:
            print("Invalid move format. Use 'e2 e4' or 'e2-e4' or just 'e2' to see moves")
            return False
    
    def play(self):
        """Main game loop"""
        print("="*50)
        print("EXTINCTION CHESS - 2 PLAYER MODE")
        print("="*50)
        print("Type 'help' for commands")
        print()
        
        self.display.display()
        
        while not self.game.game_over:
            # Show current player and get input
            player_color = self.game.current_player.name
            print(f"\n{player_color}'s turn")
            
            # Check for endangered pieces
            endangered = self.game.get_endangered_pieces(self.game.current_player)
            if endangered:
                print(f"⚠️  WARNING: Your {', '.join([p.value for p in endangered])} endangered!")
            
            command = input(f"{player_color}> ").strip().lower()
            
            if not command:
                continue
            
            # Store command in history
            self.command_history.append(command)
            
            # Handle special commands
            if command in ['quit', 'exit']:
                print("Thanks for playing!")
                return
            
            elif command == 'help':
                self.show_help()
            
            elif command == 'rules':
                self.show_rules()
            
            elif command == 'board':
                self.display.display()
            
            elif command == 'clear':
                self.clear_screen()
                self.display.display()
            
            elif command == 'moves':
                all_moves = self.game.get_legal_moves()
                print(f"\nAll legal moves ({len(all_moves)} total):")
                
                # Group by piece position
                by_position = {}
                for move in all_moves:
                    from_str = move.from_pos.to_algebraic()
                    if from_str not in by_position:
                        by_position[from_str] = []
                    by_position[from_str].append(move)
                
                for from_str, moves in sorted(by_position.items()):
                    piece = self.game.board.get_piece(self.parse_position(from_str))
                    print(f"\n{piece.piece_type.value} at {from_str}:")
                    for move in moves[:3]:
                        to_str = move.to_pos.to_algebraic()
                        target = self.game.board.get_piece(move.to_pos)
                        if target:
                            print(f"  → {to_str} (captures {target.piece_type.value})")
                        elif move.is_castling:
                            print(f"  → {to_str} (castle)")
                        elif move.is_en_passant:
                            print(f"  → {to_str} (en passant)")
                        else:
                            print(f"  → {to_str}")
                    if len(moves) > 3:
                        print(f"  ... and {len(moves)-3} more")
            
            elif command == 'counts':
                self.show_piece_counts()
            
            elif command == 'history':
                self.show_move_history()
            
            elif command == 'resign':
                confirm = input("Are you sure you want to resign? (yes/no): ").lower()
                if confirm == 'yes':
                    self.game.game_over = True
                    self.game.winner = Color.BLACK if self.game.current_player == Color.WHITE else Color.WHITE
                    print(f"{self.game.current_player.name} resigns!")
                    break
            
            elif command == 'draw':
                print("Draw offer - both players must agree")
                confirm = input("Do both players agree to a draw? (yes/no): ").lower()
                if confirm == 'yes':
                    self.game.game_over = True
                    self.game.draw_reason = "By agreement"
                    break
            
            elif command == 'fen':
                from extinction_chess_display import FENDisplay
                fen_display = FENDisplay(self.game)
                fen_display.display()
            
            elif command == 'undo':
                print("Undo not implemented yet")
                # TODO: Implement undo functionality
            
            else:
                # Try to parse as a move
                if self.handle_move_input(command):
                    # Move was successful, clear and redraw
                    self.clear_screen()
                    self.display.display()
                    
                    # Show last move
                    if self.move_history:
                        last_move = self.move_history[-1]
                        print(f"Last move: {last_move[1].name} played {last_move[2]}")
        
        # Game over
        print("\n" + "="*50)
        self.display.display()
        
        if self.game.winner:
            print(f"\n🏆 {self.game.winner.name} WINS by extinction!")
            
            # Show what went extinct
            loser = Color.BLACK if self.game.winner == Color.WHITE else Color.WHITE
            extinct = self.game.get_extinct_pieces(loser)
            if extinct:
                print(f"{loser.name} lost all: {', '.join([p.value for p in extinct])}")
        else:
            print(f"\n🤝 Game drawn: {self.game.draw_reason}")
        
        print("="*50)
        print("\nThanks for playing Extinction Chess!")


def main():
    """Entry point for interactive play"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Extinction Chess interactively')
    parser.add_argument('--ascii', action='store_true', 
                       help='Use ASCII pieces instead of Unicode')
    
    args = parser.parse_args()
    
    game = InteractiveGame(use_unicode=not args.ascii)
    
    try:
        game.play()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Thanks for playing!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()