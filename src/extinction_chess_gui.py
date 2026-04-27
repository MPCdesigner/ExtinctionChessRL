"""
Pygame GUI for Extinction Chess
Visual interface for testing and playing
"""

import pygame
import sys
import os
from typing import Optional, List, Tuple, Set
from enum import Enum
from extinction_chess import (
    ExtinctionChess, Position, Move, Color, PieceType,
    Board, Piece
)

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_OFFSET_X = 30
BOARD_OFFSET_Y = 30
SIDEBAR_X = BOARD_OFFSET_X + BOARD_SIZE + 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Yellow with transparency
LEGAL_MOVE_COLOR = (0, 255, 0, 64)    # Green with transparency
CAPTURE_MOVE_COLOR = (255, 0, 0, 64)  # Red with transparency
ENDANGERED_COLOR = (255, 100, 0)      # Orange for endangered pieces
LAST_MOVE_COLOR = (100, 100, 255, 64) # Blue for last move

class ChessGUI:
    def __init__(self, game: ExtinctionChess):
        self.game = game
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Extinction Chess")
        
        # GUI state
        self.selected_square = None
        self.legal_moves = []
        self.highlighted_squares = set()
        self.last_move = None
        self.promotion_pending = None
        self.promotion_from = None
        self.promotion_to = None
        self.message = ""
        self.message_timer = 0
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 36)
        self.piece_font = pygame.font.Font(None, int(SQUARE_SIZE * 0.7))
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
    
    def get_square_from_mouse(self, pos: Tuple[int, int]) -> Optional[Position]:
        """Convert mouse position to board square"""
        x, y = pos
        if (BOARD_OFFSET_X <= x < BOARD_OFFSET_X + BOARD_SIZE and
            BOARD_OFFSET_Y <= y < BOARD_OFFSET_Y + BOARD_SIZE):
            file = (x - BOARD_OFFSET_X) // SQUARE_SIZE
            rank = 7 - (y - BOARD_OFFSET_Y) // SQUARE_SIZE
            return Position(rank, file)
        return None
    
    def draw_board(self):
        """Draw the chess board"""
        for rank in range(8):
            for file in range(8):
                x = BOARD_OFFSET_X + file * SQUARE_SIZE
                y = BOARD_OFFSET_Y + (7 - rank) * SQUARE_SIZE
                
                # Draw square
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Draw coordinates
                if file == 0:
                    label = self.small_font.render(str(rank + 1), True, BLACK)
                    self.screen.blit(label, (x + 2, y + 2))
                if rank == 0:
                    label = self.small_font.render(chr(ord('a') + file), True, BLACK)
                    self.screen.blit(label, (x + SQUARE_SIZE - 15, y + SQUARE_SIZE - 18))
    
    def draw_highlights(self):
        """Draw highlighted squares (selection, legal moves, etc.)"""
        # Create transparent surface for overlays
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        
        # Highlight last move
        if self.last_move:
            for pos in [self.last_move.from_pos, self.last_move.to_pos]:
                x = pos.file * SQUARE_SIZE
                y = (7 - pos.rank) * SQUARE_SIZE
                pygame.draw.rect(overlay, LAST_MOVE_COLOR, (x, y, SQUARE_SIZE, SQUARE_SIZE))
        
        # Highlight selected square
        if self.selected_square:
            x = self.selected_square.file * SQUARE_SIZE
            y = (7 - self.selected_square.rank) * SQUARE_SIZE
            pygame.draw.rect(overlay, HIGHLIGHT_COLOR, (x, y, SQUARE_SIZE, SQUARE_SIZE))
        
        # Highlight legal moves (but only show each square once for promotions)
        seen_squares = set()
        for move in self.legal_moves:
            if move.to_pos not in seen_squares:
                seen_squares.add(move.to_pos)
                x = move.to_pos.file * SQUARE_SIZE
                y = (7 - move.to_pos.rank) * SQUARE_SIZE
                
                if self.game.board.get_piece(move.to_pos):
                    # Capture move
                    pygame.draw.rect(overlay, CAPTURE_MOVE_COLOR, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                else:
                    # Regular move - draw circle
                    pygame.draw.circle(overlay, LEGAL_MOVE_COLOR, 
                                     (x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2), 
                                     SQUARE_SIZE // 6)
        
        self.screen.blit(overlay, (BOARD_OFFSET_X, BOARD_OFFSET_Y))
    
    def draw_pieces(self):
        """Draw all pieces on the board using letters"""
        endangered_pieces = {
            Color.WHITE: self.game.get_endangered_pieces(Color.WHITE),
            Color.BLACK: self.game.get_endangered_pieces(Color.BLACK)
        }
        
        # Piece letters
        piece_chars = {
            PieceType.KING: 'K',
            PieceType.QUEEN: 'Q',
            PieceType.ROOK: 'R',
            PieceType.BISHOP: 'B',
            PieceType.KNIGHT: 'N',
            PieceType.PAWN: 'P'
        }
        
        for rank in range(8):
            for file in range(8):
                pos = Position(rank, file)
                piece = self.game.board.get_piece(pos)
                if piece:
                    x = BOARD_OFFSET_X + file * SQUARE_SIZE
                    y = BOARD_OFFSET_Y + (7 - rank) * SQUARE_SIZE
                    
                    # Draw endangered border
                    if piece.piece_type in endangered_pieces[piece.color]:
                        pygame.draw.rect(self.screen, ENDANGERED_COLOR, 
                                       (x, y, SQUARE_SIZE, SQUARE_SIZE), 3)
                    
                    # Get piece character
                    char = piece_chars[piece.piece_type]
                    # Lowercase for black pieces
                    if piece.color == Color.BLACK:
                        char = char.lower()
                    
                    # Draw piece letter
                    # White pieces: white text with black outline
                    # Black pieces: black text
                    if piece.color == Color.WHITE:
                        # Draw black outline
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    text = self.piece_font.render(char, True, BLACK)
                                    text_rect = text.get_rect(center=(x + SQUARE_SIZE // 2 + dx, 
                                                                     y + SQUARE_SIZE // 2 + dy))
                                    self.screen.blit(text, text_rect)
                        # Draw white piece
                        text = self.piece_font.render(char, True, WHITE)
                    else:
                        # Draw black piece
                        text = self.piece_font.render(char, True, BLACK)
                    
                    text_rect = text.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                    self.screen.blit(text, text_rect)
    
    def draw_sidebar(self):
        """Draw game information on the sidebar"""
        y = BOARD_OFFSET_Y
        
        # Current player
        player_text = f"{self.game.current_player.name}'s Turn"
        text = self.large_font.render(player_text, True, BLACK)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 50
        
        # Piece counts
        for color in [Color.WHITE, Color.BLACK]:
            counts = self.game.board.get_piece_count(color)
            endangered = self.game.get_endangered_pieces(color)
            extinct = self.game.get_extinct_pieces(color)
            
            color_name = color.name
            text = self.font.render(f"{color_name}:", True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 30
            
            for piece_type in PieceType:
                count = counts[piece_type]
                status = ""
                status_color = BLACK
                
                if piece_type in extinct:
                    status = " [EXTINCT]"
                    status_color = (255, 0, 0)
                elif piece_type in endangered:
                    status = " [ENDANGERED]"
                    status_color = ENDANGERED_COLOR
                
                text_str = f"  {piece_type.value}: {count}{status}"
                text = self.small_font.render(text_str, True, status_color)
                self.screen.blit(text, (SIDEBAR_X, y))
                y += 20
            
            y += 10
        
        # Game status
        if self.game.game_over:
            y += 20
            if self.game.winner:
                text = self.large_font.render("GAME OVER", True, (255, 0, 0))
                self.screen.blit(text, (SIDEBAR_X, y))
                y += 40
                text = self.font.render(f"{self.game.winner.name} WINS!", True, BLACK)
                self.screen.blit(text, (SIDEBAR_X, y))
            else:
                text = self.large_font.render("DRAW", True, (100, 100, 100))
                self.screen.blit(text, (SIDEBAR_X, y))
                y += 40
                if self.game.draw_reason:
                    text = self.small_font.render(self.game.draw_reason, True, BLACK)
                    self.screen.blit(text, (SIDEBAR_X, y))
        
        # Controls help
        y = WINDOW_HEIGHT - 100
        help_text = [
            "Click piece to select",
            "Click square to move",
            "Right-click to deselect",
            "R - Reset game",
            "ESC - Quit"
        ]
        for line in help_text:
            text = self.small_font.render(line, True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 18
        
        # Message
        if self.message and self.message_timer > 0:
            text = self.font.render(self.message, True, (200, 0, 0))
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 30))
            self.screen.blit(text, text_rect)
            self.message_timer -= 1
    
    def draw_promotion_dialog(self):
        """Draw promotion selection dialog"""
        if not self.promotion_pending:
            return
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        
        # Draw dialog box
        dialog_width = 400
        dialog_height = 200
        dialog_x = (WINDOW_WIDTH - dialog_width) // 2
        dialog_y = (WINDOW_HEIGHT - dialog_height) // 2
        
        pygame.draw.rect(self.screen, WHITE, 
                        (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(self.screen, BLACK, 
                        (dialog_x, dialog_y, dialog_width, dialog_height), 3)
        
        # Title
        text = self.large_font.render("Promote Pawn To:", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, dialog_y + 30))
        self.screen.blit(text, text_rect)
        
        # Options
        pieces = [
            (PieceType.QUEEN, 'Q', 'Queen'),
            (PieceType.ROOK, 'R', 'Rook'),
            (PieceType.BISHOP, 'B', 'Bishop'),
            (PieceType.KNIGHT, 'N', 'Knight'),
            (PieceType.KING, 'K', 'King!')
        ]
        
        y = dialog_y + 70
        for piece_type, key, name in pieces:
            text = self.font.render(f"[{key}] {name}", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 25
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click"""
        if self.game.game_over:
            return
        
        if self.promotion_pending:
            return  # Waiting for keyboard input
        
        square = self.get_square_from_mouse(pos)
        if not square:
            return
        
        piece = self.game.board.get_piece(square)
        
        if self.selected_square:
            # Check if this is a valid move destination
            valid_destination = False
            for move in self.legal_moves:
                if move.to_pos == square:
                    valid_destination = True
                    break
            
            if valid_destination:
                # Check if this is a pawn promotion
                moving_piece = self.game.board.get_piece(self.selected_square)
                if moving_piece and moving_piece.piece_type == PieceType.PAWN:
                    promotion_rank = 7 if self.game.current_player == Color.WHITE else 0
                    if square.rank == promotion_rank:
                        # Store promotion info and show dialog
                        self.promotion_pending = True
                        self.promotion_from = self.selected_square
                        self.promotion_to = square
                        return
                
                # Regular move (not promotion)
                for move in self.legal_moves:
                    if move.to_pos == square and move.promotion is None:
                        if self.game.make_move(move):
                            self.last_move = move
                            self.selected_square = None
                            self.legal_moves = []
                            break
            else:
                # Clicked on invalid square - maybe selecting new piece
                if piece and piece.color == self.game.current_player:
                    self.selected_square = square
                    self.legal_moves = self.game.get_legal_moves(square)
                else:
                    self.selected_square = None
                    self.legal_moves = []
        else:
            # No piece selected - select if it's current player's piece
            if piece and piece.color == self.game.current_player:
                self.selected_square = square
                self.legal_moves = self.game.get_legal_moves(square)
    
    def handle_promotion_key(self, key):
        """Handle keyboard input for promotion"""
        if not self.promotion_pending:
            return
        
        promotion_map = {
            pygame.K_q: PieceType.QUEEN,
            pygame.K_r: PieceType.ROOK,
            pygame.K_b: PieceType.BISHOP,
            pygame.K_n: PieceType.KNIGHT,
            pygame.K_k: PieceType.KING
        }
        
        if key in promotion_map:
            # Find the move with the selected promotion
            for move in self.legal_moves:
                if (move.from_pos == self.promotion_from and 
                    move.to_pos == self.promotion_to and 
                    move.promotion == promotion_map[key]):
                    if self.game.make_move(move):
                        self.last_move = move
                    break
            
            # Clear promotion state
            self.promotion_pending = False
            self.promotion_from = None
            self.promotion_to = None
            self.selected_square = None
            self.legal_moves = []
    
    def reset_game(self):
        """Reset the game to initial position"""
        self.game = ExtinctionChess()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.promotion_pending = False
        self.promotion_from = None
        self.promotion_to = None
        self.message = "Game reset!"
        self.message_timer = 60
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(pygame.mouse.get_pos())
                    elif event.button == 3:  # Right click
                        self.selected_square = None
                        self.legal_moves = []
                
                elif event.type == pygame.KEYDOWN:
                    if self.promotion_pending:
                        self.handle_promotion_key(event.key)
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_sidebar()
            self.draw_promotion_dialog()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    """Entry point for GUI"""
    # Check if pygame is installed
    try:
        import pygame
    except ImportError:
        print("pygame is not installed. Install it with: pip install pygame")
        return
    
    # Create game and GUI
    game = ExtinctionChess()
    gui = ChessGUI(game)
    
    # Run the game
    gui.run()


if __name__ == "__main__":
    main()