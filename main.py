from src.extinction_chess import ExtinctionChess
from src.extinction_chess_display import ConsoleDisplay, CompactDisplay

# Create game instance
game = ExtinctionChess()

# Create display instance
display = ConsoleDisplay(game)

# Game loop
while not game.game_over:
    display.display()
    display.display_legal_moves()
    
    # Get move (from user input or AI)
    moves = game.get_legal_moves()
    if moves:
        # For now, just take first legal move
        game.make_move(moves[0])
    else:
        break

# Final display
display.display()