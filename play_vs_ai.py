"""
Play against your trained AI model
"""

import pygame
import sys
import os
from typing import Optional
from extinction_chess import ExtinctionChess, Position, Move, Color, PieceType
from extinction_chess_gui import ChessGUI
from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator
from self_play_trainer import SelfPlayAgent


class PlayVsAI(ChessGUI):
    """Modified GUI to play against AI"""
    
    def __init__(self, game: ExtinctionChess, ai_agent: SelfPlayAgent, human_color: Color = Color.WHITE):
        super().__init__(game)
        self.ai_agent = ai_agent
        self.human_color = human_color
        self.ai_color = Color.BLACK if human_color == Color.WHITE else Color.WHITE
        self.ai_thinking = False
        self.last_ai_eval = 0
        
    def handle_click(self, pos):
        """Handle mouse click - only for human player"""
        if self.game.game_over:
            return
        
        # Only allow human to move on their turn
        if self.game.current_player != self.human_color:
            return
        
        # Use parent class click handling
        super().handle_click(pos)
    
    def make_ai_move(self):
        """Let the AI make its move"""
        if self.game.current_player == self.ai_color and not self.game.game_over:
            # Get AI's move
            self.ai_thinking = True
            move = self.ai_agent.select_move(self.game, temperature=0)  # Deterministic
            
            if move:
                # Get evaluation before move
                self.last_ai_eval = self.ai_agent.evaluator.evaluate(self.game)
                
                # Make the move
                self.game.make_move(move)
                self.last_move = move
                
                print(f"AI moves: {move}")
                print(f"AI evaluation: {self.last_ai_eval:.3f}")
            
            self.ai_thinking = False
    
    def draw_sidebar(self):
        """Enhanced sidebar with AI info"""
        super().draw_sidebar()
        
        # Add AI status
        if self.game.current_player == self.ai_color and not self.game.game_over:
            y = 400
            if self.ai_thinking:
                text = self.font.render("AI is thinking...", True, (0, 100, 200))
            else:
                text = self.font.render("AI's turn", True, (0, 100, 200))
            self.screen.blit(text, (self.sidebar_x, y))
            
            # Show AI's evaluation
            eval_text = f"AI eval: {self.last_ai_eval:.3f}"
            text = self.small_font.render(eval_text, True, (100, 100, 100))
            self.screen.blit(text, (self.sidebar_x, y + 25))
    
    def run(self):
        """Modified game loop with AI moves"""
        running = True
        clock = pygame.time.Clock()
        
        # Make first AI move if AI is white
        if self.ai_color == Color.WHITE:
            self.make_ai_move()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        old_player = self.game.current_player
                        self.handle_click(pygame.mouse.get_pos())
                        
                        # If player changed (move was made), let AI move
                        if self.game.current_player != old_player:
                            pygame.display.flip()  # Show the human move first
                            pygame.time.wait(500)  # Small delay so you can see the move
                            self.make_ai_move()
                    
                    elif event.button == 3:  # Right click
                        if self.game.current_player == self.human_color:
                            self.selected_square = None
                            self.legal_moves = []
                
                elif event.type == pygame.KEYDOWN:
                    if self.promotion_pending and self.game.current_player == self.human_color:
                        old_player = self.game.current_player
                        self.handle_promotion_key(event.key)
                        
                        # If promotion completed, let AI move
                        if not self.promotion_pending and self.game.current_player != old_player:
                            self.make_ai_move()
                    
                    elif event.key == pygame.K_r:
                        self.reset_game()
                        if self.ai_color == Color.WHITE:
                            self.make_ai_move()
                    
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Draw everything
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_sidebar()
            self.draw_promotion_dialog()
            
            # Update display
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()


def main():
    """Main function to play against AI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Extinction Chess against your AI')
    parser.add_argument('--model', type=str, default='models/final_model.json',
                       help='Path to model file (default: models/final_model.json)')
    parser.add_argument('--color', type=str, default='white',
                       help='Your color: white or black (default: white)')
    parser.add_argument('--version', type=int, default=None,
                       help='Model version to load (1, 2, etc.)')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.version:
        # Try to find versioned model
        model_path = f"models/versions/model_v{args.version}_*.json"
        import glob
        matches = glob.glob(model_path)
        if matches:
            model_path = matches[0]
            print(f"Loading model: {model_path}")
        else:
            print(f"Version {args.version} not found, using default")
            model_path = args.model
    else:
        model_path = args.model
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("\nAvailable models:")
        
        # Check for versioned models
        if os.path.exists("models/versions"):
            import glob
            versions = glob.glob("models/versions/model_v*.json")
            for v in versions:
                print(f"  - {v}")
        
        # Check for final model
        if os.path.exists("models/final_model.json"):
            print("  - models/final_model.json")
        
        print("\nRun training first or specify a valid model path")
        return
    
    # Load the model
    print(f"Loading AI model from {model_path}...")
    config = NetworkConfig(input_size=39, hidden_sizes=[128, 64, 32])
    network = SimpleNeuralNetwork(config)
    network.load(model_path)
    
    # Create AI agent
    evaluator = PositionEvaluator(network)
    ai_agent = SelfPlayAgent(evaluator, exploration_rate=0)  # No exploration during play
    
    # Determine colors
    human_color = Color.WHITE if args.color.lower() == 'white' else Color.BLACK
    ai_color_name = "Black" if human_color == Color.WHITE else "White"
    
    print(f"\nStarting game!")
    print(f"You are playing as {args.color.upper()}")
    print(f"AI is playing as {ai_color_name}")
    print("\nControls:")
    print("  - Click piece to select")
    print("  - Click square to move")
    print("  - Right-click to deselect")
    print("  - R to reset game")
    print("  - ESC to quit")
    print("\nGood luck!\n")
    
    # Initialize Pygame
    pygame.init()
    
    # Create game and GUI
    game = ExtinctionChess()
    gui = PlayVsAI(game, ai_agent, human_color)
    
    # Run the game
    gui.run()


if __name__ == "__main__":
    main()