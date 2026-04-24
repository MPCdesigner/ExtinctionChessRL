"""
Play against your trained AI model
"""

import pygame
import sys
import os
import threading
from typing import Optional
from extinction_chess import ExtinctionChess, Position, Move, Color, PieceType
from extinction_chess_gui import ChessGUI, SIDEBAR_X
from simple_evaluator import SimpleNeuralNetwork, NetworkConfig, PositionEvaluator
from self_play_trainer import SelfPlayAgent
from create_v4_model import ExtinctionFocusedEvaluator


class PlayVsAI(ChessGUI):
    """Modified GUI to play against AI"""
    
    def __init__(self, game: ExtinctionChess, ai_agent: SelfPlayAgent, human_color: Color = Color.WHITE):
        super().__init__(game)
        self.ai_agent = ai_agent
        self.human_color = human_color
        self.ai_color = Color.BLACK if human_color == Color.WHITE else Color.WHITE
        self.ai_thinking = False
        self.ai_move_result = None
        self.last_ai_eval = 0
        
    def handle_click(self, pos):
        """Handle mouse click - only for human player"""
        if self.game.game_over:
            print(f"[DEBUG] Click blocked: game_over=True, winner={self.game.winner}")
            return

        # Only allow human to move on their turn
        if self.game.current_player != self.human_color:
            print(f"[DEBUG] Click blocked: current_player={self.game.current_player}, human={self.human_color}")
            return

        # Use parent class click handling
        super().handle_click(pos)
    
    def start_ai_move(self):
        """Start AI thinking in a background thread."""
        if self.ai_thinking or self.game.current_player != self.ai_color or self.game.game_over:
            return
        self.ai_thinking = True
        self.ai_move_result = None
        thread = threading.Thread(target=self._ai_think, daemon=True)
        thread.start()

    def _ai_think(self):
        """Background thread: compute the AI's move."""
        eval_score = self.ai_agent.evaluator.evaluate(self.game)
        move = self.ai_agent.select_move(self.game, temperature=0)
        self.ai_move_result = (move, eval_score)

    def apply_ai_move(self):
        """Apply the AI's move once thinking is done (called from main loop)."""
        if not self.ai_move_result:
            return
        move, eval_score = self.ai_move_result
        self.ai_move_result = None

        if move:
            self.last_ai_eval = eval_score
            self.game.make_move(move)
            self.last_move = move
            print(f"AI moves: {move}")
            print(f"AI evaluation: {self.last_ai_eval:.3f}")

        self.ai_thinking = False
        self.selected_square = None
        self.legal_moves = []
    
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
            self.screen.blit(text, (SIDEBAR_X, y))
            
            # Show AI's evaluation
            eval_text = f"AI eval: {self.last_ai_eval:.3f}"
            text = self.small_font.render(eval_text, True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y + 25))
    
    def run(self):
        """Modified game loop with AI moves"""
        running = True
        clock = pygame.time.Clock()

        # Make first AI move if AI is white
        if self.ai_color == Color.WHITE:
            self.start_ai_move()

        while running:
            # Check if AI finished thinking
            if self.ai_thinking and self.ai_move_result is not None:
                self.apply_ai_move()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        if not self.ai_thinking:
                            old_player = self.game.current_player
                            self.handle_click(pygame.mouse.get_pos())

                            # If player changed (move was made), let AI move
                            if self.game.current_player != old_player:
                                self.start_ai_move()

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
                            self.start_ai_move()

                    elif event.key == pygame.K_r:
                        self.ai_thinking = False
                        self.ai_move_result = None
                        self.reset_game()
                        if self.ai_color == Color.WHITE:
                            self.start_ai_move()

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


class AlphaZeroAgent:
    """Agent that uses MCTS with AlphaZero network for move selection."""

    def __init__(self, evaluator, num_simulations=200, tactical_shortcuts=True):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.tactical_shortcuts = tactical_shortcuts

    def select_move(self, game, temperature=0):
        from alphazero import mcts_search
        move_visits, _ = mcts_search(
            game, self.evaluator,
            num_simulations=self.num_simulations,
            dirichlet_alpha=0, noise_weight=0,  # no noise for play
            tactical_shortcuts=self.tactical_shortcuts,
        )
        if not move_visits:
            return None
        if temperature == 0:
            return max(move_visits, key=lambda x: x[1])[0]
        import numpy as np
        moves, counts = zip(*move_visits)
        counts = np.array(counts, dtype=np.float64)
        probs = counts / counts.sum()
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]


def main():
    """Main function to play against AI"""
    import argparse

    parser = argparse.ArgumentParser(description='Play Extinction Chess against your AI')
    parser.add_argument('--model', type=str, default='models/az_latest.pt',
                       help='Path to model file')
    parser.add_argument('--color', type=str, default='white',
                       help='Your color: white or black (default: white)')
    parser.add_argument('--sims', type=int, default=200,
                       help='MCTS simulations per move (default: 200)')
    parser.add_argument('--depth', type=int, default=0,
                       help='Search depth for minimax (legacy models only)')
    parser.add_argument('--version', type=int, default=None,
                       help='Model version to load (legacy)')
    parser.add_argument('--no-shortcuts', action='store_true',
                       help='Disable tactical shortcuts (test raw network)')

    args = parser.parse_args()

    # Determine model path
    if args.version:
        import glob
        model_path = f"models/versions/model_v{args.version}_*.json"
        matches = glob.glob(model_path)
        model_path = matches[0] if matches else args.model
    else:
        model_path = args.model

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run training first or specify a valid model path")
        return

    print(f"Loading AI model from {model_path}...")

    if model_path.endswith('.pt'):
        import torch
        data = torch.load(model_path, weights_only=False, map_location="cpu")
        meta = data.get("metadata", {})

        if meta.get("model_type") == "alphazero":
            from alphazero import AlphaZeroNet, AlphaZeroEvaluator
            model, meta = AlphaZeroNet.load_checkpoint(model_path)
            evaluator = AlphaZeroEvaluator(model, device="cpu")
            ai_agent = AlphaZeroAgent(evaluator, num_simulations=args.sims,
                                     tactical_shortcuts=not args.no_shortcuts)
            print(f"Loaded AlphaZero model (iter {meta.get('iteration', '?')}, "
                  f"wr={meta.get('win_rate', '?')})")
            print(f"Using MCTS with {args.sims} simulations/move")
        elif meta.get("model_type") == "cnn":
            from torch_model import ChessCNN, CNNEvaluator
            model, meta = ChessCNN.load_checkpoint(model_path)
            evaluator = CNNEvaluator(model, device="cpu")
            print(f"Loaded CNN model (meta: {meta})")
            if args.depth > 0:
                from torch_model import SearchAgent
                ai_agent = SearchAgent(evaluator, depth=args.depth)
            else:
                ai_agent = SelfPlayAgent(evaluator, exploration_rate=0)
        else:
            from torch_model import ChessNet, TorchEvaluator
            model, meta = ChessNet.load_checkpoint(model_path, input_size=39, hidden_sizes=[256, 128, 64])
            evaluator = TorchEvaluator(model, device="cpu")
            print(f"Loaded MLP model (meta: {meta})")
            if args.depth > 0:
                from torch_model import SearchAgent
                ai_agent = SearchAgent(evaluator, depth=args.depth)
            else:
                ai_agent = SelfPlayAgent(evaluator, exploration_rate=0)
    else:
        config = NetworkConfig(input_size=39, hidden_sizes=[128, 64, 32])
        network = SimpleNeuralNetwork(config)
        network.load(model_path)
        if args.version == 4:
            evaluator = ExtinctionFocusedEvaluator(network)
        else:
            evaluator = PositionEvaluator(network)
        ai_agent = SelfPlayAgent(evaluator, exploration_rate=0)

    human_color = Color.WHITE if args.color.lower() == 'white' else Color.BLACK
    ai_color_name = "Black" if human_color == Color.WHITE else "White"

    print(f"\nYou: {args.color.upper()} | AI: {ai_color_name}")
    print("Click to select/move | Right-click to deselect | R to reset | ESC to quit\n")

    pygame.init()
    game = ExtinctionChess()
    gui = PlayVsAI(game, ai_agent, human_color)
    gui.run()


if __name__ == "__main__":
    main()