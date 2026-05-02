"""
Play against your trained AI model
"""

import glob
import pygame
import sys
import os
import threading
from typing import Optional
from extinction_chess import ExtinctionChess, Position, Move, Color, PieceType
from extinction_chess_gui import ChessGUI, SIDEBAR_X

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 750
SIM_OPTIONS = [5, 10, 20, 50, 100, 200, 400]
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)


def discover_models(models_dir):
    """Find all .pt files in models directory."""
    models = []
    for path in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
        name = os.path.basename(path)
        try:
            import torch
            data = torch.load(path, weights_only=False, map_location="cpu")
            meta = data.get("metadata", {})
            iteration = meta.get("iteration", "?")
            label = f"iter {iteration}"
        except Exception:
            label = name
        models.append({"path": path, "name": name, "label": label})
    return models


class PlayMenuScreen:
    """Menu for selecting model, sims, shortcuts, and color."""

    def __init__(self, screen, models):
        self.screen = screen
        self.models = models
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 56)

        self.model_idx = len(models) - 1 if models else 0
        self.sim_idx = 5  # default to 200
        self.shortcuts = True
        self.color = "white"

        self.active = "model"

    def run(self):
        if not self.models:
            return None

        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_TAB or event.key == pygame.K_DOWN:
                        self._next_field()
                    elif event.key in (pygame.K_UP,):
                        self._prev_field()
                    elif event.key == pygame.K_LEFT:
                        self._adjust(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._adjust(1)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if self.active == "start":
                            return {
                                "model": self.models[self.model_idx]["path"],
                                "sims": SIM_OPTIONS[self.sim_idx],
                                "shortcuts": self.shortcuts,
                                "color": self.color,
                            }
                        else:
                            self._next_field()

            self._draw()
            pygame.display.flip()
            clock.tick(30)

    def _next_field(self):
        order = ["model", "sims", "shortcuts", "color", "start"]
        idx = order.index(self.active)
        self.active = order[(idx + 1) % len(order)]

    def _prev_field(self):
        order = ["model", "sims", "shortcuts", "color", "start"]
        idx = order.index(self.active)
        self.active = order[(idx - 1) % len(order)]

    def _adjust(self, delta):
        if self.active == "model":
            self.model_idx = (self.model_idx + delta) % len(self.models)
        elif self.active == "sims":
            self.sim_idx = max(0, min(len(SIM_OPTIONS) - 1, self.sim_idx + delta))
        elif self.active == "shortcuts":
            self.shortcuts = not self.shortcuts
        elif self.active == "color":
            self.color = "black" if self.color == "white" else "white"

    def _draw(self):
        self.screen.fill((240, 240, 245))
        cx = WINDOW_WIDTH // 2
        y = 80

        text = self.title_font.render("Play vs AI", True, BLACK_COLOR)
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 70

        text = self.small_font.render(
            "Use LEFT/RIGHT to change, TAB/DOWN to next field, SPACE/ENTER to confirm",
            True, (100, 100, 100))
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 50

        self._draw_selector(y, "Model:", self.models[self.model_idx]["label"],
                           self.active == "model", (0, 0, 180))
        y += 70

        self._draw_selector(y, "Simulations:", str(SIM_OPTIONS[self.sim_idx]),
                           self.active == "sims", BLACK_COLOR)
        y += 70

        shortcuts_label = "ON" if self.shortcuts else "OFF"
        shortcuts_color = (0, 130, 0) if self.shortcuts else (180, 0, 0)
        self._draw_selector(y, "Tactical Shortcuts:", shortcuts_label,
                           self.active == "shortcuts", shortcuts_color)
        y += 70

        color_display = self.color.upper()
        color_val = (0, 0, 180) if self.color == "white" else (180, 0, 0)
        self._draw_selector(y, "Your Color:", color_display,
                           self.active == "color", color_val)
        y += 90

        btn_w, btn_h = 250, 50
        btn_rect = pygame.Rect(cx - btn_w // 2, y, btn_w, btn_h)
        btn_color = (0, 150, 0) if self.active == "start" else (100, 180, 100)
        border_color = (0, 100, 0) if self.active == "start" else (80, 130, 80)
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, btn_rect, 3, border_radius=8)
        text = self.large_font.render("Start", True, WHITE_COLOR)
        self.screen.blit(text, text.get_rect(center=btn_rect.center))

    def _draw_selector(self, y, label, value, active, value_color):
        cx = WINDOW_WIDTH // 2
        box_w = 500
        box_h = 55
        box_rect = pygame.Rect(cx - box_w // 2, y - 5, box_w, box_h)

        if active:
            pygame.draw.rect(self.screen, (220, 225, 255), box_rect, border_radius=6)
            pygame.draw.rect(self.screen, (0, 0, 180), box_rect, 2, border_radius=6)
        else:
            pygame.draw.rect(self.screen, (250, 250, 250), box_rect, border_radius=6)
            pygame.draw.rect(self.screen, (180, 180, 180), box_rect, 1, border_radius=6)

        text = self.font.render(label, True, (60, 60, 60))
        self.screen.blit(text, (cx - box_w // 2 + 15, y + 2))

        arrow_left = self.font.render("<", True, (100, 100, 100) if active else (200, 200, 200))
        arrow_right = self.font.render(">", True, (100, 100, 100) if active else (200, 200, 200))
        val_text = self.large_font.render(value, True, value_color)

        val_x = cx + 80
        self.screen.blit(arrow_left, (val_x - 80, y + 5))
        self.screen.blit(val_text, val_text.get_rect(center=(val_x, y + 22)))
        self.screen.blit(arrow_right, (val_x + 80, y + 5))


class PlayVsAI(ChessGUI):
    """Modified GUI to play against AI"""
    
    def __init__(self, game: ExtinctionChess, ai_agent, human_color: Color = Color.WHITE):
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
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (skip menu)')
    parser.add_argument('--color', type=str, default='white',
                       help='Your color: white or black (default: white)')
    parser.add_argument('--sims', type=int, default=200,
                       help='MCTS simulations per move (default: 200)')
    parser.add_argument('--no-shortcuts', action='store_true',
                       help='Disable tactical shortcuts')

    args = parser.parse_args()

    pygame.init()

    if args.model:
        # Direct mode — skip menu
        model_path = args.model
        sims = args.sims
        shortcuts = not args.no_shortcuts
        human_color_str = args.color.lower()
    else:
        # Show menu
        print("Scanning for models...", flush=True)
        models = discover_models(MODELS_DIR)
        if not models:
            print(f"No .pt files found in {MODELS_DIR}")
            sys.exit(1)
        print(f"Found {len(models)} models", flush=True)

        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Play vs AI — Settings")
        menu = PlayMenuScreen(screen, models)
        config = menu.run()
        if config is None:
            pygame.quit()
            sys.exit(0)

        model_path = config["model"]
        sims = config["sims"]
        shortcuts = config["shortcuts"]
        human_color_str = config["color"]

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run training first or specify a valid model path")
        return

    print(f"Loading AI model from {model_path}...")

    from alphazero import AlphaZeroNet, AlphaZeroEvaluator
    model, meta = AlphaZeroNet.load_checkpoint(model_path)
    evaluator = AlphaZeroEvaluator(model, device="cpu")
    ai_agent = AlphaZeroAgent(evaluator, num_simulations=sims,
                             tactical_shortcuts=shortcuts)
    print(f"Loaded AlphaZero model (iter {meta.get('iteration', '?')}, "
          f"wr={meta.get('win_rate', '?')})")
    print(f"Using MCTS with {sims} simulations/move, "
          f"tactical shortcuts {'ON' if shortcuts else 'OFF'}")

    human_color = Color.WHITE if human_color_str == 'white' else Color.BLACK
    ai_color_name = "Black" if human_color == Color.WHITE else "White"

    print(f"\nYou: {human_color_str.upper()} | AI: {ai_color_name}")
    print("Click to select/move | Right-click to deselect | R to reset | ESC to quit\n")

    game = ExtinctionChess()
    gui = PlayVsAI(game, ai_agent, human_color)
    gui.run()


if __name__ == "__main__":
    main()