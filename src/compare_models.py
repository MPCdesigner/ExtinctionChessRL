"""
Compare two AlphaZero checkpoints by having them play against each other.

    python compare_models.py

Opens pygame with a menu to select models and simulations.
Play out an opening position, then press SPACE to watch the models play.
Two games are played from that position (M1 as white, then M1 as black).
After both games, press LEFT/RIGHT to review moves. Press SPACE for next opening.
"""
import argparse
import glob
import os
import sys
import threading
from copy import deepcopy

import pygame

from extinction_chess import ExtinctionChess, Color, PieceType, Position, Move
from extinction_chess_gui import (
    ChessGUI, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE,
    SQUARE_SIZE, WHITE, BLACK,
)
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search

# Wider window to fit sidebar and review panel
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750
SIDEBAR_X = BOARD_OFFSET_X + BOARD_SIZE + 20
PANEL_WIDTH = WINDOW_WIDTH - SIDEBAR_X - 10

SIM_OPTIONS = [5, 10, 20, 50, 100, 200, 400]
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def discover_models(models_dir):
    """Find all .pt files in models directory and load metadata."""
    models = []
    for path in sorted(glob.glob(os.path.join(models_dir, "*.pt"))):
        name = os.path.basename(path)
        try:
            _, meta = AlphaZeroNet.load_checkpoint(path)
            iteration = meta.get("iteration", "?")
            label = f"iter {iteration}"
        except Exception:
            label = name
        models.append({"path": path, "name": name, "label": label})
    return models


class MenuScreen:
    """Model and sim selection menu."""

    def __init__(self, screen, models):
        self.screen = screen
        self.models = models
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 56)

        self.m1_idx = len(models) - 1 if models else 0  # default to newest
        self.m2_idx = 0  # default to oldest
        self.sim_idx = 5  # default to 200

        # Which selector is active: "m1", "m2", "sims", "start"
        self.active = "m1"

    def run(self):
        """Run menu loop, returns (model1_path, model2_path, sims) or None."""
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
                            return (
                                self.models[self.m1_idx]["path"],
                                self.models[self.m2_idx]["path"],
                                SIM_OPTIONS[self.sim_idx],
                            )
                        else:
                            self._next_field()

            self._draw()
            pygame.display.flip()
            clock.tick(30)

    def _next_field(self):
        order = ["m1", "m2", "sims", "start"]
        idx = order.index(self.active)
        self.active = order[(idx + 1) % len(order)]

    def _prev_field(self):
        order = ["m1", "m2", "sims", "start"]
        idx = order.index(self.active)
        self.active = order[(idx - 1) % len(order)]

    def _adjust(self, delta):
        if self.active == "m1":
            self.m1_idx = (self.m1_idx + delta) % len(self.models)
        elif self.active == "m2":
            self.m2_idx = (self.m2_idx + delta) % len(self.models)
        elif self.active == "sims":
            self.sim_idx = max(0, min(len(SIM_OPTIONS) - 1, self.sim_idx + delta))

    def _draw(self):
        self.screen.fill((240, 240, 245))

        cx = WINDOW_WIDTH // 2
        y = 80

        # Title
        text = self.title_font.render("Compare Models", True, BLACK)
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 80

        # Instructions
        text = self.small_font.render(
            "Use LEFT/RIGHT to change, TAB/DOWN to next field, SPACE/ENTER to confirm",
            True, (100, 100, 100))
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 50

        # Model 1
        self._draw_selector(y, "Model 1 (Blue):", self.models[self.m1_idx]["label"],
                           self.active == "m1", (0, 0, 180))
        y += 70

        # Model 2
        self._draw_selector(y, "Model 2 (Red):", self.models[self.m2_idx]["label"],
                           self.active == "m2", (180, 0, 0))
        y += 70

        # Sims
        self._draw_selector(y, "Simulations:", str(SIM_OPTIONS[self.sim_idx]),
                           self.active == "sims", BLACK)
        y += 90

        # Start button
        btn_w, btn_h = 250, 50
        btn_rect = pygame.Rect(cx - btn_w // 2, y, btn_w, btn_h)
        btn_color = (0, 150, 0) if self.active == "start" else (100, 180, 100)
        border_color = (0, 100, 0) if self.active == "start" else (80, 130, 80)
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, btn_rect, 3, border_radius=8)
        text = self.large_font.render("Start", True, WHITE)
        self.screen.blit(text, text.get_rect(center=btn_rect.center))

        # Warning if same model
        if self.m1_idx == self.m2_idx:
            y += 70
            text = self.small_font.render("(Same model selected for both — mirror match)",
                                          True, (180, 100, 0))
            self.screen.blit(text, text.get_rect(center=(cx, y)))

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

        # Label
        text = self.font.render(label, True, (60, 60, 60))
        self.screen.blit(text, (cx - box_w // 2 + 15, y + 2))

        # Value with arrows
        arrow_left = self.font.render("<", True, (100, 100, 100) if active else (200, 200, 200))
        arrow_right = self.font.render(">", True, (100, 100, 100) if active else (200, 200, 200))
        val_text = self.large_font.render(value, True, value_color)

        val_x = cx + 60
        self.screen.blit(arrow_left, (val_x - 80, y + 5))
        self.screen.blit(val_text, val_text.get_rect(center=(val_x, y + 22)))
        self.screen.blit(arrow_right, (val_x + 60, y + 5))


class CompareGUI(ChessGUI):
    """GUI for setting up opening positions, then watching models play live."""

    def __init__(self, game, eval1, eval2, sims, meta1, meta2):
        super().__init__(game)
        # Override window size
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Compare Models — Play an opening, then press SPACE")
        self.eval1 = eval1
        self.eval2 = eval2
        self.sims = sims
        self.meta1 = meta1
        self.meta2 = meta2

        # State machine: setup -> game_a -> game_a_done -> game_b -> review
        self.phase = "setup"
        self.frozen_state = None

        # Live game state
        self.current_game_label = ""
        self.m1_is_white = True
        self.move_count = 0
        self.pending_move = None
        self.thinking = False
        self.game_a_result = 0
        self.game_b_result = 0

        # Move history for replay
        self.game_a_history = []
        self.game_b_history = []
        self.game_a_moves = []
        self.game_b_moves = []
        self.current_history = []
        self.current_moves = []

        # Review state
        self.review_game = "a"
        self.review_idx = 0

        # Totals across rounds
        self.total_m1_wins = 0
        self.total_m2_wins = 0
        self.total_draws = 0
        # Color-specific breakdown (from M1's perspective as white/black)
        self.m1_wins_as_white = 0
        self.m1_wins_as_black = 0
        self.m2_wins_as_white = 0  # i.e. M1 lost as black
        self.m2_wins_as_black = 0  # i.e. M1 lost as white
        self.draws_as_white = 0    # M1 was white
        self.draws_as_black = 0    # M1 was black
        self.round_num = 0

    def handle_click(self, pos):
        if self.phase != "setup":
            return
        super().handle_click(pos)

    def start_comparison(self):
        self.frozen_state = deepcopy(self.game)
        self.round_num += 1
        print(f"\n=== Round {self.round_num} ===", flush=True)
        self._start_game("Game A", m1_is_white=True)

    def _start_game(self, label, m1_is_white):
        self.game = deepcopy(self.frozen_state)
        self.m1_is_white = m1_is_white
        self.current_game_label = f"{label} (M1={'White' if m1_is_white else 'Black'})"
        self.move_count = 0
        self.current_history = [deepcopy(self.game)]
        self.current_moves = []
        self.pending_move = None
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        color_str = "White" if m1_is_white else "Black"
        print(f"  {label} — M1 as {color_str}", flush=True)
        self.phase = "game_a" if "A" in label else "game_b"
        self._start_thinking()

    def _start_thinking(self):
        if self.game.game_over or self.move_count >= 300:
            self._finish_current_game()
            return
        self.thinking = True
        self.pending_move = None
        thread = threading.Thread(target=self._compute_move, daemon=True)
        thread.start()

    def _compute_move(self):
        try:
            is_model1_turn = (self.game.current_player == Color.WHITE) == self.m1_is_white
            evaluator = self.eval1 if is_model1_turn else self.eval2
            who = "M1" if is_model1_turn else "M2"

            mv, value = mcts_search(self.game, evaluator, num_simulations=self.sims,
                                    dirichlet_alpha=0, noise_weight=0,
                                    tactical_shortcuts=False)
            # Normalize eval to white's perspective
            if self.game.current_player == Color.BLACK:
                value = -value
            if not mv:
                self.pending_move = (None, who, 0, 0.0)
                return
            best = max(mv, key=lambda x: x[1])
            self.pending_move = (best[0], who, best[1], value)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.pending_move = (None, "??", 0, 0.0)

    def apply_pending_move(self):
        move, who, visits, value = self.pending_move
        self.pending_move = None
        self.thinking = False

        if move is None:
            self._finish_current_game()
            return

        self.move_count += 1
        self.current_moves.append((who, str(move), visits, value))
        print(f"    move {self.move_count}: {who} plays {move} (visits={visits}, eval={value:+.3f})", flush=True)
        self.game.make_move(move)
        self.last_move = move
        self.current_history.append(deepcopy(self.game))

        if self.game.game_over or self.move_count >= 300:
            self._finish_current_game()
        else:
            self._start_thinking()

    def _finish_current_game(self):
        if self.game.winner:
            model1_won = (self.game.winner == Color.WHITE) == self.m1_is_white
            result = 1 if model1_won else -1
        else:
            result = 0

        sym = "M1 wins" if result == 1 else ("M2 wins" if result == -1 else "Draw")
        print(f"  Result: {sym} ({self.move_count} moves)", flush=True)

        if self.phase == "game_a":
            self.game_a_result = result
            self.game_a_history = self.current_history[:]
            self.game_a_moves = self.current_moves[:]
            self.phase = "game_a_done"
        elif self.phase == "game_b":
            self.game_b_result = result
            self.game_b_history = self.current_history[:]
            self.game_b_moves = self.current_moves[:]
            # Game A: M1 is white. Game B: M1 is black.
            for r, m1_white in [(self.game_a_result, True), (self.game_b_result, False)]:
                if r == 1:
                    self.total_m1_wins += 1
                    if m1_white:
                        self.m1_wins_as_white += 1
                    else:
                        self.m1_wins_as_black += 1
                elif r == -1:
                    self.total_m2_wins += 1
                    if m1_white:
                        self.m2_wins_as_black += 1  # M2 won when M1 was white (M2 was black)
                    else:
                        self.m2_wins_as_white += 1  # M2 won when M1 was black (M2 was white)
                else:
                    self.total_draws += 1
                    if m1_white:
                        self.draws_as_white += 1
                    else:
                        self.draws_as_black += 1
            total = self.total_m1_wins + self.total_m2_wins + self.total_draws
            print(f"  Totals: M1={self.total_m1_wins}W  M2={self.total_m2_wins}W  D={self.total_draws}  ({total} games)", flush=True)
            self.phase = "review"
            self.review_game = "a"
            self.review_idx = len(self.game_a_history) - 1
            self._load_review_position()

        self.thinking = False

    def _load_review_position(self):
        history = self.game_a_history if self.review_game == "a" else self.game_b_history
        if 0 <= self.review_idx < len(history):
            self.game = deepcopy(history[self.review_idx])
            self.selected_square = None
            self.legal_moves = []
            self.last_move = None

    def reset_for_new_opening(self):
        self.game = ExtinctionChess()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.promotion_pending = False
        self.promotion_from = None
        self.promotion_to = None
        self.frozen_state = None
        self.pending_move = None
        self.thinking = False
        self.phase = "setup"

    def draw_sidebar(self):
        y = BOARD_OFFSET_Y

        iter1 = self.meta1.get('iteration', '?')
        iter2 = self.meta2.get('iteration', '?')
        total = self.total_m1_wins + self.total_m2_wins + self.total_draws

        # Title with game count
        if total > 0:
            title = self.large_font.render(f"Compare Models ({total} games)", True, BLACK)
        else:
            title = self.large_font.render("Compare Models", True, BLACK)
        self.screen.blit(title, (SIDEBAR_X, y))
        y += 40

        # Model info with win counts and color breakdown
        m1_label = f"M1: iter {iter1}"
        m2_label = f"M2: iter {iter2}"
        if total > 0:
            m1_parts = []
            if self.total_m1_wins > 0:
                m1_parts.append(f"{self.total_m1_wins}W ({self.m1_wins_as_white}W/{self.m1_wins_as_black}B)")
            if self.total_draws > 0:
                m1_parts.append(f"{self.total_draws}D ({self.draws_as_white}W/{self.draws_as_black}B)")
            m1_losses = self.total_m2_wins
            if m1_losses > 0:
                m1_l_as_w = self.m2_wins_as_black  # M1 lost as white = M2 won as black
                m1_l_as_b = self.m2_wins_as_white   # M1 lost as black = M2 won as white
                m1_parts.append(f"{m1_losses}L ({m1_l_as_w}W/{m1_l_as_b}B)")
            m1_label += "  —  " + ", ".join(m1_parts) if m1_parts else ""

            m2_parts = []
            if self.total_m2_wins > 0:
                m2_parts.append(f"{self.total_m2_wins}W ({self.m2_wins_as_white}W/{self.m2_wins_as_black}B)")
            if self.total_draws > 0:
                m2_parts.append(f"{self.total_draws}D ({self.draws_as_black}W/{self.draws_as_white}B)")
            m2_losses = self.total_m1_wins
            if m2_losses > 0:
                m2_l_as_w = self.m1_wins_as_black  # M2 lost as white = M1 won as black
                m2_l_as_b = self.m1_wins_as_white   # M2 lost as black = M1 won as white
                m2_parts.append(f"{m2_losses}L ({m2_l_as_w}W/{m2_l_as_b}B)")
            m2_label += "  —  " + ", ".join(m2_parts) if m2_parts else ""
        text = self.small_font.render(m1_label, True, (0, 0, 180))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 20
        text = self.small_font.render(m2_label, True, (180, 0, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 20
        text = self.small_font.render(f"Sims: {self.sims}", True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 24

        # Piece counts
        y = self._draw_piece_counts(y)
        y += 10

        if self.phase == "setup":
            text = self.font.render("Play opening moves on the board", True, (0, 130, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 24
            text = self.font.render("Press SPACE to start comparison", True, (0, 130, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 24
            text = self.small_font.render("R = reset board, ESC = quit", True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y))

        elif self.phase in ("game_a", "game_b"):
            text = self.font.render(self.current_game_label, True, (200, 100, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 24
            if self.thinking:
                is_m1 = (self.game.current_player == Color.WHITE) == self.m1_is_white
                who = "M1" if is_m1 else "M2"
                text = self.font.render(f"{who} thinking...", True, (200, 100, 0))
                self.screen.blit(text, (SIDEBAR_X, y))
            y += 24
            text = self.small_font.render(f"Move {self.move_count}", True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 30
            self._draw_move_list(y, self.current_moves, len(self.current_moves))

        elif self.phase == "game_a_done":
            sym = "M1 wins" if self.game_a_result == 1 else ("M2 wins" if self.game_a_result == -1 else "Draw")
            text = self.font.render(f"Game A: {sym}", True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 24
            text = self.font.render("Starting Game B...", True, (200, 100, 0))
            self.screen.blit(text, (SIDEBAR_X, y))

        elif self.phase == "review":
            self._draw_review_sidebar(y)

    def _draw_piece_counts(self, y):
        """Draw compact piece counts for both sides."""
        piece_chars = {
            PieceType.KING: 'K', PieceType.QUEEN: 'Q', PieceType.ROOK: 'R',
            PieceType.BISHOP: 'B', PieceType.KNIGHT: 'N', PieceType.PAWN: 'P',
        }
        endangered = {
            Color.WHITE: self.game.get_endangered_pieces(Color.WHITE),
            Color.BLACK: self.game.get_endangered_pieces(Color.BLACK),
        }

        for color in [Color.WHITE, Color.BLACK]:
            counts = self.game.board.get_piece_count(color)
            label = "White:" if color == Color.WHITE else "Black:"
            text = self.small_font.render(label, True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))

            x = SIDEBAR_X + 50
            for pt in [PieceType.KING, PieceType.QUEEN, PieceType.ROOK,
                        PieceType.BISHOP, PieceType.KNIGHT, PieceType.PAWN]:
                count = counts.get(pt, 0)
                char = piece_chars[pt]
                clr = (255, 80, 0) if pt in endangered[color] else (80, 80, 80)
                if count == 0:
                    clr = (200, 0, 0)
                entry = f"{char}:{count}"
                text = self.small_font.render(entry, True, clr)
                self.screen.blit(text, (x, y))
                x += 42
            y += 18

        return y

    def _draw_move_list(self, y, moves, highlight_idx):
        max_visible = 18
        start = max(0, len(moves) - max_visible)
        for i in range(start, len(moves)):
            who, move_str, visits, value = moves[i]
            move_num = i + 1
            color = (0, 0, 180) if who == "M1" else (180, 0, 0)

            if i == highlight_idx - 1:
                pygame.draw.rect(self.screen, (230, 230, 255),
                                (SIDEBAR_X - 2, y - 1, PANEL_WIDTH, 18))

            # Move info
            text = self.small_font.render(f"{move_num}. {who}: {move_str}", True, color)
            self.screen.blit(text, (SIDEBAR_X, y))

            # Visits and eval on the right
            eval_color = (0, 130, 0) if value > 0 else ((180, 0, 0) if value < 0 else (100, 100, 100))
            info = self.small_font.render(f"v:{visits} e:{value:+.2f}", True, eval_color)
            self.screen.blit(info, (SIDEBAR_X + 200, y))
            y += 18

    def _draw_review_sidebar(self, y):
        def sym(r):
            return "M1 wins" if r == 1 else ("M2 wins" if r == -1 else "Draw")

        text = self.font.render(f"Game A (M1=W): {sym(self.game_a_result)}", True, BLACK)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22
        text = self.font.render(f"Game B (M1=B): {sym(self.game_b_result)}", True, BLACK)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 30

        reviewing_label = "Game A" if self.review_game == "a" else "Game B"
        history = self.game_a_history if self.review_game == "a" else self.game_b_history
        moves = self.game_a_moves if self.review_game == "a" else self.game_b_moves
        total_positions = len(history)

        active_color = (0, 0, 180) if self.review_game == "a" else (180, 0, 0)
        text = self.font.render(f"Reviewing: {reviewing_label}", True, active_color)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22
        text = self.small_font.render(
            f"Position {self.review_idx}/{total_positions - 1}", True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 24

        text = self.small_font.render("LEFT/RIGHT = step through moves", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render("HOME/END = jump to start/end", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render("TAB = switch Game A / Game B", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render("SPACE = next opening, ESC = quit", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 28

        self._draw_move_list(y, moves, self.review_idx)

    def run(self):
        running = True
        clock = pygame.time.Clock()
        game_a_done_timer = 0

        while running:
            if self.thinking and self.pending_move is not None:
                self.apply_pending_move()

            if self.phase == "game_a_done":
                game_a_done_timer += 1
                if game_a_done_timer > 90:
                    game_a_done_timer = 0
                    self._start_game("Game B", m1_is_white=False)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(pygame.mouse.get_pos())
                    elif event.button == 3 and self.phase == "setup":
                        self.selected_square = None
                        self.legal_moves = []

                elif event.type == pygame.KEYDOWN:
                    if self.phase == "setup" and self.promotion_pending:
                        self.handle_promotion_key(event.key)

                    elif event.key == pygame.K_SPACE:
                        if self.phase == "setup":
                            self.start_comparison()
                        elif self.phase == "review":
                            self.reset_for_new_opening()

                    elif event.key == pygame.K_r and self.phase == "setup":
                        self.reset_for_new_opening()

                    elif event.key == pygame.K_ESCAPE:
                        running = False

                    elif self.phase == "review":
                        history = self.game_a_history if self.review_game == "a" else self.game_b_history

                        if event.key == pygame.K_LEFT:
                            if self.review_idx > 0:
                                self.review_idx -= 1
                                self._load_review_position()
                        elif event.key == pygame.K_RIGHT:
                            if self.review_idx < len(history) - 1:
                                self.review_idx += 1
                                self._load_review_position()
                        elif event.key == pygame.K_HOME:
                            self.review_idx = 0
                            self._load_review_position()
                        elif event.key == pygame.K_END:
                            self.review_idx = len(history) - 1
                            self._load_review_position()
                        elif event.key == pygame.K_TAB:
                            if self.review_game == "a" and self.game_b_history:
                                self.review_game = "b"
                                self.review_idx = len(self.game_b_history) - 1
                            else:
                                self.review_game = "a"
                                self.review_idx = len(self.game_a_history) - 1
                            self._load_review_position()

            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_sidebar()
            self.draw_promotion_dialog()
            pygame.display.flip()
            clock.tick(60)

        # Print final summary
        total = self.total_m1_wins + self.total_m2_wins + self.total_draws
        if total > 0:
            m1_score = (self.total_m1_wins + 0.5 * self.total_draws) / total
            m2_score = (self.total_m2_wins + 0.5 * self.total_draws) / total
            print(f"\nFinal Results ({total} games across {self.round_num} openings):")
            print(f"  M1 (iter {self.meta1.get('iteration', '?')}): {self.total_m1_wins}W ({self.m1_wins_as_white}W/{self.m1_wins_as_black}B) {self.total_draws}D {self.total_m2_wins}L — {m1_score:.1%}")
            print(f"  M2 (iter {self.meta2.get('iteration', '?')}): {self.total_m2_wins}W ({self.m2_wins_as_white}W/{self.m2_wins_as_black}B) {self.total_draws}D {self.total_m1_wins}L — {m2_score:.1%}")

        pygame.quit()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description='Compare two AlphaZero models')
    parser.add_argument('--model1', type=str, default=None, help='Path to first model (optional, use menu if omitted)')
    parser.add_argument('--model2', type=str, default=None, help='Path to second model (optional, use menu if omitted)')
    parser.add_argument('--sims', type=int, default=None, help='MCTS simulations per move (optional, use menu if omitted)')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR, help='Directory to scan for models')
    args = parser.parse_args()

    pygame.init()

    if args.model1 and args.model2 and args.sims:
        # Direct mode — skip menu
        m1_path, m2_path, sims = args.model1, args.model2, args.sims
    else:
        # Show menu
        print("Scanning for models...", flush=True)
        models = discover_models(args.models_dir)
        if not models:
            print(f"No .pt files found in {args.models_dir}")
            sys.exit(1)
        print(f"Found {len(models)} models", flush=True)

        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Compare Models — Select Models")
        menu = MenuScreen(screen, models)
        result = menu.run()
        if result is None:
            pygame.quit()
            sys.exit(0)
        m1_path, m2_path, sims = result

    print(f"Loading model 1: {m1_path}")
    model1, meta1 = AlphaZeroNet.load_checkpoint(m1_path)
    eval1 = AlphaZeroEvaluator(model1, device="cpu")
    print(f"  Iteration: {meta1.get('iteration', '?')}", flush=True)

    print(f"Loading model 2: {m2_path}")
    model2, meta2 = AlphaZeroNet.load_checkpoint(m2_path)
    eval2 = AlphaZeroEvaluator(model2, device="cpu")
    print(f"  Iteration: {meta2.get('iteration', '?')}", flush=True)

    print(f"\nSims: {sims}")
    print(f"Play opening moves, then press SPACE to watch the models play.")
    print(f"After both games, use LEFT/RIGHT to review, TAB to switch games.")
    print(f"Press SPACE for a new opening, ESC to quit.\n")

    game = ExtinctionChess()
    gui = CompareGUI(game, eval1, eval2, sims, meta1, meta2)
    gui.run()


if __name__ == "__main__":
    main()
