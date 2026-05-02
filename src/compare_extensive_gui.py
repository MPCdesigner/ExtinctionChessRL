"""
Visual extensive head-to-head comparison between two AlphaZero checkpoints.

Plays all 20 legal opening moves x 2 games (each model as white) at a chosen
sim setting, displaying every game on a pygame board.

Usage:
    python compare_extensive_gui.py
    python compare_extensive_gui.py --model1 ../models/az_iter100.pt --model2 ../models/az_iter280.pt --sims 100
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

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750
SIDEBAR_X = BOARD_OFFSET_X + BOARD_SIZE + 20
PANEL_WIDTH = WINDOW_WIDTH - SIDEBAR_X - 10

SIM_OPTIONS = [5, 10, 20, 50, 100, 200, 400]
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def _copy_game(game):
    """Copy a game state (C++ objects can't be deepcopied)."""
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


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
    """Model and sim selection menu for extensive comparison."""

    def __init__(self, screen, models):
        self.screen = screen
        self.models = models
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 56)

        self.m1_idx = len(models) - 1 if models else 0
        self.m2_idx = 0
        self.sim_idx = 4  # default to 100
        self.active = "m1"

    def run(self):
        """Run menu loop. Returns config dict or None if cancelled."""
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
                    elif event.key in (pygame.K_TAB, pygame.K_DOWN):
                        self._next_field()
                    elif event.key == pygame.K_UP:
                        self._prev_field()
                    elif event.key == pygame.K_LEFT:
                        self._adjust(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._adjust(1)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if self.active == "start":
                            return {
                                "model1": self.models[self.m1_idx]["path"],
                                "model2": self.models[self.m2_idx]["path"],
                                "sims": SIM_OPTIONS[self.sim_idx],
                            }
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
        y = 60

        text = self.title_font.render("Extensive Comparison", True, BLACK)
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 60

        text = self.small_font.render(
            "20 openings x 2 games each = 40 games total",
            True, (100, 100, 100))
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 25
        text = self.small_font.render(
            "Use LEFT/RIGHT to change, TAB/DOWN to next field, SPACE/ENTER to confirm",
            True, (100, 100, 100))
        self.screen.blit(text, text.get_rect(center=(cx, y)))
        y += 50

        self._draw_selector(y, "Model 1 (Blue):", self.models[self.m1_idx]["label"],
                            self.active == "m1", (0, 0, 180))
        y += 65

        self._draw_selector(y, "Model 2 (Red):", self.models[self.m2_idx]["label"],
                            self.active == "m2", (180, 0, 0))
        y += 65

        self._draw_selector(y, "Simulations:", str(SIM_OPTIONS[self.sim_idx]),
                            self.active == "sims", BLACK)
        y += 80

        btn_w, btn_h = 250, 50
        btn_rect = pygame.Rect(cx - btn_w // 2, y, btn_w, btn_h)
        btn_color = (0, 150, 0) if self.active == "start" else (100, 180, 100)
        border_color = (0, 100, 0) if self.active == "start" else (80, 130, 80)
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, btn_rect, 3, border_radius=8)
        text = self.large_font.render("Start", True, WHITE)
        self.screen.blit(text, text.get_rect(center=btn_rect.center))

        if self.m1_idx == self.m2_idx:
            y += 70
            text = self.small_font.render("(Same model selected — mirror match)",
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

        text = self.font.render(label, True, (60, 60, 60))
        self.screen.blit(text, (cx - box_w // 2 + 15, y + 2))

        arrow_left = self.font.render("<", True, (100, 100, 100) if active else (200, 200, 200))
        arrow_right = self.font.render(">", True, (100, 100, 100) if active else (200, 200, 200))
        val_text = self.large_font.render(value, True, value_color)

        val_x = cx + 60
        self.screen.blit(arrow_left, (val_x - 80, y + 5))
        self.screen.blit(val_text, val_text.get_rect(center=(val_x, y + 22)))
        self.screen.blit(arrow_right, (val_x + 60, y + 5))


class ExtensiveCompareGUI(ChessGUI):
    """GUI that auto-plays through all 20 openings without stopping.

    Games run continuously in the background. SPACE toggles between live
    view (current game) and review mode (browse all completed games).
    """

    MAX_MOVES = 300

    def __init__(self, eval1, eval2, sims, meta1, meta2):
        game = ExtinctionChess()
        super().__init__(game)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Extensive Comparison")

        self.eval1 = eval1
        self.eval2 = eval2
        self.sims = sims
        self.meta1 = meta1
        self.meta2 = meta2
        self.label1 = f"iter {meta1.get('iteration', '?')}"
        self.label2 = f"iter {meta2.get('iteration', '?')}"

        # All 20 opening moves
        start = ExtinctionChess()
        self.opening_moves = start.get_legal_moves()
        self.total_openings = len(self.opening_moves)
        self.total_games = self.total_openings * 2

        # ── Live game state (always running) ──
        self.live_game = ExtinctionChess()
        self.live_history = []
        self.live_moves = []
        self.live_m1_is_white = True
        self.live_move_count = 0
        self.live_last_move = None
        self.thinking = False
        self.pending_move = None

        # Progress
        self.opening_idx = 0
        self.game_in_pair = 0  # 0 = game A (M1 white), 1 = game B (M1 black)
        self.games_played = 0
        self.all_done = False

        # Aggregate results
        self.m1_score = 0.0
        self.m2_score = 0.0
        self.m1_wins = 0
        self.m2_wins = 0
        self.draws = 0
        self.opening_results = []  # [(opening_str, m1_score, m2_score), ...]
        self.current_opening_m1 = 0.0
        self.current_opening_m2 = 0.0

        # ── Completed games archive ──
        # Each entry: {"opening": str, "opening_idx": int, "label": "A"/"B",
        #   "m1_is_white": bool, "history": [...], "moves": [...],
        #   "result": int, "result_str": str}
        self.completed_games = []

        # ── View mode ──
        self.viewing = "live"  # "live" or "review"
        self.review_game_idx = 0
        self.review_move_idx = 0

        # Start first game
        self._start_opening(0)

    # ── Game engine (runs regardless of view) ────────────────────────

    def _start_opening(self, idx):
        """Start a new opening pair."""
        self.opening_idx = idx
        self.game_in_pair = 0
        self.current_opening_m1 = 0.0
        self.current_opening_m2 = 0.0
        self._start_game(m1_is_white=True)

    def _start_game(self, m1_is_white):
        """Set up and start a single game from the current opening."""
        opening_move = self.opening_moves[self.opening_idx]
        start = ExtinctionChess()
        start.make_move(opening_move)
        self.live_game = _copy_game(start)

        self.live_m1_is_white = m1_is_white
        self.live_move_count = 0
        self.live_moves = []
        self.live_history = [_copy_game(self.live_game)]
        self.live_last_move = None
        self.pending_move = None

        label = "A" if self.game_in_pair == 0 else "B"
        color = "White" if m1_is_white else "Black"
        print(f"  Opening {self.opening_idx + 1}/{self.total_openings}: "
              f"1.{opening_move} — Game {label} (M1={color})", flush=True)
        self._start_thinking()

    def _start_thinking(self):
        if self.live_game.game_over or self.live_move_count >= self.MAX_MOVES:
            self._finish_game()
            return
        self.thinking = True
        self.pending_move = None
        thread = threading.Thread(target=self._compute_move, daemon=True)
        thread.start()

    def _compute_move(self):
        try:
            is_m1 = ((self.live_game.current_player == Color.WHITE)
                      == self.live_m1_is_white)
            who = "M1" if is_m1 else "M2"
            evaluator = self.eval1 if is_m1 else self.eval2

            mv, value = mcts_search(
                self.live_game, evaluator,
                num_simulations=self.sims,
                dirichlet_alpha=0, noise_weight=0,
                tactical_shortcuts=False,
            )
            if self.live_game.current_player == Color.BLACK:
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

    def _apply_move(self):
        move, who, visits, value = self.pending_move
        self.pending_move = None
        self.thinking = False

        if move is None:
            self._finish_game()
            return

        self.live_move_count += 1
        self.live_moves.append((who, str(move), visits, value))
        print(f"    {self.live_move_count:>3}. {who}: {move} "
              f"(v={visits}, e={value:+.3f})", flush=True)
        self.live_game.make_move(move)
        self.live_last_move = move
        self.live_history.append(_copy_game(self.live_game))

        if self.live_game.game_over or self.live_move_count >= self.MAX_MOVES:
            self._finish_game()
        else:
            self._start_thinking()

    def _finish_game(self):
        """Handle end of a single game — archive and immediately start next."""
        if self.live_game.winner:
            m1_won = ((self.live_game.winner == Color.WHITE)
                       == self.live_m1_is_white)
            result = 1 if m1_won else -1
        else:
            result = 0

        if result == 1:
            self.current_opening_m1 += 1
        elif result == -1:
            self.current_opening_m2 += 1
        else:
            self.current_opening_m1 += 0.5
            self.current_opening_m2 += 0.5

        self.games_played += 1
        sym = ("M1 wins" if result == 1
               else "M2 wins" if result == -1
               else "Draw")
        print(f"    -> {sym} ({self.live_move_count} moves)", flush=True)

        # Archive completed game
        opening_str = str(self.opening_moves[self.opening_idx])
        game_label = "A" if self.game_in_pair == 0 else "B"
        self.completed_games.append({
            "opening": opening_str,
            "opening_idx": self.opening_idx,
            "label": game_label,
            "m1_is_white": self.live_m1_is_white,
            "history": self.live_history[:],
            "moves": self.live_moves[:],
            "result": result,
            "result_str": sym,
        })

        # If game B just finished, accumulate opening pair result
        if self.game_in_pair == 1:
            self.m1_score += self.current_opening_m1
            self.m2_score += self.current_opening_m2
            if self.current_opening_m1 > self.current_opening_m2:
                self.m1_wins += 1
            elif self.current_opening_m2 > self.current_opening_m1:
                self.m2_wins += 1
            else:
                self.draws += 1
            self.opening_results.append(
                (opening_str, self.current_opening_m1,
                 self.current_opening_m2))
            print(f"  Opening result: M1 {self.current_opening_m1:.1f}"
                  f"-{self.current_opening_m2:.1f} M2  "
                  f"(running: {self.m1_score:.1f}-{self.m2_score:.1f})",
                  flush=True)

        self.thinking = False

        # Immediately start next game (no pause)
        if self.game_in_pair == 0:
            self.game_in_pair = 1
            self._start_game(m1_is_white=False)
        else:
            next_idx = self.opening_idx + 1
            if next_idx >= self.total_openings:
                self.all_done = True
                self._print_summary()
            else:
                self._start_opening(next_idx)

    def _print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULTS ({self.games_played} games, "
              f"{self.total_openings} openings)")
        print(f"{'=' * 60}")
        if self.games_played > 0:
            m1_pct = self.m1_score / self.games_played * 100
            m2_pct = self.m2_score / self.games_played * 100
            print(f"  {self.label1}: {self.m1_score:.1f}/{self.games_played} "
                  f"({m1_pct:.1f}%)")
            print(f"  {self.label2}: {self.m2_score:.1f}/{self.games_played} "
                  f"({m2_pct:.1f}%)")
            print(f"  Opening W-D-L (M1): "
                  f"{self.m1_wins}-{self.draws}-{self.m2_wins}")

    # ── Display: set self.game for drawing ───────────────────────────

    def _sync_display(self):
        """Point self.game at the right state for the current view."""
        if self.viewing == "live" or not self.completed_games:
            self.game = self.live_game
            self.last_move = self.live_last_move
        else:
            cg = self.completed_games[self.review_game_idx]
            history = cg["history"]
            idx = max(0, min(self.review_move_idx, len(history) - 1))
            self.game = _copy_game(history[idx])
            # Show the move that led to this position
            if idx > 0 and idx - 1 < len(cg["moves"]):
                move_str = cg["moves"][idx - 1][1]
                # We don't have the Move object, so clear last_move highlight
                self.last_move = None
            else:
                self.last_move = None
        self.selected_square = None
        self.legal_moves = []

    # ── Drawing ──────────────────────────────────────────────────────

    def draw_sidebar(self):
        y = BOARD_OFFSET_Y

        # Title
        title = self.large_font.render("Extensive Comparison", True, BLACK)
        self.screen.blit(title, (SIDEBAR_X, y))
        y += 35

        # Models and sims
        iter1 = self.meta1.get('iteration', '?')
        iter2 = self.meta2.get('iteration', '?')
        text = self.small_font.render(f"M1: iter {iter1}", True, (0, 0, 180))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render(f"M2: iter {iter2}", True, (180, 0, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render(
            f"Sims: {self.sims}", True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22

        # Progress
        if not self.all_done:
            opening_str = str(self.opening_moves[self.opening_idx])
            game_label = "A (M1=W)" if self.game_in_pair == 0 else "B (M1=B)"
            text = self.font.render(
                f"Game {self.games_played + 1}/{self.total_games}  |  "
                f"Opening {self.opening_idx + 1}/{self.total_openings}",
                True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            text = self.small_font.render(
                f"1.{opening_str} — Game {game_label}",
                True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
        else:
            text = self.font.render(
                f"All {self.total_games} games complete", True, (0, 130, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 44

        # Score
        pygame.draw.line(self.screen, (200, 200, 200),
                         (SIDEBAR_X, y), (SIDEBAR_X + PANEL_WIDTH, y))
        y += 8
        text = self.font.render(
            f"Score: M1 {self.m1_score:.1f} - {self.m2_score:.1f} M2",
            True, BLACK)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22
        text = self.small_font.render(
            f"Openings: M1 {self.m1_wins}W  {self.draws}D  {self.m2_wins}L",
            True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22

        # Piece counts
        y = self._draw_piece_counts(y)
        y += 8
        pygame.draw.line(self.screen, (200, 200, 200),
                         (SIDEBAR_X, y), (SIDEBAR_X + PANEL_WIDTH, y))
        y += 8

        if self.viewing == "live":
            self._draw_live_sidebar(y)
        else:
            self._draw_review_sidebar(y)

    def _draw_piece_counts(self, y):
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
        max_visible = 14
        start = max(0, len(moves) - max_visible)
        for i in range(start, len(moves)):
            who, move_str, visits, value = moves[i]
            move_num = i + 1
            color = (0, 0, 180) if who == "M1" else (180, 0, 0)
            if i == highlight_idx - 1:
                pygame.draw.rect(self.screen, (230, 230, 255),
                                 (SIDEBAR_X - 2, y - 1, PANEL_WIDTH, 18))
            text = self.small_font.render(
                f"{move_num}. {who}: {move_str}", True, color)
            self.screen.blit(text, (SIDEBAR_X, y))
            eval_color = ((0, 130, 0) if value > 0
                          else (180, 0, 0) if value < 0
                          else (100, 100, 100))
            info = self.small_font.render(
                f"v:{visits} e:{value:+.2f}", True, eval_color)
            self.screen.blit(info, (SIDEBAR_X + 200, y))
            y += 18

    def _draw_live_sidebar(self, y):
        """Sidebar content when viewing the live game."""
        # LIVE indicator
        text = self.font.render("LIVE", True, (0, 180, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 24

        if self.all_done:
            self._draw_finished_section(y)
            return

        if self.thinking:
            is_m1 = ((self.live_game.current_player == Color.WHITE)
                      == self.live_m1_is_white)
            who = "M1" if is_m1 else "M2"
            text = self.font.render(
                f"{who} thinking... (move {self.live_move_count + 1})",
                True, (200, 100, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
        y += 24

        # Hint
        if self.completed_games:
            text = self.small_font.render(
                f"SPACE = review ({len(self.completed_games)} completed)",
                True, (0, 130, 0))
        else:
            text = self.small_font.render(
                "SPACE = review (waiting for first game...)",
                True, (150, 150, 150))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22

        # Current game move list
        self._draw_move_list(
            y, self.live_moves, len(self.live_moves))

    def _draw_review_sidebar(self, y):
        """Sidebar content when reviewing completed games."""
        if not self.completed_games:
            return

        cg = self.completed_games[self.review_game_idx]

        # REVIEW header
        if not self.all_done:
            text = self.font.render("REVIEW (games still running)",
                                    True, (180, 100, 0))
        else:
            text = self.font.render("REVIEW", True, (180, 100, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 24

        # Which game
        m1_color = "W" if cg["m1_is_white"] else "B"
        sym = ("M1 wins" if cg["result"] == 1
               else "M2 wins" if cg["result"] == -1
               else "Draw")
        text = self.font.render(
            f"Game {self.review_game_idx + 1}/{len(self.completed_games)}: "
            f"1.{cg['opening']} {cg['label']} (M1={m1_color})",
            True, BLACK)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22
        text = self.small_font.render(
            f"Result: {sym}  |  {len(cg['moves'])} moves",
            True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 20

        # Position
        history = cg["history"]
        text = self.small_font.render(
            f"Position {self.review_move_idx}/{len(history) - 1}",
            True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22

        # Controls
        text = self.small_font.render(
            "LEFT/RIGHT = step, HOME/END = jump", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render(
            "UP/DOWN = prev/next game", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        text = self.small_font.render(
            "SPACE = back to live, ESC = quit", True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22

        # Move list
        self._draw_move_list(y, cg["moves"], self.review_move_idx)

    def _draw_finished_section(self, y):
        """Draw final results (used inside live sidebar when all done)."""
        if self.games_played == 0:
            return

        m1_pct = self.m1_score / self.games_played * 100
        m2_pct = self.m2_score / self.games_played * 100

        text = self.font.render(
            f"{self.label1}: {self.m1_score:.1f}/{self.games_played} "
            f"({m1_pct:.1f}%)", True, (0, 0, 180))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 24
        text = self.font.render(
            f"{self.label2}: {self.m2_score:.1f}/{self.games_played} "
            f"({m2_pct:.1f}%)", True, (180, 0, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 24
        text = self.font.render(
            f"Openings: M1 {self.m1_wins}W  {self.draws}D  {self.m2_wins}L",
            True, BLACK)
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 28

        # Per-opening results
        text = self.small_font.render(
            "Per-opening scores:", True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 18
        max_visible = 20
        start = max(0, len(self.opening_results) - max_visible)
        for i in range(start, len(self.opening_results)):
            opening, m1s, m2s = self.opening_results[i]
            if m1s > m2s:
                clr = (0, 0, 180)
            elif m2s > m1s:
                clr = (180, 0, 0)
            else:
                clr = (100, 100, 100)
            text = self.small_font.render(
                f"  1.{opening}: M1 {m1s:.1f}-{m2s:.1f} M2", True, clr)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 16

        y += 10
        text = self.small_font.render(
            f"SPACE = review all {len(self.completed_games)} games",
            True, (0, 130, 0))
        self.screen.blit(text, (SIDEBAR_X, y))

    # ── Main loop ────────────────────────────────────────────────────

    def run(self):
        running = True
        clock = pygame.time.Clock()

        while running:
            # Always process AI moves (regardless of view mode)
            if (not self.all_done and self.thinking
                    and self.pending_move is not None):
                self._apply_move()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if self.viewing == "live":
                            if self.completed_games:
                                self.viewing = "review"
                                self.review_game_idx = (
                                    len(self.completed_games) - 1)
                                cg = self.completed_games[
                                    self.review_game_idx]
                                self.review_move_idx = (
                                    len(cg["history"]) - 1)
                        else:
                            self.viewing = "live"

                    elif self.viewing == "review" and self.completed_games:
                        cg = self.completed_games[self.review_game_idx]
                        history = cg["history"]

                        if event.key == pygame.K_LEFT:
                            if self.review_move_idx > 0:
                                self.review_move_idx -= 1
                        elif event.key == pygame.K_RIGHT:
                            if self.review_move_idx < len(history) - 1:
                                self.review_move_idx += 1
                        elif event.key == pygame.K_HOME:
                            self.review_move_idx = 0
                        elif event.key == pygame.K_END:
                            self.review_move_idx = len(history) - 1
                        elif event.key == pygame.K_UP:
                            if self.review_game_idx > 0:
                                self.review_game_idx -= 1
                                new_cg = self.completed_games[
                                    self.review_game_idx]
                                self.review_move_idx = (
                                    len(new_cg["history"]) - 1)
                        elif event.key == pygame.K_DOWN:
                            if (self.review_game_idx
                                    < len(self.completed_games) - 1):
                                self.review_game_idx += 1
                                new_cg = self.completed_games[
                                    self.review_game_idx]
                                self.review_move_idx = (
                                    len(new_cg["history"]) - 1)

            # Set display state and draw
            self._sync_display()
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_sidebar()
            pygame.display.flip()
            clock.tick(60)

        # Print final summary on exit
        if self.games_played > 0:
            self._print_summary()

        pygame.quit()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(
        description="Visual extensive head-to-head comparison")
    parser.add_argument('--model1', type=str, default=None,
                        help='Path to model 1 (skip menu)')
    parser.add_argument('--model2', type=str, default=None,
                        help='Path to model 2 (skip menu)')
    parser.add_argument('--sims', type=int, default=None,
                        help='MCTS simulations per move (skip menu)')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR,
                        help='Directory to scan for models')
    args = parser.parse_args()

    pygame.init()

    if args.model1 and args.model2 and args.sims:
        config = {
            "model1": args.model1,
            "model2": args.model2,
            "sims": args.sims,
        }
    else:
        print("Scanning for models...", flush=True)
        models = discover_models(args.models_dir)
        if not models:
            print(f"No .pt files found in {args.models_dir}")
            sys.exit(1)
        print(f"Found {len(models)} models", flush=True)

        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Extensive Comparison — Select Models")
        menu = MenuScreen(screen, models)
        config = menu.run()
        if config is None:
            pygame.quit()
            sys.exit(0)

    print(f"Loading model 1: {config['model1']}")
    model1, meta1 = AlphaZeroNet.load_checkpoint(config['model1'])
    eval1 = AlphaZeroEvaluator(model1, device="cpu")
    print(f"  Iteration: {meta1.get('iteration', '?')}")

    print(f"Loading model 2: {config['model2']}")
    model2, meta2 = AlphaZeroNet.load_checkpoint(config['model2'])
    eval2 = AlphaZeroEvaluator(model2, device="cpu")
    print(f"  Iteration: {meta2.get('iteration', '?')}")

    print(f"\nSims: {config['sims']}")
    print(f"20 openings x 2 games = 40 games total")
    print(f"Games run continuously — SPACE toggles review mode")
    print(f"Review: LEFT/RIGHT step, UP/DOWN switch games, SPACE back to live")
    print(f"ESC = quit\n")

    gui = ExtensiveCompareGUI(eval1, eval2, config['sims'], meta1, meta2)
    gui.run()


if __name__ == "__main__":
    main()
