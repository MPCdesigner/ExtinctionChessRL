"""
Win-taking visual test: generates random positions where an instant win
exists, then tests whether each selected model (at various sim counts)
finds the winning capture.

Pygame interface with a config screen to select models, sim counts,
and number of positions to test. Saves results to a JSON file.

Usage:
    python win_taking_test2.py
"""

import pygame
import sys
import os
import json
import random
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from extinction_chess import ExtinctionChess, Position, Color, PieceType, Piece
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search

pygame.init()

# ── Layout constants ──────────────────────────────────────────────────────
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750
BOARD_SIZE = 560
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_X = 30
BOARD_Y = 30
PANEL_X = BOARD_X + BOARD_SIZE + 20
PANEL_W = WINDOW_WIDTH - PANEL_X - 10

# ── Colors ────────────────────────────────────────────────────────────────
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (255, 255, 0, 128)
WIN_GREEN = (0, 180, 0)
MISS_RED = (220, 0, 0)
ENDANGERED_COLOR = (255, 100, 0)
BTN_COLOR = (70, 130, 200)
BTN_HOVER = (90, 150, 220)
BTN_DISABLED = (160, 160, 160)
CHECKBOX_ON = (70, 130, 200)
CHECKBOX_OFF = (200, 200, 200)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ═════════════════════════════════════════════════════════════════════════════
# Game helpers
# ═════════════════════════════════════════════════════════════════════════════

def copy_game(game):
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


def find_winning_moves(game, target_piece_types=None):
    """Return list of (move, captured_piece_type) that immediately win.
    If target_piece_types is set, only return wins that extinct one of those types."""
    current = game.current_player
    opponent = Color.BLACK if current == Color.WHITE else Color.WHITE
    winners = []
    for m in game.get_legal_moves():
        gc = copy_game(game)
        # Check what piece is on the target square before the move
        captured_piece = game.board.get_piece(m.to_pos)
        if gc.make_move(m) and gc.game_over and gc.winner == current:
            # Determine which piece type was extinctd
            captured_type = None
            if captured_piece and captured_piece.color == opponent:
                captured_type = captured_piece.piece_type
            if target_piece_types is None or captured_type in target_piece_types:
                winners.append((m, captured_type))
    return winners


# ═════════════════════════════════════════════════════════════════════════════
# Config screen
# ═════════════════════════════════════════════════════════════════════════════

class ConfigScreen:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 42)

        # Discover models
        self.model_files = sorted([
            f for f in os.listdir(MODELS_DIR)
            if f.endswith(".pt") and f.startswith("az_iter")
        ])
        self.model_selected = [False] * len(self.model_files)

        # Sim options
        self.all_sims = [20, 50, 100, 200, 400]
        self.sim_selected = [False, False, True, True, False]  # default: 100, 200

        # Position count
        self.num_positions = 50
        self.pos_input_active = False
        self.pos_input_text = "50"

        # Save results toggle
        self.save_results = False

        # Target piece types filter
        self.piece_types = [
            PieceType.KING, PieceType.QUEEN, PieceType.ROOK,
            PieceType.BISHOP, PieceType.KNIGHT, PieceType.PAWN,
        ]
        self.piece_labels = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
        self.piece_selected = [True] * len(self.piece_types)  # all selected by default

        # Scroll for model list
        self.scroll_offset = 0
        self.max_visible_models = 12

    def run(self):
        """Returns (model_paths, sim_counts, num_positions, save_results, target_pieces) or None."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if self.pos_input_active:
                        if event.key == pygame.K_RETURN:
                            self.pos_input_active = False
                            try:
                                self.num_positions = max(1, int(self.pos_input_text))
                            except ValueError:
                                self.num_positions = 50
                                self.pos_input_text = "50"
                        elif event.key == pygame.K_BACKSPACE:
                            self.pos_input_text = self.pos_input_text[:-1]
                        elif event.unicode.isdigit():
                            self.pos_input_text += event.unicode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        result = self._handle_click(event.pos)
                        if result is not None:
                            return result
                    elif event.button == 4:  # scroll up
                        self.scroll_offset = max(0, self.scroll_offset - 1)
                    elif event.button == 5:  # scroll down
                        max_scroll = max(0, len(self.model_files) - self.max_visible_models)
                        self.scroll_offset = min(max_scroll, self.scroll_offset + 1)

            self._draw()
            self.clock.tick(30)

    def _handle_click(self, pos):
        mx, my = pos

        # Model checkboxes
        model_start_y = 100
        for i in range(self.max_visible_models):
            idx = i + self.scroll_offset
            if idx >= len(self.model_files):
                break
            y = model_start_y + i * 30
            if 50 <= mx <= 500 and y <= my <= y + 25:
                self.model_selected[idx] = not self.model_selected[idx]
                return None

        # Sim checkboxes
        sim_start_y = 100
        for i, s in enumerate(self.all_sims):
            y = sim_start_y + i * 35
            if 550 <= mx <= 800 and y <= my <= y + 30:
                self.sim_selected[i] = not self.sim_selected[i]
                return None

        # Position count input
        input_rect = pygame.Rect(700, sim_start_y + len(self.all_sims) * 35 + 50, 100, 30)
        if input_rect.collidepoint(mx, my):
            self.pos_input_active = True
            return None
        else:
            if self.pos_input_active:
                self.pos_input_active = False
                try:
                    self.num_positions = max(1, int(self.pos_input_text))
                except ValueError:
                    self.num_positions = 50
                    self.pos_input_text = "50"

        # Start button
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 70, 200, 45)
        if btn_rect.collidepoint(mx, my):
            models = [self.model_files[i] for i, sel in enumerate(self.model_selected) if sel]
            sims = [self.all_sims[i] for i, sel in enumerate(self.sim_selected) if sel]
            if models and sims:
                paths = [os.path.join(MODELS_DIR, m) for m in models]
                try:
                    self.num_positions = max(1, int(self.pos_input_text))
                except ValueError:
                    self.num_positions = 50
                target = None if all(self.piece_selected) else set(
                    pt for pt, sel in zip(self.piece_types, self.piece_selected) if sel
                )
                return (paths, sims, self.num_positions, self.save_results, target)

        # Piece type checkboxes
        piece_start_y = sim_start_y + len(self.all_sims) * 35 + 40
        for i in range(len(self.piece_types)):
            y = piece_start_y + i * 30
            if 550 <= mx <= 800 and y <= my <= y + 25:
                self.piece_selected[i] = not self.piece_selected[i]
                return None

        # Save results checkbox
        save_y = piece_start_y + len(self.piece_types) * 30 + 15
        if 550 <= mx <= 800 and save_y <= my <= save_y + 25:
            self.save_results = not self.save_results
            return None

        # Select all / deselect all models
        sel_all_rect = pygame.Rect(50, 65, 80, 25)
        desel_all_rect = pygame.Rect(140, 65, 100, 25)
        if sel_all_rect.collidepoint(mx, my):
            self.model_selected = [True] * len(self.model_files)
        elif desel_all_rect.collidepoint(mx, my):
            self.model_selected = [False] * len(self.model_files)

        return None

    def _draw(self):
        self.screen.fill(WHITE)

        # Title
        text = self.title_font.render("Win-Taking Test", True, BLACK)
        self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, 15))

        # ── Models section ──
        text = self.font.render("Models:", True, BLACK)
        self.screen.blit(text, (50, 40))

        # Select all / deselect all
        sel_text = self.small_font.render("[Select All]", True, BTN_COLOR)
        self.screen.blit(sel_text, (50, 68))
        desel_text = self.small_font.render("[Deselect All]", True, BTN_COLOR)
        self.screen.blit(desel_text, (140, 68))

        model_start_y = 100
        for i in range(self.max_visible_models):
            idx = i + self.scroll_offset
            if idx >= len(self.model_files):
                break
            y = model_start_y + i * 30

            # Checkbox
            box_rect = pygame.Rect(50, y + 2, 20, 20)
            color = CHECKBOX_ON if self.model_selected[idx] else CHECKBOX_OFF
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, BLACK, box_rect, 1)
            if self.model_selected[idx]:
                pygame.draw.line(self.screen, WHITE, (53, y + 12), (57, y + 18), 2)
                pygame.draw.line(self.screen, WHITE, (57, y + 18), (67, y + 6), 2)

            label = self.small_font.render(self.model_files[idx], True, BLACK)
            self.screen.blit(label, (78, y + 3))

        # Scroll indicators
        if self.scroll_offset > 0:
            text = self.small_font.render("^ scroll up ^", True, DARK_GRAY)
            self.screen.blit(text, (50, model_start_y - 18))
        if self.scroll_offset + self.max_visible_models < len(self.model_files):
            y = model_start_y + self.max_visible_models * 30
            text = self.small_font.render("v scroll down v", True, DARK_GRAY)
            self.screen.blit(text, (50, y))

        # ── Sim counts section ──
        sim_start_y = 100
        text = self.font.render("Sim Counts:", True, BLACK)
        self.screen.blit(text, (550, 40))

        for i, s in enumerate(self.all_sims):
            y = sim_start_y + i * 35

            box_rect = pygame.Rect(550, y + 2, 20, 20)
            color = CHECKBOX_ON if self.sim_selected[i] else CHECKBOX_OFF
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, BLACK, box_rect, 1)
            if self.sim_selected[i]:
                pygame.draw.line(self.screen, WHITE, (553, y + 12), (557, y + 18), 2)
                pygame.draw.line(self.screen, WHITE, (557, y + 18), (567, y + 6), 2)

            label = self.font.render(str(s), True, BLACK)
            self.screen.blit(label, (580, y + 1))

        # ── Position count ──
        pos_y = sim_start_y + len(self.all_sims) * 35 + 30
        text = self.font.render("Positions to test:", True, BLACK)
        self.screen.blit(text, (550, pos_y))

        input_rect = pygame.Rect(700, pos_y + 30, 100, 30)
        border_color = BTN_COLOR if self.pos_input_active else GRAY
        pygame.draw.rect(self.screen, WHITE, input_rect)
        pygame.draw.rect(self.screen, border_color, input_rect, 2)
        text = self.font.render(self.pos_input_text, True, BLACK)
        self.screen.blit(text, (input_rect.x + 5, input_rect.y + 4))

        # ── Target piece types ──
        piece_start_y = sim_start_y + len(self.all_sims) * 35 + 40
        text = self.font.render("Target captures:", True, BLACK)
        self.screen.blit(text, (550, piece_start_y - 25))

        for i, plabel in enumerate(self.piece_labels):
            y = piece_start_y + i * 30

            box_rect = pygame.Rect(550, y + 2, 20, 20)
            color = CHECKBOX_ON if self.piece_selected[i] else CHECKBOX_OFF
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, BLACK, box_rect, 1)
            if self.piece_selected[i]:
                pygame.draw.line(self.screen, WHITE, (553, y + 12), (557, y + 18), 2)
                pygame.draw.line(self.screen, WHITE, (557, y + 18), (567, y + 6), 2)

            label = self.small_font.render(plabel, True, BLACK)
            self.screen.blit(label, (578, y + 3))

        # ── Save results checkbox ──
        save_y = piece_start_y + len(self.piece_types) * 30 + 15
        box_rect = pygame.Rect(550, save_y + 2, 20, 20)
        color = CHECKBOX_ON if self.save_results else CHECKBOX_OFF
        pygame.draw.rect(self.screen, color, box_rect)
        pygame.draw.rect(self.screen, BLACK, box_rect, 1)
        if self.save_results:
            pygame.draw.line(self.screen, WHITE, (553, save_y + 12), (557, save_y + 18), 2)
            pygame.draw.line(self.screen, WHITE, (557, save_y + 18), (567, save_y + 6), 2)
        label = self.small_font.render("Save results to JSON", True, BLACK)
        self.screen.blit(label, (578, save_y + 3))

        # ── Start button ──
        btn_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 70, 200, 45)
        models_selected = any(self.model_selected)
        sims_selected = any(self.sim_selected)
        pieces_selected = any(self.piece_selected)
        can_start = models_selected and sims_selected and pieces_selected

        color = BTN_COLOR if can_start else BTN_DISABLED
        mx, my = pygame.mouse.get_pos()
        if can_start and btn_rect.collidepoint(mx, my):
            color = BTN_HOVER
        pygame.draw.rect(self.screen, color, btn_rect, border_radius=8)
        text = self.font.render("Start Test", True, WHITE)
        self.screen.blit(text, (btn_rect.centerx - text.get_width() // 2,
                                btn_rect.centery - text.get_height() // 2))

        if not can_start:
            msg = ""
            if not models_selected:
                msg = "Select at least one model"
            elif not sims_selected:
                msg = "Select at least one sim count"
            elif not pieces_selected:
                msg = "Select at least one target piece type"
            text = self.small_font.render(msg, True, MISS_RED)
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2,
                                    WINDOW_HEIGHT - 20))

        pygame.display.flip()


# ═════════════════════════════════════════════════════════════════════════════
# Test runner with board display
# ═════════════════════════════════════════════════════════════════════════════

class WinTakingTest:
    def __init__(self, screen, model_paths, sim_counts, num_positions,
                 target_piece_types=None):
        self.screen = screen
        self.sim_counts = sorted(sim_counts)
        self.num_positions = num_positions
        self.target_piece_types = target_piece_types  # None = all types
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 32)
        self.piece_font = pygame.font.Font(None, int(SQUARE_SIZE * 0.7))

        # Load models
        self.models = []
        for path in model_paths:
            model, meta = AlphaZeroNet.load_checkpoint(path)
            evaluator = AlphaZeroEvaluator(model, device="cpu")
            iteration = meta.get("iteration", "?")
            label = f"iter {iteration}"
            self.models.append({
                "label": label,
                "evaluator": evaluator,
                "path": path,
            })

        # Results: list of position records
        self.results = []
        # Aggregate: {model_label: {sims: {"hits": N, "misses": N}}}
        self.aggregate = {}
        for m in self.models:
            self.aggregate[m["label"]] = {}
            for s in self.sim_counts:
                self.aggregate[m["label"]][s] = {"hits": 0, "misses": 0}

        # Display state
        self.game = None
        self.winning_moves = []
        self.winning_types = []
        self.status_lines = []
        self.positions_tested = 0
        self.games_played = 0
        self.phase = "generating"  # "generating", "testing", "done"
        self.current_test_model = 0
        self.current_test_sim = 0
        self.test_results_for_position = {}
        self.paused = False

    def run(self):
        """Main test loop. Returns results list."""
        self.game = ExtinctionChess()
        self.status_lines = ["Generating random positions..."]

        while self.positions_tested < self.num_positions:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return self.results
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return self.results
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused

            if self.paused:
                self._draw()
                self.clock.tick(30)
                continue

            if self.phase == "generating":
                self._generate_step()
            elif self.phase == "testing":
                self._test_step()

            self._draw()
            self.clock.tick(60)

        # Show final results
        self.phase = "done"
        self._show_results_screen()

        return self.results

    def _generate_step(self):
        """Make random moves until an instant win position arises."""
        if self.game.game_over:
            self.game = ExtinctionChess()
            self.games_played += 1

        legal = self.game.get_legal_moves()
        if not legal:
            self.game = ExtinctionChess()
            self.games_played += 1
            return

        # Check for instant win
        # If filtering by piece type, only stop when ALL winning moves
        # capture a targeted type (no easy king/queen captures available)
        all_winners = find_winning_moves(self.game)
        if self.target_piece_types is not None:
            winners = [
                (m, pt) for m, pt in all_winners
                if pt in self.target_piece_types
            ]
            # Skip if there are also winning moves outside the target set
            # (model could just take the easy capture instead)
            non_target = [
                (m, pt) for m, pt in all_winners
                if pt not in self.target_piece_types
            ]
            if non_target:
                winners = []
        else:
            winners = all_winners
        if winners:
            self.winning_moves = [m for m, _ in winners]
            self.winning_types = [pt for _, pt in winners]
            self.phase = "testing"
            self.current_test_model = 0
            self.current_test_sim = 0
            self.test_results_for_position = {}
            side = "White" if self.game.current_player == Color.WHITE else "Black"
            win_strs = ", ".join(
                f"{m}({pt.value if pt else '?'})" for m, pt in winners
            )
            self.status_lines = [
                f"Position {self.positions_tested + 1}/{self.num_positions} "
                f"(game #{self.games_played + 1})",
                f"{side} to move — winning capture(s): {win_strs}",
                "",
            ]
            return

        # Random move
        move = random.choice(legal)
        self.game.make_move(move)

    def _test_step(self):
        """Test current model at current sim count on the position."""
        model = self.models[self.current_test_model]
        sims = self.sim_counts[self.current_test_sim]
        evaluator = model["evaluator"]
        label = model["label"]

        # Run MCTS
        game_copy = copy_game(self.game)
        mv, root_val = mcts_search(
            game_copy, evaluator,
            num_simulations=sims,
            dirichlet_alpha=0, noise_weight=0,
            tactical_shortcuts=False,
        )

        if mv:
            mv_sorted = sorted(mv, key=lambda x: x[1], reverse=True)
            best_move = mv_sorted[0][0]
            best_visits = mv_sorted[0][1]

            # Check if model found a winning move
            is_hit = any(str(best_move) == str(wm) for wm in self.winning_moves)

            # Top 3 for display
            top3 = mv_sorted[:3]
            top3_str = ", ".join(f"{m}:{v}" for m, v in top3)

            if is_hit:
                self.aggregate[label][sims]["hits"] += 1
                marker = "HIT"
                color_tag = "green"
            else:
                self.aggregate[label][sims]["misses"] += 1
                marker = "MISS"
                color_tag = "red"

            self.test_results_for_position[(label, sims)] = {
                "move": str(best_move),
                "visits": best_visits,
                "hit": is_hit,
                "top3": [(str(m), v) for m, v in top3],
            }

            self.status_lines.append(
                f"  {label} @{sims} sims: {best_move} ({best_visits}v) "
                f"[{top3_str}] — {marker}"
            )
        else:
            self.test_results_for_position[(label, sims)] = {
                "move": None, "visits": 0, "hit": False, "top3": [],
            }
            self.aggregate[label][sims]["misses"] += 1
            self.status_lines.append(f"  {label} @{sims} sims: no moves — MISS")

        # Advance to next test
        self.current_test_sim += 1
        if self.current_test_sim >= len(self.sim_counts):
            self.current_test_sim = 0
            self.current_test_model += 1
            if self.current_test_model >= len(self.models):
                # Done with this position
                self._save_position_result()
                self.positions_tested += 1
                self.phase = "generating"

                # If any model found the win, start a fresh game
                # (otherwise the same position likely persists)
                any_hit = any(
                    r["hit"] for r in self.test_results_for_position.values()
                )
                if any_hit:
                    self.game = ExtinctionChess()
                    self.games_played += 1
                else:
                    # No model found it — continue with a random move
                    legal = self.game.get_legal_moves()
                    if legal:
                        self.game.make_move(random.choice(legal))

    def _save_position_result(self):
        """Save the results for the current position."""
        # Encode board state
        board_repr = []
        for rank in range(8):
            row = ""
            for file in range(8):
                piece = self.game.board.get_piece(Position(rank, file))
                if piece:
                    c = piece.piece_type.value
                    if piece.color == Color.BLACK:
                        c = c.lower()
                    row += c
                else:
                    row += "."
            board_repr.append(row)

        side = "white" if self.game.current_player == Color.WHITE else "black"
        winning_strs = [str(m) for m in self.winning_moves]
        captured_types = [pt.value if pt else "?" for pt in self.winning_types]

        record = {
            "position_index": self.positions_tested,
            "board": board_repr,
            "side_to_move": side,
            "winning_moves": winning_strs,
            "captured_types": captured_types,
            "tests": {},
        }
        for (label, sims), result in self.test_results_for_position.items():
            key = f"{label}@{sims}"
            record["tests"][key] = result

        self.results.append(record)

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self):
        self.screen.fill(WHITE)
        self._draw_board()
        self._draw_pieces()
        self._draw_winning_highlights()
        self._draw_panel()
        pygame.display.flip()

    def _draw_board(self):
        for rank in range(8):
            for file in range(8):
                x = BOARD_X + file * SQUARE_SIZE
                y = BOARD_Y + (7 - rank) * SQUARE_SIZE
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

                if file == 0:
                    label = self.small_font.render(str(rank + 1), True, BLACK)
                    self.screen.blit(label, (x + 2, y + 2))
                if rank == 0:
                    label = self.small_font.render(chr(ord('a') + file), True, BLACK)
                    self.screen.blit(label, (x + SQUARE_SIZE - 14, y + SQUARE_SIZE - 16))

    def _draw_pieces(self):
        if not self.game:
            return
        piece_chars = {
            PieceType.KING: 'K', PieceType.QUEEN: 'Q', PieceType.ROOK: 'R',
            PieceType.BISHOP: 'B', PieceType.KNIGHT: 'N', PieceType.PAWN: 'P',
        }
        for rank in range(8):
            for file in range(8):
                pos = Position(rank, file)
                piece = self.game.board.get_piece(pos)
                if piece:
                    x = BOARD_X + file * SQUARE_SIZE
                    y = BOARD_Y + (7 - rank) * SQUARE_SIZE
                    char = piece_chars[piece.piece_type]
                    if piece.color == Color.BLACK:
                        char = char.lower()
                    if piece.color == Color.WHITE:
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    t = self.piece_font.render(char, True, BLACK)
                                    r = t.get_rect(center=(x + SQUARE_SIZE//2 + dx,
                                                           y + SQUARE_SIZE//2 + dy))
                                    self.screen.blit(t, r)
                        t = self.piece_font.render(char, True, WHITE)
                    else:
                        t = self.piece_font.render(char, True, BLACK)
                    r = t.get_rect(center=(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2))
                    self.screen.blit(t, r)

    def _draw_winning_highlights(self):
        if self.phase != "testing" or not self.winning_moves:
            return
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        for m in self.winning_moves:
            # Highlight target square in green
            x = m.to_pos.file * SQUARE_SIZE
            y = (7 - m.to_pos.rank) * SQUARE_SIZE
            pygame.draw.rect(overlay, (0, 255, 0, 80), (x, y, SQUARE_SIZE, SQUARE_SIZE))
            # Highlight source square in yellow
            x = m.from_pos.file * SQUARE_SIZE
            y = (7 - m.from_pos.rank) * SQUARE_SIZE
            pygame.draw.rect(overlay, (255, 255, 0, 80), (x, y, SQUARE_SIZE, SQUARE_SIZE))
        self.screen.blit(overlay, (BOARD_X, BOARD_Y))

    def _draw_panel(self):
        y = BOARD_Y

        # Title / progress
        progress = f"Position {self.positions_tested}/{self.num_positions}"
        text = self.title_font.render(progress, True, BLACK)
        self.screen.blit(text, (PANEL_X, y))
        y += 35

        games_text = f"Games played: {self.games_played}"
        text = self.small_font.render(games_text, True, DARK_GRAY)
        self.screen.blit(text, (PANEL_X, y))
        y += 25

        if self.paused:
            text = self.font.render("PAUSED (Space to resume)", True, BTN_COLOR)
            self.screen.blit(text, (PANEL_X, y))
            y += 25
        y += 5

        # Aggregate table
        text = self.font.render("Hit Rate:", True, BLACK)
        self.screen.blit(text, (PANEL_X, y))
        y += 25

        # Header row
        header = f"{'Model':<12}"
        for s in self.sim_counts:
            header += f"{'@'+str(s):>8}"
        text = self.small_font.render(header, True, BLACK)
        self.screen.blit(text, (PANEL_X, y))
        y += 20

        # Separator
        pygame.draw.line(self.screen, GRAY, (PANEL_X, y), (PANEL_X + PANEL_W, y))
        y += 5

        for m in self.models:
            label = m["label"]
            row = f"{label:<12}"
            for s in self.sim_counts:
                h = self.aggregate[label][s]["hits"]
                mi = self.aggregate[label][s]["misses"]
                total = h + mi
                if total > 0:
                    pct = h / total * 100
                    row += f"{pct:>7.0f}%"
                else:
                    row += f"{'—':>8}"
            text = self.small_font.render(row, True, BLACK)
            self.screen.blit(text, (PANEL_X, y))
            y += 20
        y += 10

        # Raw counts row
        for m in self.models:
            label = m["label"]
            row = f"{label:<12}"
            for s in self.sim_counts:
                h = self.aggregate[label][s]["hits"]
                mi = self.aggregate[label][s]["misses"]
                row += f"{h:>4}/{h+mi:<3}"
            text = self.small_font.render(row, True, DARK_GRAY)
            self.screen.blit(text, (PANEL_X, y))
            y += 20
        y += 10

        # Status lines (last N that fit)
        pygame.draw.line(self.screen, GRAY, (PANEL_X, y), (PANEL_X + PANEL_W, y))
        y += 5
        max_lines = (WINDOW_HEIGHT - y - 30) // 16
        visible = self.status_lines[-max_lines:] if len(self.status_lines) > max_lines else self.status_lines
        for line in visible:
            if "MISS" in line:
                color = MISS_RED
            elif "HIT" in line:
                color = WIN_GREEN
            else:
                color = BLACK
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (PANEL_X, y))
            y += 16

        # Bottom: controls
        ctrl = self.small_font.render("Space: pause  |  Esc: stop early", True, DARK_GRAY)
        self.screen.blit(ctrl, (BOARD_X, WINDOW_HEIGHT - 20))

    def _show_results_screen(self):
        """Show final results. Enter goes to review, Esc closes."""
        go_to_review = False
        while not go_to_review:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_RETURN:
                        go_to_review = True

            self.screen.fill(WHITE)
            y = 30

            text = self.title_font.render("Win-Taking Test Results", True, BLACK)
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
            y += 50

            text = self.font.render(f"Positions tested: {self.positions_tested}", True, BLACK)
            self.screen.blit(text, (50, y))
            y += 30
            text = self.font.render(f"Games played: {self.games_played}", True, BLACK)
            self.screen.blit(text, (50, y))
            y += 40

            # Table header
            header = f"{'Model':<14}"
            for s in self.sim_counts:
                header += f"{'@'+str(s):>10}"
            text = self.font.render(header, True, BLACK)
            self.screen.blit(text, (50, y))
            y += 30
            pygame.draw.line(self.screen, BLACK, (50, y), (WINDOW_WIDTH - 50, y), 2)
            y += 10

            for m in self.models:
                label = m["label"]
                row = f"{label:<14}"
                for s in self.sim_counts:
                    h = self.aggregate[label][s]["hits"]
                    mi = self.aggregate[label][s]["misses"]
                    total = h + mi
                    if total > 0:
                        pct = h / total * 100
                        row += f"{h}/{total} ({pct:.0f}%)".rjust(10)
                    else:
                        row += f"{'—':>10}"
                text = self.font.render(row, True, BLACK)
                self.screen.blit(text, (50, y))
                y += 30

            y += 20
            text = self.small_font.render("Enter: review positions  |  Esc: close", True, DARK_GRAY)
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))

            pygame.display.flip()
            self.clock.tick(30)

        # After results screen, go to review
        self._review_positions()

    def _review_positions(self):
        """Browse tested positions one by one."""
        if not self.results:
            return
        idx = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        idx = min(idx + 1, len(self.results) - 1)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        idx = max(idx - 1, 0)
                    elif event.key == pygame.K_HOME:
                        idx = 0
                    elif event.key == pygame.K_END:
                        idx = len(self.results) - 1

            self._draw_review(idx)
            self.clock.tick(30)

    def _draw_review(self, idx):
        """Draw a single reviewed position."""
        self.screen.fill(WHITE)
        record = self.results[idx]

        # Draw board from saved state
        board_repr = record["board"]
        piece_chars_upper = {
            'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
            'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p',
        }

        for rank in range(8):
            for file in range(8):
                x = BOARD_X + file * SQUARE_SIZE
                y = BOARD_Y + (7 - rank) * SQUARE_SIZE
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

                if file == 0:
                    label = self.small_font.render(str(rank + 1), True, BLACK)
                    self.screen.blit(label, (x + 2, y + 2))
                if rank == 0:
                    label = self.small_font.render(chr(ord('a') + file), True, BLACK)
                    self.screen.blit(label, (x + SQUARE_SIZE - 14, y + SQUARE_SIZE - 16))

                char = board_repr[rank][file]
                if char != '.':
                    is_white = char.isupper()
                    if is_white:
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    t = self.piece_font.render(char, True, BLACK)
                                    r = t.get_rect(center=(x + SQUARE_SIZE//2 + dx,
                                                           y + SQUARE_SIZE//2 + dy))
                                    self.screen.blit(t, r)
                        t = self.piece_font.render(char, True, WHITE)
                    else:
                        t = self.piece_font.render(char, True, BLACK)
                    r = t.get_rect(center=(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2))
                    self.screen.blit(t, r)

        # Highlight winning moves
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        for wm_str in record["winning_moves"]:
            # Parse move string like "e2-e4" or "e7xd8=Q"
            # Extract from/to positions from the move string
            # Format: "a2-a3" or "a2xa3" or "a7-a8=Q"
            parts = wm_str.replace('x', '-').split('=')[0]
            squares = parts.split('-')
            if len(squares) == 2:
                from_file = ord(squares[0][0]) - ord('a')
                from_rank = int(squares[0][1]) - 1
                to_file = ord(squares[1][0]) - ord('a')
                to_rank = int(squares[1][1]) - 1

                fx = from_file * SQUARE_SIZE
                fy = (7 - from_rank) * SQUARE_SIZE
                pygame.draw.rect(overlay, (255, 255, 0, 80),
                                 (fx, fy, SQUARE_SIZE, SQUARE_SIZE))
                tx = to_file * SQUARE_SIZE
                ty = (7 - to_rank) * SQUARE_SIZE
                pygame.draw.rect(overlay, (0, 255, 0, 80),
                                 (tx, ty, SQUARE_SIZE, SQUARE_SIZE))
        self.screen.blit(overlay, (BOARD_X, BOARD_Y))

        # Panel
        py = BOARD_Y
        text = self.title_font.render(
            f"Position {idx + 1} / {len(self.results)}", True, BLACK)
        self.screen.blit(text, (PANEL_X, py))
        py += 35

        side = record["side_to_move"].capitalize()
        text = self.font.render(f"{side} to move", True, BLACK)
        self.screen.blit(text, (PANEL_X, py))
        py += 25

        captured = record.get("captured_types", [])
        if captured:
            win_strs = ", ".join(
                f"{m}({c})" for m, c in zip(record["winning_moves"], captured)
            )
        else:
            win_strs = ", ".join(record["winning_moves"])
        text = self.small_font.render(f"Winning: {win_strs}", True, WIN_GREEN)
        self.screen.blit(text, (PANEL_X, py))
        py += 30

        # Test results table
        text = self.font.render("Results:", True, BLACK)
        self.screen.blit(text, (PANEL_X, py))
        py += 25

        for key, result in record["tests"].items():
            move_str = result["move"] or "none"
            visits = result["visits"]
            hit = result["hit"]

            marker = "HIT" if hit else "MISS"
            color = WIN_GREEN if hit else MISS_RED

            top3_str = ""
            if result["top3"]:
                top3_str = " [" + ", ".join(
                    f"{m}:{v}" for m, v in result["top3"]) + "]"

            line = f"{key}: {move_str} ({visits}v){top3_str} — {marker}"
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (PANEL_X, py))
            py += 18

        # Navigation
        nav = self.small_font.render(
            "Left/Right: navigate  |  Home/End: first/last  |  Esc: close",
            True, DARK_GRAY)
        self.screen.blit(nav, (BOARD_X, WINDOW_HEIGHT - 20))

        pygame.display.flip()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Win-Taking Test")

    # Config screen
    config = ConfigScreen(screen)
    result = config.run()
    if result is None:
        pygame.quit()
        return

    model_paths, sim_counts, num_positions, save_results, target_pieces = result

    print(f"Models: {[os.path.basename(p) for p in model_paths]}")
    print(f"Sims: {sim_counts}")
    print(f"Positions: {num_positions}")
    print(f"Save results: {save_results}")
    if target_pieces:
        print(f"Target captures: {[pt.value for pt in target_pieces]}")
    else:
        print("Target captures: all")

    # Run test
    test = WinTakingTest(screen, model_paths, sim_counts, num_positions,
                         target_piece_types=target_pieces)
    results = test.run()

    # Print summary
    if results:
        print(f"\nWin-Taking Test Summary ({len(results)} positions)")
        print(f"{'Model':<14}", end="")
        for s in sim_counts:
            print(f"{'@'+str(s):>10}", end="")
        print()
        print("-" * (14 + 10 * len(sim_counts)))
        for m in test.models:
            label = m["label"]
            print(f"{label:<14}", end="")
            for s in sim_counts:
                h = test.aggregate[label][s]["hits"]
                mi = test.aggregate[label][s]["misses"]
                total = h + mi
                if total > 0:
                    pct = h / total * 100
                    print(f"{h}/{total}({pct:.0f}%)".rjust(10), end="")
                else:
                    print(f"{'—':>10}", end="")
            print()

    # Save results if requested
    if results and save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"win_taking_results_{timestamp}.json"
        )
        output = {
            "timestamp": timestamp,
            "models": [os.path.basename(p) for p in model_paths],
            "sim_counts": sim_counts,
            "num_positions_target": num_positions,
            "num_positions_tested": len(results),
            "aggregate": test.aggregate,
            "positions": results,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")

    pygame.quit()


if __name__ == "__main__":
    main()
