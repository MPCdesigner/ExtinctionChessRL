"""Timed match tool — play against a chosen model checkpoint under a clock.

Phase 1 scope:
  - Startup dialog: model + side + time controls (with odds via per-side settings)
  - Board + click-click move input + legal-move highlighting
  - Two clocks (mm:ss.d) — active side ticks in real time, low-time red
  - Model moves in a background thread using a time-based sim budget
  - Extinction, time forfeit → game-end banner
  - "New Game" button → restarts with same settings (pre-filled dialog)

NOT in Phase 1:
  - Pondering (worker thread continues MCTS during user's turn)
  - Post-game review (step through moves with revealed model evals)
  - Resign / draw offers

Layout (1200 × 800):

    [ banner or status bar (60 px) ]
    +----------------+ +--------------------+
    |                | |  Model             |
    |     Board      | |  clock             |
    |    (560x560)   | |                    |
    |                | |  History           |
    +----------------+ |  (scroll)          |
                       |                    |
                       |  You               |
                       |  clock             |
                       |                    |
                       |  [ New Game ]      |
                       +--------------------+
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import Color, PieceType, Position  # noqa: E402

# Reuse the board renderer from positional_eval — it's stateless.
from tools.positional_eval.board_widget import BoardWidget  # noqa: E402

from .engine import Engine  # noqa: E402
from .startup import show_startup_dialog  # noqa: E402
from .state import (  # noqa: E402
    MatchState, TimeControl,
    OUTCOME_ONGOING, OUTCOME_YOU_WIN_EXTINCT, OUTCOME_MODEL_WIN_EXTINCT,
    OUTCOME_YOU_FLAGGED, OUTCOME_MODEL_FLAGGED, OUTCOME_DRAW,
)


# ── Layout ──────────────────────────────────────────────────────────────────
SCREEN_W = 1200
SCREEN_H = 800
BANNER_H = 60
BOARD_SIZE = 560
BOARD_X = 20
BOARD_Y = BANNER_H + 20
SIDE_X = BOARD_X + BOARD_SIZE + 30
SIDE_Y = BANNER_H + 20
SIDE_W = SCREEN_W - SIDE_X - 20


# ── Small position wrapper (avoids typing PositionState here) ───────────────

class _MockPositionState:
    """BoardWidget expects a `position_state` object with `.get_game()` and
    `.get_piece_at(pos)`. We wrap our game to satisfy that."""
    def __init__(self, game):
        self._game = game

    def get_game(self):
        return self._game

    def get_piece_at(self, pos: Position):
        return self._game.board.get_piece(pos)


# ── Format helpers ──────────────────────────────────────────────────────────

def _fmt_clock(seconds: float) -> str:
    """mm:ss.d format. Sub-second precision only shown when < 20s remaining
    (matches chess.com convention)."""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = seconds - m * 60
    if seconds < 20:
        return f"{m:02d}:{s:04.1f}"
    return f"{m:02d}:{int(s):02d}"


def _move_str(move) -> str:
    """Compact move display: 'e2-e4' or 'e7-e8=Q'."""
    ff = chr(ord('a') + move.from_pos.file) + str(move.from_pos.rank + 1)
    tt = chr(ord('a') + move.to_pos.file) + str(move.to_pos.rank + 1)
    base = f"{ff}-{tt}"
    if move.promotion is not None:
        p = move.promotion
        # PieceType enum values
        promo_char = {
            PieceType.QUEEN: "Q",
            PieceType.ROOK: "R",
            PieceType.BISHOP: "B",
            PieceType.KNIGHT: "N",
            PieceType.KING: "K",
        }.get(p, "?")
        base += f"={promo_char}"
    return base


# ── The app ─────────────────────────────────────────────────────────────────

class TimedMatchApp:
    """Encapsulates the pygame loop + match state + engine coordination."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.screen = pygame.display.get_surface()
        self.font_banner = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_clock = pygame.font.SysFont("Consolas,Menlo,Courier", 40, bold=True)
        self.font_label = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_row = pygame.font.SysFont("Consolas,Menlo,Courier", 13)
        self.font_status = pygame.font.SysFont("Arial", 14)

        # Board renderer (stateless).
        self.board_widget = BoardWidget(BOARD_X, BOARD_Y, BOARD_SIZE)

        # Match state.
        user_color = Color.WHITE if settings["user_side"] == "W" else Color.BLACK
        user_tc = TimeControl(settings["user_base_seconds"],
                              settings["user_increment_seconds"])
        model_tc = TimeControl(settings["model_base_seconds"],
                               settings["model_increment_seconds"])
        self.state = MatchState(settings["model_path"], user_color,
                                user_tc, model_tc)

        # Engine (loads the model). Uses CPU — laptop; matches the user-
        # reported ~10 sims/sec figure. Change to "cuda" if you have one.
        self.engine = Engine(settings["model_path"], device="cpu")

        # Warmup measures sims/sec on the starting position. Do this
        # BEFORE we call state.start() so the clock isn't running yet.
        self.engine.warmup(self.state.game, sample_sims=30)

        # UI state
        self.selected_square: Optional[Position] = None
        self.legal_targets: set = set()
        self.status_message = (f"Model loaded (iter {self.engine.iteration}), "
                               f"{self.engine.sims_per_second:.1f} sims/sec measured. "
                               f"Good luck.")
        self.promotion_pending: Optional[Tuple[Position, Position]] = None
        self.new_game_button = pygame.Rect(0, 0, 0, 0)
        self.want_new_game = False

        # Whether the engine has an in-flight search.
        self._engine_running = False

        # Kick off the match.
        self.state.start()

    # ── Main loop ────────────────────────────────────────────────────────

    def run(self) -> bool:
        """Runs one game to completion (or until the user closes the window).

        Returns True if the user clicked 'New Game' (main() should restart
        with the same settings). Returns False if the window was closed
        (main() should exit).
        """
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self.state.check_flag()

            if self.want_new_game:
                return True

            # If it's the model's turn and no search is in flight, launch.
            if (self.state.is_model_turn() and not self._engine_running
                    and self.state.is_ongoing()):
                self._launch_model_search()

            # If a model search finished, apply the result.
            if self._engine_running and self.engine.is_done():
                self._apply_model_move()

            self._draw()
            pygame.display.flip()
            clock.tick(30)

    # ── Click handling ────────────────────────────────────────────────────

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        # New Game button click (always active)
        if self.new_game_button.collidepoint(pos):
            self.want_new_game = True
            return

        # Promotion overlay eats all other clicks while active
        if self.promotion_pending is not None:
            self._handle_promotion_click(pos)
            return

        # Only accept board clicks when it's the user's turn
        if not self.state.is_user_turn():
            return

        sq = self.board_widget.pixel_to_square(*pos)
        if sq is None:
            self.selected_square = None
            self.legal_targets = set()
            return

        piece = self.state.game.board.get_piece(sq)

        # If nothing selected: try to select a piece
        if self.selected_square is None:
            if piece is not None and piece.color == self.state.user_color:
                self.selected_square = sq
                self.legal_targets = {
                    m.to_pos for m in self.state.game.get_legal_moves()
                    if m.from_pos == sq
                }
            return

        # Something is selected: attempt a move OR reselect
        if piece is not None and piece.color == self.state.user_color:
            # Clicked own piece — reselect
            self.selected_square = sq
            self.legal_targets = {
                m.to_pos for m in self.state.game.get_legal_moves()
                if m.from_pos == sq
            }
            return

        # Clicked a legal target
        if sq in self.legal_targets:
            self._attempt_user_move(self.selected_square, sq)
        else:
            # Clicked elsewhere — deselect
            self.selected_square = None
            self.legal_targets = set()

    def _attempt_user_move(self, from_sq: Position, to_sq: Position) -> None:
        """Find the matching legal move, handling promotion (may open dialog)."""
        legal = [m for m in self.state.game.get_legal_moves()
                 if m.from_pos == from_sq and m.to_pos == to_sq]

        if not legal:
            return

        # Pawn promotion: multiple legal moves with different promotion types
        promo_moves = [m for m in legal if m.promotion is not None]
        if promo_moves:
            # Open promotion dialog; actual move applied on dialog click.
            self.promotion_pending = (from_sq, to_sq)
            return

        # Single, non-promotion move
        move = legal[0]
        applied = self.state.apply_move(move)
        if applied:
            self.selected_square = None
            self.legal_targets = set()
            self.status_message = f"You played {_move_str(move)}."

    def _handle_promotion_click(self, pos: Tuple[int, int]) -> None:
        """Promotion dialog is drawn in _draw; layout there mirrors this."""
        rects = self._promotion_rects()
        for label, rect in rects.items():
            if not rect.collidepoint(pos):
                continue
            if label == "Cancel":
                # Revert: put pawn back logically (nothing was applied)
                self.promotion_pending = None
                self.selected_square = None
                self.legal_targets = set()
                self.status_message = "Promotion cancelled."
                return
            # Otherwise, apply the promotion move with the chosen piece type
            from_sq, to_sq = self.promotion_pending
            promo_type = {
                "Q": PieceType.QUEEN, "R": PieceType.ROOK, "B": PieceType.BISHOP,
                "N": PieceType.KNIGHT, "K": PieceType.KING,
            }[label]
            for m in self.state.game.get_legal_moves():
                if (m.from_pos == from_sq and m.to_pos == to_sq
                        and m.promotion == promo_type):
                    self.state.apply_move(m)
                    self.status_message = f"You played {_move_str(m)}."
                    break
            self.promotion_pending = None
            self.selected_square = None
            self.legal_targets = set()
            return

    def _promotion_rects(self) -> dict:
        """Compute rects for the promotion dialog buttons. Centered over board."""
        labels = ["Q", "R", "B", "N", "K", "Cancel"]
        rects = {}
        btn_w, btn_h = 70, 42
        gap = 8
        total_w = len(labels) * btn_w + (len(labels) - 1) * gap
        start_x = BOARD_X + (BOARD_SIZE - total_w) // 2
        y = BOARD_Y + (BOARD_SIZE - btn_h) // 2
        for i, label in enumerate(labels):
            rects[label] = pygame.Rect(start_x + i * (btn_w + gap), y, btn_w, btn_h)
        return rects

    # ── Model move flow ───────────────────────────────────────────────────

    def _launch_model_search(self) -> None:
        budget_sec = self.state.model_thinking_budget_seconds()
        sim_budget = self.engine.sims_for_time_budget(budget_sec)
        self.engine.start_move_search(self.state.game, sim_budget)
        self._engine_running = True
        self.status_message = (f"Model thinking… (budget ≈ {budget_sec:.1f}s, "
                               f"{sim_budget} sims)")

    def _apply_model_move(self) -> None:
        result = self.engine.take_result()
        self._engine_running = False
        if result is None or result.get("move") is None:
            self.status_message = "Model produced no move — something's off."
            return
        move = result["move"]
        snap = result.get("search_snapshot")
        applied = self.state.apply_move(move, search_snapshot=snap)
        if applied:
            self.status_message = f"Model played {_move_str(move)}."

    # ── Drawing ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.screen.fill((245, 245, 250))
        self._draw_banner()
        self._draw_board()
        self._draw_side_panel()
        if self.promotion_pending is not None:
            self._draw_promotion_dialog()

    def _draw_banner(self) -> None:
        pygame.draw.rect(self.screen, (232, 232, 240),
                         pygame.Rect(0, 0, SCREEN_W, BANNER_H))
        pygame.draw.line(self.screen, (200, 200, 210),
                         (0, BANNER_H), (SCREEN_W, BANNER_H), 1)

        if self.state.outcome != OUTCOME_ONGOING:
            outcome_labels = {
                OUTCOME_YOU_WIN_EXTINCT:   ("You win — extinction",     (10, 100, 30)),
                OUTCOME_MODEL_WIN_EXTINCT: ("Model wins — extinction",  (140, 20, 20)),
                OUTCOME_YOU_FLAGGED:       ("You lose on time",         (140, 20, 20)),
                OUTCOME_MODEL_FLAGGED:     ("Model loses on time — you win", (10, 100, 30)),
                OUTCOME_DRAW:              ("Draw",                     (60, 60, 60)),
            }
            label, color = outcome_labels.get(self.state.outcome, ("?", (0, 0, 0)))
            text = self.font_banner.render(label, True, color)
            self.screen.blit(text, (20, (BANNER_H - text.get_height()) // 2))
        else:
            turn_str = ("Your turn" if self.state.is_user_turn()
                        else "Model's turn")
            text = self.font_banner.render(turn_str, True, (20, 20, 30))
            self.screen.blit(text, (20, (BANNER_H - text.get_height()) // 2))

        # Status message (right side of banner)
        if self.status_message:
            text = self.font_status.render(self.status_message, True, (60, 60, 80))
            self.screen.blit(text, (SCREEN_W - text.get_width() - 20,
                                    (BANNER_H - text.get_height()) // 2))

    def _draw_board(self) -> None:
        ps = _MockPositionState(self.state.game)
        last_from = last_to = None
        if self.state.moves:
            last = self.state.moves[-1]
            last_from = last.move.from_pos
            last_to = last.move.to_pos
        self.board_widget.draw(
            self.screen, ps,
            selected=self.selected_square,
            legal_targets=self.legal_targets or None,
            last_move_from=last_from,
            last_move_to=last_to,
        )

    def _draw_side_panel(self) -> None:
        panel = pygame.Rect(SIDE_X, SIDE_Y, SIDE_W, SCREEN_H - SIDE_Y - 20)
        pygame.draw.rect(self.screen, (255, 255, 255), panel)
        pygame.draw.rect(self.screen, (200, 200, 210), panel, width=1)

        # Model clock (top). Highlight if it's model's turn OR if low.
        self._draw_clock_block(
            x=SIDE_X + 12, y=SIDE_Y + 12,
            label=f"Model  (iter {self.engine.iteration})  {self.state.model_tc.label()}",
            seconds=self.state.model_clock_display(),
            ticking=self.state.is_model_turn(),
        )

        # Your clock (bottom-ish above the New Game button).
        your_y = panel.bottom - 60 - 90
        self._draw_clock_block(
            x=SIDE_X + 12, y=your_y,
            label=f"You  ({self.settings['user_side']})  {self.state.user_tc.label()}",
            seconds=self.state.user_clock_display(),
            ticking=self.state.is_user_turn(),
        )

        # Move history sidebar (between the two clocks).
        hist_y = SIDE_Y + 12 + 90 + 8
        hist_h = your_y - hist_y - 8
        self._draw_history(SIDE_X + 12, hist_y, panel.right - SIDE_X - 24, hist_h)

        # New Game button.
        btn = pygame.Rect(SIDE_X + 12, panel.bottom - 46,
                          panel.right - SIDE_X - 24, 34)
        self.new_game_button = btn
        pygame.draw.rect(self.screen, (240, 240, 250), btn)
        pygame.draw.rect(self.screen, (100, 100, 130), btn, width=1)
        label = self.font_label.render("New Game", True, (30, 30, 60))
        self.screen.blit(label, label.get_rect(center=btn.center))

    def _draw_clock_block(self, x: int, y: int, label: str,
                          seconds: float, ticking: bool) -> None:
        """A clock display: label above, big mm:ss below."""
        # Label
        lbl_surf = self.font_label.render(label, True, (60, 60, 90))
        self.screen.blit(lbl_surf, (x, y))

        # Clock text
        clock_str = _fmt_clock(seconds)
        color = (30, 30, 30)
        if seconds < 10:
            color = (200, 20, 20)  # red urgency
        elif seconds < 30:
            color = (200, 100, 20)  # amber
        if ticking:
            # Small pulse hint via a background box.
            box = pygame.Rect(x - 4, y + 22, 220, 55)
            pygame.draw.rect(self.screen, (250, 245, 200), box)
            pygame.draw.rect(self.screen, (200, 170, 60), box, width=1)
        clk_surf = self.font_clock.render(clock_str, True, color)
        self.screen.blit(clk_surf, (x, y + 26))

    def _draw_history(self, x: int, y: int, w: int, h: int) -> None:
        title = self.font_label.render("Moves", True, (60, 60, 90))
        self.screen.blit(title, (x, y))

        # Rows
        row_y = y + 20
        row_h = 18
        # Show the last N moves that fit
        n_visible = max(1, h // row_h - 1)
        visible = self.state.moves[-n_visible:]
        start_ply = len(self.state.moves) - len(visible)
        for i, rec in enumerate(visible):
            side_letter = "W" if rec.side == Color.WHITE else "B"
            ply_num = start_ply + i + 1
            text = f"{ply_num:>3}. {side_letter} {_move_str(rec.move)}"
            surf = self.font_row.render(text, True, (30, 30, 30))
            self.screen.blit(surf, (x, row_y + i * row_h))

    def _draw_promotion_dialog(self) -> None:
        # Dim the board area behind the buttons a bit.
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 90))
        self.screen.blit(overlay, (BOARD_X, BOARD_Y))

        rects = self._promotion_rects()
        for label, rect in rects.items():
            bg = (250, 250, 245) if label != "Cancel" else (255, 230, 230)
            pygame.draw.rect(self.screen, bg, rect)
            pygame.draw.rect(self.screen, (80, 80, 80), rect, width=1)
            label_font = self.font_label if label == "Cancel" else self.font_clock
            surf = label_font.render(label, True, (30, 30, 30))
            self.screen.blit(surf, surf.get_rect(center=rect.center))


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.display.set_caption("Extinction Chess — Timed Match")
    pygame.display.set_mode((SCREEN_W, SCREEN_H))

    # Persistent defaults across "New Game" restarts.
    defaults = {
        "default_model_path": None,
        "default_side": "W",
        "default_your_min": 5,
        "default_your_inc": 3,
        "default_model_min": 5,
        "default_model_inc": 3,
    }

    while True:
        settings = show_startup_dialog(**defaults)
        if settings is None:
            pygame.quit()
            return

        # Update defaults from what the user chose so 'New Game' repeats it.
        defaults = {
            "default_model_path": settings["model_path"],
            "default_side": settings["user_side"],
            "default_your_min": settings["user_base_seconds"] // 60,
            "default_your_inc": settings["user_increment_seconds"],
            "default_model_min": settings["model_base_seconds"] // 60,
            "default_model_inc": settings["model_increment_seconds"],
        }

        app = TimedMatchApp(settings)
        wants_restart = app.run()
        if not wants_restart:
            pygame.quit()
            return
        # else: loop back to the startup dialog with defaults pre-filled.


if __name__ == "__main__":
    main()
