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
import time
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

        # Phase 2 pondering state — when it's the model's turn, main computes
        # a deadline (monotonic time) and lets the engine keep pondering
        # until then. When deadline passes, main reads the current best move.
        # None means "no deadline set yet" (either user's turn or we already
        # played the model's move for this turn).
        self._model_move_deadline: Optional[float] = None

        # Phase 3 review state — None during play, int in [-1, len(moves)-1]
        # during review (game over). -1 = initial position, N = after move N.
        # Managed by _enter_review_mode / _navigate_review.
        self._review_index: Optional[int] = None
        self._review_game: Optional = None            # ExtinctionChess snapshot
        # Rects for clickable review widgets (recomputed each draw).
        self._review_prev_button = pygame.Rect(0, 0, 0, 0)
        self._review_next_button = pygame.Rect(0, 0, 0, 0)
        self._review_history_rects: List[Tuple[int, pygame.Rect]] = []

        # Kick off the match: start pondering from the initial position.
        self.engine.start_from(self.state.game)
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
                if event.type == pygame.KEYDOWN and self._review_index is not None:
                    if event.key == pygame.K_LEFT:
                        self._navigate_review(-1)
                    elif event.key == pygame.K_RIGHT:
                        self._navigate_review(+1)
                    elif event.key == pygame.K_HOME:
                        self._set_review_index(-1)
                    elif event.key == pygame.K_END:
                        self._set_review_index(len(self.state.moves) - 1)

            self.state.check_flag()

            if self.want_new_game:
                self.engine.stop()
                return True

            # If game just ended (from a move or a flag), auto-enter review.
            if (not self.state.is_ongoing() and self._review_index is None):
                self._enter_review_mode()

            # Model-turn handling (Phase 2 pondering flow):
            # 1. When it becomes model's turn: set a deadline based on
            #    time budget. Engine has already been pondering during
            #    user's turn.
            # 2. Every frame after that: if deadline has passed, take
            #    the current best move from engine and play it.
            if (self.state.is_ongoing() and self.state.is_model_turn()):
                if self._model_move_deadline is None:
                    budget_sec = self.state.model_thinking_budget_seconds()
                    self._model_move_deadline = time.monotonic() + budget_sec
                    self.status_message = (
                        f"Model thinking (budget ≈ {budget_sec:.1f}s, "
                        f"pondering during your turn adds to this)…")
                elif time.monotonic() >= self._model_move_deadline:
                    self._apply_model_move()
                    self._model_move_deadline = None

            # If we transitioned to game-over during the model's think,
            # clear the deadline so we don't retry.
            if not self.state.is_ongoing():
                self._model_move_deadline = None

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

        # Review-mode clicks (navigation + click-to-jump on history)
        if self._review_index is not None:
            if self._review_prev_button.collidepoint(pos):
                self._navigate_review(-1)
                return
            if self._review_next_button.collidepoint(pos):
                self._navigate_review(+1)
                return
            for ply, rect in self._review_history_rects:
                if rect.collidepoint(pos):
                    self._set_review_index(ply)
                    return
            # Board is not interactive in review mode.
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
            # Tell the engine to descend into your move so it can keep
            # pondering from the new position.
            self.engine.descend(move)

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
                    self.engine.descend(m)
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

    # ── Review mode (Phase 3) ────────────────────────────────────────────

    def _enter_review_mode(self) -> None:
        """Called when the game ends. Positions the cursor at the final move
        and populates the review board. Engine is left alone — it'll idle
        naturally on the terminal position."""
        self.engine.stop()
        n = len(self.state.moves)
        # Start at the final position so the reviewer sees the outcome.
        # Cursor -1 means "before any moves" (initial position). If no moves
        # were played (unlikely but possible: instant flag on move 1?) we
        # still enter review at -1.
        self._set_review_index(n - 1 if n > 0 else -1)
        self.status_message = (
            "Review mode: click any move (or use ← / → keys) to jump. "
            "Model's search stats are revealed for each of its moves.")

    def _set_review_index(self, idx: int) -> None:
        idx = max(-1, min(len(self.state.moves) - 1, idx))
        self._review_index = idx
        # Reconstruct the position and cache it. Cheap for extinction chess
        # (game lengths are ~40 moves).
        self._review_game = self.state.reconstruct_at(idx)
        # Clear per-position UI state so it doesn't leak from play mode.
        self.selected_square = None
        self.legal_targets = set()

    def _navigate_review(self, delta: int) -> None:
        if self._review_index is None:
            return
        self._set_review_index(self._review_index + delta)

    # ── Model move flow (Phase 2: ponder-aware) ──────────────────────────

    def _apply_model_move(self) -> None:
        """Pull the model's current best move from the ponder tree and play it."""
        result = self.engine.get_current_result()
        if result is None or result.get("move") is None:
            # Engine had no result yet — rare, but can happen if the position
            # has extremely few legal moves + the ponder chunk hasn't run.
            # Give it a tiny bit more time.
            time.sleep(0.1)
            result = self.engine.get_current_result()
            if result is None or result.get("move") is None:
                self.status_message = ("Model has no result yet — waiting…")
                self._model_move_deadline = time.monotonic() + 1.0
                return
        move = result["move"]
        snap = result.get("search_snapshot")
        applied = self.state.apply_move(move, search_snapshot=snap)
        if applied:
            sim_count = snap.get("sim_count", 0) if snap else 0
            self.status_message = (
                f"Model played {_move_str(move)} "
                f"({sim_count} sims accumulated).")
            # Engine descends into its own move so it can start pondering
            # from the new position (waiting for the user's move).
            self.engine.descend(move)

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
        # In review mode, show the reconstructed position at the current
        # review index. Otherwise show the live game.
        if self._review_index is not None and self._review_game is not None:
            display_game = self._review_game
            # Highlight the move THAT WAS PLAYED to reach this position
            # (i.e., the move at index _review_index, if any).
            if 0 <= self._review_index < len(self.state.moves):
                rec = self.state.moves[self._review_index]
                last_from, last_to = rec.move.from_pos, rec.move.to_pos
            else:
                last_from = last_to = None
        else:
            display_game = self.state.game
            last_from = last_to = None
            if self.state.moves:
                last = self.state.moves[-1]
                last_from, last_to = last.move.from_pos, last.move.to_pos

        ps = _MockPositionState(display_game)
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

        # New Game button (bottom, always visible).
        btn = pygame.Rect(SIDE_X + 12, panel.bottom - 46,
                          panel.right - SIDE_X - 24, 34)
        self.new_game_button = btn
        pygame.draw.rect(self.screen, (240, 240, 250), btn)
        pygame.draw.rect(self.screen, (100, 100, 130), btn, width=1)
        label = self.font_label.render("New Game", True, (30, 30, 60))
        self.screen.blit(label, label.get_rect(center=btn.center))

        if self._review_index is not None:
            self._draw_review_panel(panel)
        else:
            self._draw_play_panel(panel)

    def _draw_play_panel(self, panel: pygame.Rect) -> None:
        """Live-game side panel: model clock, engine status, history, your clock."""
        # Model clock (top). Highlight if it's model's turn OR if low.
        self._draw_clock_block(
            x=SIDE_X + 12, y=SIDE_Y + 12,
            label=f"Model  (iter {self.engine.iteration})  {self.state.model_tc.label()}",
            seconds=self.state.model_clock_display(),
            ticking=self.state.is_model_turn(),
        )

        # Engine status line — cheap read, updates every frame.
        eng = self.engine.get_status_snapshot()
        eng_str = (f"engine: {eng['state']} | root sims: {eng['sim_count']}"
                   + (f"  (top move: {eng['top_visits']})"
                      if eng['top_visits'] else ""))
        eng_surf = self.font_row.render(eng_str, True, (60, 60, 90))
        self.screen.blit(eng_surf, (SIDE_X + 12, SIDE_Y + 12 + 90))

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

    def _draw_review_panel(self, panel: pygame.Rect) -> None:
        """Post-game review side panel: navigation, snapshot info, clickable history."""
        pad = 12
        cx = SIDE_X + pad
        cy = SIDE_Y + pad
        width = panel.right - SIDE_X - pad * 2

        # Header: "Review: move X of Y" + prev/next buttons
        n = len(self.state.moves)
        cur = self._review_index + 1  # 1-indexed for display; 0 means initial
        header = f"Review: move {cur}/{n}" if cur > 0 else f"Review: initial position (of {n})"
        header_surf = self.font_label.render(header, True, (30, 30, 60))
        self.screen.blit(header_surf, (cx, cy))
        cy += 24

        # Prev / Next buttons
        btn_w = 88
        prev_rect = pygame.Rect(cx, cy, btn_w, 26)
        next_rect = pygame.Rect(cx + btn_w + 8, cy, btn_w, 26)
        self._review_prev_button = prev_rect
        self._review_next_button = next_rect
        for rect, txt, enabled in [
            (prev_rect, "◀ Prev", self._review_index > -1),
            (next_rect, "Next ▶", self._review_index < n - 1),
        ]:
            bg = (240, 240, 250) if enabled else (220, 220, 220)
            pygame.draw.rect(self.screen, bg, rect)
            pygame.draw.rect(self.screen, (100, 100, 130), rect, width=1)
            fg = (30, 30, 60) if enabled else (150, 150, 150)
            surf = self.font_label.render(txt, True, fg)
            self.screen.blit(surf, surf.get_rect(center=rect.center))
        cy += 34

        # Reconstructed clocks at this position
        user_t, model_t = self.state.clocks_at(self._review_index)
        clk = self.font_row.render(
            f"Model: {_fmt_clock(model_t)}   You: {_fmt_clock(user_t)}",
            True, (60, 60, 90))
        self.screen.blit(clk, (cx, cy))
        cy += 20

        # Search snapshot (only for model moves; user moves have no snapshot).
        cy = self._draw_snapshot_block(cx, cy, width)

        # Clickable move history (rest of panel above New Game button).
        hist_bottom = panel.bottom - 60
        self._draw_review_history(cx, cy + 4, width, hist_bottom - cy - 4)

    def _draw_snapshot_block(self, cx: int, cy: int, width: int) -> int:
        """Render the search snapshot for the currently-reviewed move.
        Returns the y-coordinate just below the snapshot (for the next
        widget to place itself). Returns cy unchanged if nothing to show."""
        idx = self._review_index
        if idx < 0 or idx >= len(self.state.moves):
            # Initial position — no snapshot.
            note = self.font_row.render(
                "(initial position — no move played yet)",
                True, (120, 120, 130))
            self.screen.blit(note, (cx, cy))
            return cy + 20

        rec = self.state.moves[idx]
        side_letter = "W" if rec.side == Color.WHITE else "B"
        header = self.font_label.render(
            f"Move {idx + 1}: {side_letter} {_move_str(rec.move)}"
            f"  (thought {rec.thinking_time_seconds:.1f}s)",
            True, (30, 30, 60))
        self.screen.blit(header, (cx, cy))
        cy += 22

        snap = rec.search_snapshot
        if snap is None:
            # User's move — no MCTS ran, no snapshot to show.
            note = self.font_row.render(
                "(your move — no engine analysis stored)",
                True, (120, 120, 130))
            self.screen.blit(note, (cx, cy))
            return cy + 20

        # Model's move — show sim count, root value, top-N moves.
        sim_ct = snap.get("sim_count", 0)
        rv = snap.get("root_value", 0.0)
        # root_value is from the mover's perspective — flip to White's
        # perspective for consistency with positional_eval tool convention.
        white_val = rv if rec.side == Color.WHITE else -rv
        info = self.font_row.render(
            f"engine sims: {sim_ct}   value (W): {white_val:+.3f}",
            True, (60, 60, 90))
        self.screen.blit(info, (cx, cy))
        cy += 20

        # Top moves table
        top_moves = snap.get("top_moves", [])[:8]  # show up to 8
        for tm in top_moves:
            m_str = self._snapshot_move_str(tm)
            visits = tm.get("visits", 0)
            prob = tm.get("prob", 0.0) * 100
            line = f"  {m_str:<10s} {visits:>4d}  {prob:>5.1f}%"
            surf = self.font_row.render(line, True, (30, 30, 30))
            self.screen.blit(surf, (cx, cy))
            cy += 16
        return cy + 4

    def _snapshot_move_str(self, tm: dict) -> str:
        """Render a top-move dict (from search_snapshot) as 'e2-e4' etc."""
        fr = chr(ord('a') + tm["from"][1]) + str(tm["from"][0] + 1)
        to = chr(ord('a') + tm["to"][1]) + str(tm["to"][0] + 1)
        base = f"{fr}-{to}"
        if tm.get("promotion"):
            promo_char = {"Q": "Q", "R": "R", "B": "B", "N": "N", "K": "K"}.get(
                tm["promotion"], "?")
            base += f"={promo_char}"
        return base

    def _draw_review_history(self, x: int, y: int, w: int, h: int) -> None:
        """Clickable move history. Populates self._review_history_rects
        with (ply_index, rect) pairs for click detection."""
        title = self.font_label.render("Move history", True, (60, 60, 90))
        self.screen.blit(title, (x, y))

        row_y = y + 20
        row_h = 16
        n_visible = max(1, (h - 20) // row_h)

        # Show a window of moves centered around the current review index.
        moves = self.state.moves
        cur = max(0, self._review_index)  # clamp -1 → 0 for windowing
        start = max(0, min(len(moves) - n_visible, cur - n_visible // 2))
        end = min(len(moves), start + n_visible)

        self._review_history_rects = []
        for i, rec in enumerate(moves[start:end], start=start):
            side_letter = "W" if rec.side == Color.WHITE else "B"
            selected = (i == self._review_index)
            rect = pygame.Rect(x, row_y + (i - start) * row_h, w, row_h - 1)
            if selected:
                pygame.draw.rect(self.screen, (255, 245, 200), rect)
            self._review_history_rects.append((i, rect))
            text = f"{i + 1:>3}. {side_letter} {_move_str(rec.move)}"
            fg = (10, 10, 60) if selected else (30, 30, 30)
            surf = self.font_row.render(text, True, fg)
            self.screen.blit(surf, (x + 4, row_y + (i - start) * row_h))

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
        try:
            wants_restart = app.run()
        finally:
            # Clean up the engine's worker thread regardless of exit reason.
            app.engine.shutdown()
        if not wants_restart:
            pygame.quit()
            return
        # else: loop back to the startup dialog with defaults pre-filled.


if __name__ == "__main__":
    main()
