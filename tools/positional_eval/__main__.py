"""Positional evaluation tool — main entry point.

Run from the project root:
    python -m tools.positional_eval

Startup flow:
  1. tkinter file picker for model checkpoints (multi-select)
  2. Models load onto GPU if available, else CPU
  3. pygame window opens with:
      - Board on the left
      - Palette (Construction mode only) below the board
      - Controls row at the top
      - Eval columns filling the right side (empty until you click Evaluate)
      - Settings panel toggles as an overlay on the right side

Click "Evaluate" to run the currently-enabled models at each sim count
allowed by the max-sims setting. Runs synchronously — UI freezes for a few
seconds to a minute depending on model count and max sims.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, List, Optional

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import Color, Position, PieceType  # noqa: E402

from .board_widget import BoardWidget  # noqa: E402
from .palette_widget import PaletteWidget  # noqa: E402
from .controls_widget import (  # noqa: E402
    ControlsWidget,
    ACT_EVALUATE, ACT_TOGGLE_MODE, ACT_SET_TURN_W, ACT_SET_TURN_B,
    ACT_STEP_BACK, ACT_STEP_FWD, ACT_SAVE, ACT_LOAD, ACT_TOGGLE_SETTINGS,
)
from .settings_panel import SettingsPanel  # noqa: E402
from .eval_column import EvalColumn  # noqa: E402
from .position_state import PositionState, Mode  # noqa: E402
from .model_manager import ModelManager, EvalResult  # noqa: E402
from .startup_dialog import pick_models  # noqa: E402


# ── Screen layout ────────────────────────────────────────────────────────────
SCREEN_W = 1500
SCREEN_H = 900
CONTROLS_HEIGHT = 40
BOARD_SIZE = 520
BOARD_X = 12
BOARD_Y = CONTROLS_HEIGHT + 12
PALETTE_Y = BOARD_Y + BOARD_SIZE + 12
STATUS_Y = PALETTE_Y  # if palette not shown, status text goes here
EVAL_AREA_X = BOARD_X + BOARD_SIZE + 20
EVAL_AREA_Y = CONTROLS_HEIGHT + 12
COLUMN_GAP = 8


class App:
    def __init__(self, model_paths: List[str]):
        pygame.init()
        pygame.display.set_caption("Extinction Chess — Positional Evaluator")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        self.font_status = pygame.font.SysFont("Arial", 13)
        self.font_notice = pygame.font.SysFont("Arial", 12)

        # State
        self.state = PositionState(Mode.GAME_SETUP)
        self.model_mgr = ModelManager(model_paths)
        self.selected_square: Optional[Position] = None
        # results[model_index][sim_count] = EvalResult
        self.results: Dict[int, Dict[int, EvalResult]] = {}
        # Horizontal scroll (column offset) for the eval area
        self.column_scroll = 0
        # Vertical scroll offset (in pixels) applied to each visible column
        self.vert_scroll = 0
        # Transient status message
        self.status_message: str = ""

        # Live evaluation progress (updated by worker thread, read by main).
        # Simple dict + Lock is enough — we're single-writer, single-reader.
        self._progress_lock = threading.Lock()
        self._progress = {
            "active": False,
            "model": "",
            "sim_count": 0,
            "sims_done": 0,
            "sims_total": 0,
            "cells_done": 0,
            "cells_total": 0,
        }
        self._eval_thread: Optional[threading.Thread] = None
        self._eval_results: Optional[Dict[int, Dict[int, EvalResult]]] = None
        self._eval_error: Optional[str] = None

        # UI widgets
        self.board = BoardWidget(BOARD_X, BOARD_Y, BOARD_SIZE)
        self.palette = PaletteWidget(BOARD_X, PALETTE_Y)
        self.controls = ControlsWidget(0, 0, SCREEN_W)
        self.eval_col = EvalColumn()
        self.settings = SettingsPanel(
            SCREEN_W - 340, CONTROLS_HEIGHT + 10,
            SCREEN_H - CONTROLS_HEIGHT - 20,
            self.model_mgr.get_labels(),
        )

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        running = True
        while running:
            self._sync_controls()
            self._poll_eval_thread()
            self._draw_frame()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Ignore clicks while eval thread is running to avoid
                    # accidentally editing the position mid-evaluation.
                    if self._eval_thread and self._eval_thread.is_alive():
                        continue
                    if event.button == 1:
                        self._handle_click(event.pos)
                    elif event.button in (4, 5):
                        direction = -1 if event.button == 4 else 1
                        self._handle_scroll(event.pos, direction)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.column_scroll = max(0, self.column_scroll - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.column_scroll = min(
                            max(0, len(self._visible_column_indices()) - 1),
                            self.column_scroll + 1,
                        )
                    elif event.key == pygame.K_UP:
                        self.vert_scroll = max(0, self.vert_scroll - 40)
                    elif event.key == pygame.K_DOWN:
                        self.vert_scroll += 40

            self.clock.tick(60)

        pygame.quit()

    def _poll_eval_thread(self) -> None:
        """Called each frame from main. Reap results if the worker finished."""
        if self._eval_thread is None:
            return
        if self._eval_thread.is_alive():
            return
        # Worker finished — pull results / error and clear.
        self._eval_thread = None
        if self._eval_error is not None:
            self.status_message = f"Evaluation error: {self._eval_error}"
            self._eval_error = None
        elif self._eval_results is not None:
            self.results = self._eval_results
            self._eval_results = None
            self.status_message = "Evaluation complete."
            self.column_scroll = 0
            self.vert_scroll = 0
        with self._progress_lock:
            self._progress["active"] = False

    # ── Event handling ─────────────────────────────────────────────────────

    def _handle_click(self, pos):
        mx, my = pos

        # 1. Settings panel — if open, consumes clicks in its area
        if self.settings.is_open() and self.settings.handle_click(mx, my):
            return

        # 2. Controls row
        action = self.controls.handle_click(mx, my)
        if action:
            self._dispatch(action)
            return

        # 3. Board
        sq = self.board.pixel_to_square(mx, my)
        if sq is not None:
            self._handle_board_click(sq)
            return

        # 4. Palette (construction mode only)
        if self.state.mode == Mode.CONSTRUCTION:
            if self.palette.handle_click(mx, my):
                return

        # 5. Horizontal scroll bar area for eval columns (arrow buttons)
        if self._handle_scroll_arrows(mx, my):
            return

    def _handle_scroll(self, pos, direction: int):
        """Mouse-wheel vertical scroll when hovering the eval area."""
        mx, my = pos
        if mx >= EVAL_AREA_X:
            self.vert_scroll = max(0, self.vert_scroll + direction * 40)

    def _handle_scroll_arrows(self, mx: int, my: int) -> bool:
        """Left/Right arrow buttons above the eval columns."""
        left_rect = pygame.Rect(EVAL_AREA_X, EVAL_AREA_Y - 4, 24, 22)
        right_rect = pygame.Rect(EVAL_AREA_X + 30, EVAL_AREA_Y - 4, 24, 22)
        if left_rect.collidepoint(mx, my):
            self.column_scroll = max(0, self.column_scroll - 1)
            return True
        if right_rect.collidepoint(mx, my):
            n = len(self._visible_column_indices())
            if n > 0:
                self.column_scroll = min(n - 1, self.column_scroll + 1)
            return True
        return False

    def _handle_board_click(self, sq: Position):
        if self.state.mode == Mode.CONSTRUCTION:
            pt, col = self.palette.get_selection()
            if pt is None:
                self.state.remove_piece(sq)
            else:
                self.state.place_piece(sq, pt, col)
            self.results = {}   # invalidate cached results
            self.selected_square = None
            return

        # Game setup mode: click piece then click destination
        if self.selected_square is None:
            piece = self.state.get_piece_at(sq)
            if piece and piece.color == self.state.get_current_player():
                self.selected_square = sq
        else:
            if sq == self.selected_square:
                self.selected_square = None
                return
            # Attempt the move (auto-promote to queen for pawn promotions)
            promo = None
            piece = self.state.get_piece_at(self.selected_square)
            if piece and piece.piece_type == PieceType.PAWN:
                if (piece.color == Color.WHITE and sq.rank == 7) or \
                   (piece.color == Color.BLACK and sq.rank == 0):
                    promo = PieceType.QUEEN
            ok = self.state.make_move(self.selected_square, sq, promotion=promo)
            self.selected_square = None
            if ok:
                self.results = {}   # invalidate

    # ── Button dispatch ────────────────────────────────────────────────────

    def _dispatch(self, action: str) -> None:
        if action == ACT_EVALUATE:
            self._run_evaluation()
        elif action == ACT_TOGGLE_MODE:
            new_mode = (Mode.CONSTRUCTION if self.state.mode == Mode.GAME_SETUP
                        else Mode.GAME_SETUP)
            self.state = PositionState(new_mode)
            self.results = {}
            self.selected_square = None
            self.status_message = f"Switched to {new_mode.value} mode"
        elif action == ACT_SET_TURN_W:
            self.state.set_current_player(Color.WHITE)
            self.results = {}
        elif action == ACT_SET_TURN_B:
            self.state.set_current_player(Color.BLACK)
            self.results = {}
        elif action == ACT_STEP_BACK:
            if self.state.step_back():
                self.results = {}
                self.selected_square = None
        elif action == ACT_STEP_FWD:
            if self.state.step_forward():
                self.results = {}
                self.selected_square = None
        elif action == ACT_SAVE:
            self._save_position()
        elif action == ACT_LOAD:
            self._load_position()
        elif action == ACT_TOGGLE_SETTINGS:
            self.settings.toggle()

    # ── Save / Load ────────────────────────────────────────────────────────

    def _save_position(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.asksaveasfilename(
                parent=root,
                title="Save position",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                initialdir=os.getcwd(),
            )
            root.destroy()
            if path:
                self.state.save_json(path)
                self.status_message = f"Saved to {os.path.basename(path)}"
        except Exception as e:
            self.status_message = f"Save failed: {e}"

    def _load_position(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                parent=root,
                title="Load position",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
                initialdir=os.getcwd(),
            )
            root.destroy()
            if path:
                self.state = PositionState.load_json(path)
                self.results = {}
                self.selected_square = None
                self.status_message = f"Loaded {os.path.basename(path)}"
        except Exception as e:
            self.status_message = f"Load failed: {e}"

    # ── Evaluation ─────────────────────────────────────────────────────────

    def _run_evaluation(self):
        # Ignore if one is already running.
        if self._eval_thread and self._eval_thread.is_alive():
            self.status_message = "Evaluation already in progress"
            return

        ok, errors = self.state.validate()
        if not ok:
            self.status_message = "Validation failed: " + "; ".join(errors[:3])
            return

        model_indices = self.settings.selected_model_indices()
        sim_counts = self.settings.selected_sim_counts()
        if not model_indices:
            self.status_message = "No models selected in Settings"
            return
        if not sim_counts:
            self.status_message = "No sim counts allowed by Settings"
            return

        # Count cells for progress tracking. Copy the game object first — the
        # C++ backend isn't guaranteed thread-safe if the UI thread mutates it
        # during evaluation.
        from copy import deepcopy
        game_snapshot = deepcopy(self.state.get_game())
        cells_total = len(model_indices) * len(sim_counts)

        with self._progress_lock:
            self._progress["active"] = True
            self._progress["cells_done"] = 0
            self._progress["cells_total"] = cells_total
            self._progress["model"] = ""
            self._progress["sim_count"] = 0
            self._progress["sims_done"] = 0
            self._progress["sims_total"] = 0

        def _on_progress(stage, label, sim_count, sims_done, sims_total):
            with self._progress_lock:
                if stage == "start":
                    self._progress["model"] = label
                    self._progress["sim_count"] = sim_count
                    self._progress["sims_done"] = 0
                    self._progress["sims_total"] = sims_total
                elif stage == "progress":
                    self._progress["sims_done"] = sims_done
                    self._progress["sims_total"] = sims_total
                elif stage == "done":
                    self._progress["cells_done"] += 1

        def _worker():
            try:
                self._eval_results = self.model_mgr.evaluate(
                    game_snapshot, model_indices, sim_counts,
                    progress_callback=_on_progress,
                )
                self._eval_error = None
            except Exception as e:
                self._eval_error = str(e)
                self._eval_results = None

        self._eval_thread = threading.Thread(target=_worker, daemon=True)
        self._eval_thread.start()
        self.status_message = "Evaluation started..."

    # ── Controls sync ──────────────────────────────────────────────────────

    def _sync_controls(self):
        self.controls.sync(
            mode_label=("Game Setup"
                        if self.state.mode == Mode.GAME_SETUP
                        else "Construction"),
            current_turn=self.state.get_current_player(),
            show_construction_controls=(self.state.mode == Mode.CONSTRUCTION),
            can_step_back=(self.state.mode == Mode.GAME_SETUP
                           and self.state.current_step() > 0),
            can_step_forward=(self.state.mode == Mode.GAME_SETUP
                              and self.state.current_step()
                              < self.state.move_history_length()),
            settings_open=self.settings.is_open(),
        )

    # ── Draw frame ─────────────────────────────────────────────────────────

    def _draw_frame(self):
        self.screen.fill((248, 248, 250))

        # Board with highlights
        legal_targets = None
        if self.selected_square is not None:
            legal_targets = {
                m.to_pos for m in self.state.get_legal_moves()
                if m.from_pos == self.selected_square
            }
        self.board.draw(
            self.screen, self.state,
            selected=self.selected_square,
            legal_targets=legal_targets,
        )

        # Palette (construction only)
        if self.state.mode == Mode.CONSTRUCTION:
            self.palette.draw(self.screen)

        # Controls row
        self.controls.draw(self.screen)

        # Status message
        if self.status_message:
            text = self.font_status.render(
                self.status_message, True, (30, 30, 30))
            y = STATUS_Y + (self.palette.height + 8
                            if self.state.mode == Mode.CONSTRUCTION else 0)
            self.screen.blit(text, (BOARD_X, y))

        # Turn indicator
        turn_str = (f"To move: "
                    f"{'White' if self.state.get_current_player() == Color.WHITE else 'Black'}"
                    f"   |   move # {self.state.current_step()}"
                    f"/{self.state.move_history_length()}"
                    if self.state.mode == Mode.GAME_SETUP
                    else f"To move: "
                    f"{'White' if self.state.get_current_player() == Color.WHITE else 'Black'}"
                    f"   |   Construction mode")
        text = self.font_notice.render(turn_str, True, (60, 60, 80))
        self.screen.blit(text, (BOARD_X, BOARD_Y - 18))

        # Eval area (columns + scroll arrows)
        self._draw_eval_area()

        # Settings overlay (drawn last so it's on top)
        self.settings.draw(self.screen)

        # Progress overlay on top of everything if evaluation is running
        self._draw_progress_overlay()

        pygame.display.flip()

    def _draw_progress_overlay(self) -> None:
        with self._progress_lock:
            active = self._progress["active"]
            model = self._progress["model"]
            sim_count = self._progress["sim_count"]
            sims_done = self._progress["sims_done"]
            sims_total = self._progress["sims_total"]
            cells_done = self._progress["cells_done"]
            cells_total = self._progress["cells_total"]

        if not active:
            return

        # Dim the whole screen slightly
        dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        self.screen.blit(dim, (0, 0))

        # Panel centered on screen
        pw, ph = 460, 150
        px = (SCREEN_W - pw) // 2
        py = (SCREEN_H - ph) // 2
        pygame.draw.rect(self.screen, (250, 250, 252),
                         pygame.Rect(px, py, pw, ph))
        pygame.draw.rect(self.screen, (60, 60, 80),
                         pygame.Rect(px, py, pw, ph), width=2)

        font_h = pygame.font.SysFont("Arial", 16, bold=True)
        font_r = pygame.font.SysFont("Arial", 13)

        title = font_h.render("Evaluating...", True, (30, 30, 60))
        self.screen.blit(title, (px + 16, py + 12))

        # Cell-level progress (which model & sim setting)
        if sim_count == 1:
            sim_label = "raw NN"
        else:
            sim_label = f"{sim_count} sims"
        text1 = font_r.render(
            f"Model: {model or '-'}   |   Sim count: {sim_label}",
            True, (30, 30, 30))
        self.screen.blit(text1, (px + 16, py + 40))

        # Cell counter
        text2 = font_r.render(
            f"Cell {cells_done + (1 if active else 0)} / {cells_total}",
            True, (30, 30, 30))
        self.screen.blit(text2, (px + 16, py + 62))

        # Sim-level progress bar
        if sims_total > 0:
            frac = sims_done / max(sims_total, 1)
        else:
            frac = 0.0
        bar_x = px + 16
        bar_y = py + 92
        bar_w = pw - 32
        bar_h = 18
        pygame.draw.rect(self.screen, (225, 225, 230),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (60, 120, 200),
                         pygame.Rect(bar_x, bar_y,
                                     int(bar_w * frac), bar_h))
        pygame.draw.rect(self.screen, (100, 100, 120),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h), width=1)
        bar_label = font_r.render(
            f"{sims_done}/{sims_total} sims" if sims_total > 1
            else "raw NN eval",
            True, (30, 30, 30))
        self.screen.blit(bar_label, (bar_x + 4, bar_y + 20))

    def _visible_column_indices(self) -> List[int]:
        return self.settings.selected_model_indices()

    def _draw_eval_area(self):
        # Scroll arrows
        left_rect = pygame.Rect(EVAL_AREA_X, EVAL_AREA_Y - 4, 24, 22)
        right_rect = pygame.Rect(EVAL_AREA_X + 30, EVAL_AREA_Y - 4, 24, 22)
        pygame.draw.rect(self.screen, (230, 230, 235), left_rect)
        pygame.draw.rect(self.screen, (230, 230, 235), right_rect)
        pygame.draw.rect(self.screen, (150, 150, 160), left_rect, width=1)
        pygame.draw.rect(self.screen, (150, 150, 160), right_rect, width=1)
        arrow_font = pygame.font.SysFont("Arial", 14, bold=True)
        for r, arr in ((left_rect, "<"), (right_rect, ">")):
            t = arrow_font.render(arr, True, (30, 30, 30))
            self.screen.blit(t, t.get_rect(center=r.center))

        # Compute how many columns fit horizontally
        columns_start_x = EVAL_AREA_X
        columns_start_y = EVAL_AREA_Y + 24
        col_width = EvalColumn.WIDTH
        # Reserve space if settings panel is open (drawn as overlay)
        avail_width = SCREEN_W - columns_start_x - 12
        if self.settings.is_open():
            avail_width -= self.settings.WIDTH + 12
        cols_per_view = max(1, avail_width // (col_width + COLUMN_GAP))
        visible_h = SCREEN_H - columns_start_y - 12

        model_indices = self._visible_column_indices()
        if not model_indices:
            note = self.font_notice.render(
                "(No models enabled in Settings)",
                True, (120, 120, 120))
            self.screen.blit(note, (columns_start_x + 60, columns_start_y + 20))
            return

        # Clamp scroll
        self.column_scroll = max(0, min(self.column_scroll,
                                        max(0, len(model_indices) - cols_per_view)))
        show = model_indices[
            self.column_scroll: self.column_scroll + cols_per_view]

        sim_counts = self.settings.selected_sim_counts()

        cx = columns_start_x
        for mi in show:
            label = self.model_mgr.get_labels()[mi]
            results = self.results.get(mi, {})
            self.eval_col.draw(
                self.screen, cx, columns_start_y, visible_h,
                label, results, sim_counts,
                vert_scroll=self.vert_scroll,
            )
            cx += col_width + COLUMN_GAP


def main():
    print("[positional_eval] launching model picker...", flush=True)
    paths = pick_models()
    if not paths:
        print("[positional_eval] no models selected — exiting", flush=True)
        return

    print(f"[positional_eval] {len(paths)} model(s) selected, loading...",
          flush=True)
    app = App(paths)
    app.run()


if __name__ == "__main__":
    main()
