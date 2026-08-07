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
    ACT_TOGGLE_VALUE_GEN,
    ACT_SAVE_VALUE_WHITE, ACT_SAVE_VALUE_DRAW, ACT_SAVE_VALUE_BLACK,
)
from .settings_panel import SettingsPanel  # noqa: E402
from .eval_column import EvalColumn  # noqa: E402
from .position_state import PositionState, Mode  # noqa: E402
from .model_manager import ModelManager, EvalResult, SIM_UNLIMITED  # noqa: E402
from .startup_dialog import pick_models  # noqa: E402
from .value_dataset import ValueDataset, default_dataset_path  # noqa: E402


# ── Screen layout ────────────────────────────────────────────────────────────
SCREEN_W = 1500
SCREEN_H = 940   # +40 vs pre-value-drilling: second controls row is always
                  # reserved so BOARD_Y is constant regardless of value-gen state
CONTROLS_HEIGHT = 80   # 2 * ControlsWidget.ROW_HEIGHT (see controls_widget.py)
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
        # Position-keyed cache of prior evaluations. When the user steps back
        # / forward / navigates to a position they previously evaluated, we
        # restore results from here instead of showing empty cells.
        # Key: opaque tuple returned by _position_key().
        self._eval_cache: Dict[tuple, Dict[int, Dict[int, EvalResult]]] = {}
        # Horizontal scroll (column offset) for the eval area
        self.column_scroll = 0
        # Vertical scroll offset (in pixels) applied to each visible column
        self.vert_scroll = 0
        # Transient status message
        self.status_message: str = ""

        # Value-drilling dataset. Loaded eagerly (cheap — small JSON file);
        # a session is not started until the user actually toggles on
        # generation mode, so pure evaluation workflows leave zero trace.
        self.dataset = ValueDataset(default_dataset_path())
        # Count of entries added in the current tool run — used for the
        # toggle button's compact status label ("K this session").
        self._session_entry_count = 0

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
        # Stop flag: set from UI when Stop button pressed; consulted by
        # mcts_search's should_stop hook. Cleared before each new model.
        self._stop_current_model = threading.Event()
        # Rect of the stop button in the progress overlay — populated each
        # frame when overlay is drawn; used to detect clicks.
        self._stop_button_rect: Optional[pygame.Rect] = None
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
                    eval_running = (self._eval_thread
                                    and self._eval_thread.is_alive())
                    # Scroll wheel always works — user needs to browse cells
                    # while eval is running.
                    if event.button in (4, 5):
                        direction = -1 if event.button == 4 else 1
                        self._handle_scroll(event.pos, direction)
                        continue
                    if eval_running:
                        # During eval, only Stop button and column arrows
                        # accept clicks. Everything else could mutate state.
                        if event.button == 1:
                            if (self._stop_button_rect and
                                    self._stop_button_rect.collidepoint(event.pos)):
                                self._stop_current_model.set()
                                self.status_message = ("Stopping current "
                                                       "model...")
                            else:
                                self._handle_scroll_arrows(*event.pos)
                        continue
                    if event.button == 1:
                        self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    # Arrow keys only affect display scrolling — safe during
                    # eval since they don't mutate position state.
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

    # ── Evaluation cache ───────────────────────────────────────────────────

    def _position_key(self) -> tuple:
        """Hashable identity of the current position for eval caching.

        In game_setup mode the position is fully determined by the move
        history up to step_position. In construction mode we hash the
        board grid + current player + halfmove/fullmove counters + en
        passant target. Two branches that reach the same board state may
        still get different keys if the move sequence differs; that's the
        correct behavior since the model's history planes depend on the
        move sequence.
        """
        if self.state.mode == Mode.GAME_SETUP:
            moves = self.state.move_history[:self.state.step_position]
            return ("game_setup",
                    self.state.step_position,
                    tuple(str(m) for m in moves))
        # Construction mode: canonical board hash
        game = self.state.get_game()
        pieces = []
        for r in range(8):
            for f in range(8):
                p = game.board.grid[r][f]
                if p is not None:
                    pieces.append((
                        r, f,
                        getattr(p.piece_type, "value", p.piece_type),
                        getattr(p.color, "value", p.color),
                        bool(p.has_moved),
                    ))
        ep = game.board.en_passant_target
        return ("construction",
                tuple(pieces),
                getattr(game.current_player, "value", game.current_player),
                game.board.halfmove_clock,
                game.board.fullmove_number,
                (ep.rank, ep.file) if ep is not None else None,
                )

    def _restore_or_clear_results(self) -> None:
        """Called after any state change that could invalidate results.

        Restores previously cached results for this position if available,
        otherwise clears self.results. Resets scroll to top of the (new)
        cell layout.
        """
        cached = self._eval_cache.get(self._position_key())
        self.results = cached if cached is not None else {}
        self.column_scroll = 0
        self.vert_scroll = 0

    def _cache_current_results(self) -> None:
        """Save self.results to the cache under the current position key."""
        if self.results:
            self._eval_cache[self._position_key()] = self.results

    def _poll_eval_thread(self) -> None:
        """Called each frame from main. Reap results if the worker finished."""
        if self._eval_thread is None:
            return
        if self._eval_thread.is_alive():
            return
        # Worker finished — pull results / error and clear. Because live
        # updates were streaming into self.results during the run, the final
        # eval_results dict may be an extension; take it if present so we
        # capture any late-fired checkpoint results.
        self._eval_thread = None
        if self._eval_error is not None:
            self.status_message = f"Evaluation error: {self._eval_error}"
            self._eval_error = None
        elif self._eval_results is not None:
            self.results = self._eval_results
            self._eval_results = None
            self.status_message = "Evaluation complete."
            # Persist to cache so we can restore on step-back/-forward.
            self._cache_current_results()
        with self._progress_lock:
            self._progress["active"] = False
        self._stop_current_model.clear()
        self._stop_button_rect = None

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
            self._restore_or_clear_results()
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
                self._restore_or_clear_results()

    # ── Board display helpers ──────────────────────────────────────────────

    def _compute_endangered_squares(self) -> set:
        """Squares whose occupying piece is the last of its type for its color.

        Returns a set of Position objects. Extinction chess loses the game
        when a piece TYPE goes extinct, so "endangered" means "type count == 1"
        (game.get_endangered_pieces returns the list of endangered types per
        color; we then locate the squares those pieces occupy).
        """
        game = self.state.get_game()
        endangered = set()
        for color in (Color.WHITE, Color.BLACK):
            types = set(game.get_endangered_pieces(color))
            if not types:
                continue
            for rank in range(8):
                for file in range(8):
                    piece = self.state.get_piece_at(Position(rank, file))
                    if piece is None:
                        continue
                    if piece.color == color and piece.piece_type in types:
                        endangered.add(Position(rank, file))
        return endangered

    # ── Value-dataset UI helpers ───────────────────────────────────────────

    def _value_match_text(self) -> str:
        """One-line summary of dataset matches for the current position.

        Returns "" if the current position has never been saved. Otherwise
        returns e.g. "In dataset: 2x +1, 1x -1  [forced_win, material_adv]".
        The trailing active-tags section shows what tags WOULD be applied on
        the next save (they persist across saves until user toggles them).
        """
        breakdown = self.dataset.get_value_breakdown(self.state.to_dict())
        active_tags = self.controls.get_active_tags()
        parts = []
        if breakdown.get(1, 0):
            parts.append(f"{breakdown[1]}x +1")
        if breakdown.get(0, 0):
            parts.append(f"{breakdown[0]}x 0")
        if breakdown.get(-1, 0):
            parts.append(f"{breakdown[-1]}x -1")

        # Always show active tags (even with no dataset match) so the user
        # can see what's set BEFORE saving anything.
        tag_part = f"  [{', '.join(active_tags)}]" if active_tags else ""

        if not parts:
            # No dataset match — only render if there are active tags to show.
            return f"Not in dataset{tag_part}" if tag_part else ""
        return "In dataset: " + ", ".join(parts) + tag_part

    # ── Button dispatch ────────────────────────────────────────────────────

    def _dispatch(self, action: str) -> None:
        if action == ACT_EVALUATE:
            self._run_evaluation()
        elif action == ACT_TOGGLE_MODE:
            new_mode = (Mode.CONSTRUCTION if self.state.mode == Mode.GAME_SETUP
                        else Mode.GAME_SETUP)
            self.state = PositionState(new_mode)
            self._restore_or_clear_results()
            self.selected_square = None
            self.status_message = f"Switched to {new_mode.value} mode"
        elif action == ACT_SET_TURN_W:
            self.state.set_current_player(Color.WHITE)
            self._restore_or_clear_results()
        elif action == ACT_SET_TURN_B:
            self.state.set_current_player(Color.BLACK)
            self._restore_or_clear_results()
        elif action == ACT_STEP_BACK:
            if self.state.step_back():
                self._restore_or_clear_results()
                self.selected_square = None
        elif action == ACT_STEP_FWD:
            if self.state.step_forward():
                self._restore_or_clear_results()
                self.selected_square = None
        elif action == ACT_SAVE:
            self._save_position()
        elif action == ACT_LOAD:
            self._load_position()
        elif action == ACT_TOGGLE_SETTINGS:
            self.settings.toggle()
        elif action == ACT_TOGGLE_VALUE_GEN:
            # controls_widget already flipped its internal enabled flag; we
            # just react. First time ON in this run: lazily start a session
            # and print a summary of the existing dataset so the user knows
            # what they're building on top of.
            if self.controls.value_gen_enabled:
                if self.dataset.current_session_id() is None:
                    self.dataset.start_session()
                sessions = self.dataset.session_summary()
                self.status_message = (
                    f"Value dataset: {self.dataset.total_count()} positions "
                    f"across {len(sessions)} sessions | "
                    f"current session: {self.dataset.current_session_id()}"
                )
            else:
                self.status_message = "Value dataset generation OFF"
        elif action == ACT_SAVE_VALUE_WHITE:
            self._save_value_entry(1)
        elif action == ACT_SAVE_VALUE_DRAW:
            self._save_value_entry(0)
        elif action == ACT_SAVE_VALUE_BLACK:
            self._save_value_entry(-1)

    # ── Value-drilling save flow ───────────────────────────────────────────

    def _save_value_entry(self, value: int) -> None:
        """Add the current position to the value-drilling dataset.

        If the position (semantic key — see value_dataset._canonical_position_key)
        already exists in the dataset, show a modal confirm dialog with the
        per-value breakdown before appending. Duplicates ARE allowed by design;
        the warning is there so the user doesn't do it by accident.
        """
        # Only Game Setup positions carry move_history; Construction-mode
        # positions have zero NN history planes at export time (out of
        # distribution). Warn but don't refuse — user's choice.
        pos_dict = self.state.to_dict()

        # Existing entries for this position
        breakdown = self.dataset.get_value_breakdown(pos_dict)
        total_existing = sum(breakdown.values())
        proceed = True
        if total_existing > 0:
            proceed = self._confirm_duplicate_save(value, breakdown)

        if not proceed:
            self.status_message = "Value save cancelled."
            return

        tags = self.controls.get_active_tags()
        # Extra safety: warn on Construction-mode saves (they'll have zero
        # history planes at export — out of distribution for the NN).
        if self.state.mode == Mode.CONSTRUCTION:
            self.status_message = (
                f"Saved value={value:+d} (Construction mode — zero history "
                f"planes at export; use Game Setup for real games)"
            )
        else:
            self.status_message = (
                f"Saved value={value:+d}"
                + (f" tags={','.join(tags)}" if tags else "")
                + f" | session count: {self._session_entry_count + 1}"
            )
        self.dataset.add_entry(pos_dict, value, tags)
        self._session_entry_count += 1

    def _confirm_duplicate_save(self, new_value: int,
                                breakdown: Dict[int, int]) -> bool:
        """Modal confirm dialog for duplicate-save. Returns True to proceed."""
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            msg = (
                f"This position was already saved:\n"
                f"  {breakdown.get(1, 0)} times with value +1\n"
                f"  {breakdown.get(0, 0)} times with value 0\n"
                f"  {breakdown.get(-1, 0)} times with value -1\n\n"
                f"Save again with value {new_value:+d}?"
            )
            result = messagebox.askyesno(
                "Duplicate position", msg, parent=root)
            root.destroy()
            return bool(result)
        except Exception as e:
            # If the dialog can't open, err on the side of not saving.
            self.status_message = f"Confirm dialog failed: {e}. Not saved."
            return False

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
                self._restore_or_clear_results()
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

        # Clear stop flag from any previous run and reset live results.
        self._stop_current_model.clear()
        # Show live-updating results as they stream in. Start empty.
        self.results = {}
        self.column_scroll = 0
        self.vert_scroll = 0

        def _on_progress(stage, label, sim_count, sims_done, sims_total):
            with self._progress_lock:
                if stage == "start":
                    self._progress["model"] = label
                    self._progress["sim_count"] = sim_count
                    self._progress["sims_done"] = 0
                    self._progress["sims_total"] = sims_total
                    # Starting a new model — clear any pending stop signal
                    # so it only applies to the model it was pressed for.
                    self._stop_current_model.clear()
                elif stage == "progress":
                    self._progress["sims_done"] = sims_done
                    self._progress["sims_total"] = sims_total
                elif stage == "done":
                    self._progress["cells_done"] += 1

        def _on_live(model_index, cell_sim_count, partial_result):
            # Update the visible results dict incrementally. Safe because
            # only this worker thread writes to self.results while eval is
            # active, and the main thread only reads it for drawing.
            per_model = self.results.setdefault(model_index, {})
            per_model[cell_sim_count] = partial_result

        def _should_stop():
            return self._stop_current_model.is_set()

        def _worker():
            try:
                self._eval_results = self.model_mgr.evaluate(
                    game_snapshot, model_indices, sim_counts,
                    progress_callback=_on_progress,
                    live_callback=_on_live,
                    should_stop_current=_should_stop,
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
            dataset_total=self.dataset.total_count(),
            dataset_session_count=self._session_entry_count,
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

        # Compute endangered squares (extinction-chess: a piece is endangered
        # if it's the last of its type for its color). Only if the setting
        # is on — cheap check dodges the 64-square scan otherwise.
        endangered_squares = None
        if self.settings.highlight_endangered_enabled():
            endangered_squares = self._compute_endangered_squares()

        self.board.draw(
            self.screen, self.state,
            selected=self.selected_square,
            legal_targets=legal_targets,
            endangered_squares=endangered_squares,
        )

        # Palette (construction only)
        if self.state.mode == Mode.CONSTRUCTION:
            self.palette.draw(self.screen)

        # Controls row
        self.controls.draw(self.screen)

        # Status message
        status_y_base = STATUS_Y + (self.palette.height + 8
                                    if self.state.mode == Mode.CONSTRUCTION
                                    else 0)
        if self.status_message:
            text = self.font_status.render(
                self.status_message, True, (30, 30, 30))
            self.screen.blit(text, (BOARD_X, status_y_base))

        # Position-match indicator — only when value generation is ON. Shows
        # whether the current position is already in the dataset (and if so,
        # the per-value breakdown + active tags). Renders one line below the
        # status message so it doesn't clobber the "Evaluation complete." /
        # "Save cancelled." transient messages.
        if self.controls.value_gen_enabled:
            match_text = self._value_match_text()
            if match_text:
                indicator_y = status_y_base + 18
                text = self.font_status.render(match_text, True, (80, 60, 20))
                self.screen.blit(text, (BOARD_X, indicator_y))

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

        # Panel at bottom-left (below the board). No screen dim — user needs
        # to see and scroll cells while eval is running.
        pw, ph = 460, 155
        px = BOARD_X
        py = SCREEN_H - ph - 12
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
        elif sim_count >= SIM_UNLIMITED:
            sim_label = "unlimited"
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

        # Sim-level progress bar. In unlimited mode we can't show a
        # fraction, so display an animated indeterminate stripe instead.
        bar_x = px + 16
        bar_y = py + 92
        bar_w = pw - 32
        bar_h = 18
        is_unlimited = sims_total >= SIM_UNLIMITED
        pygame.draw.rect(self.screen, (225, 225, 230),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        if is_unlimited:
            # Moving stripe: a small filled band that scrolls across the bar
            # once every ~1.6s. Purely visual — no completion signal.
            stripe_w = bar_w // 5
            t = (pygame.time.get_ticks() % 1600) / 1600.0
            sx = bar_x + int((bar_w + stripe_w) * t) - stripe_w
            visible = pygame.Rect(
                max(bar_x, sx), bar_y,
                min(stripe_w, bar_x + bar_w - max(bar_x, sx)),
                bar_h,
            )
            if visible.width > 0:
                pygame.draw.rect(self.screen, (60, 120, 200), visible)
        else:
            if sims_total > 0:
                frac = sims_done / max(sims_total, 1)
            else:
                frac = 0.0
            pygame.draw.rect(self.screen, (60, 120, 200),
                             pygame.Rect(bar_x, bar_y,
                                         int(bar_w * frac), bar_h))
        pygame.draw.rect(self.screen, (100, 100, 120),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h), width=1)
        if is_unlimited:
            bar_label_text = f"{sims_done} sims (unlimited)"
        elif sims_total > 1:
            bar_label_text = f"{sims_done}/{sims_total} sims"
        else:
            bar_label_text = "raw NN eval"
        bar_label = font_r.render(bar_label_text, True, (30, 30, 30))
        self.screen.blit(bar_label, (bar_x + 4, bar_y + 20))

        # Stop button (right side of the panel, below the bar). Clicking it
        # stops the CURRENT model's MCTS and moves on to the next model.
        stopping = self._stop_current_model.is_set()
        btn_w, btn_h = 90, 24
        btn_x = px + pw - btn_w - 16
        btn_y = py + ph - btn_h - 12
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self._stop_button_rect = btn_rect
        btn_bg = (200, 60, 60) if not stopping else (150, 100, 100)
        btn_fg = (255, 255, 255)
        pygame.draw.rect(self.screen, btn_bg, btn_rect, border_radius=4)
        pygame.draw.rect(self.screen, (80, 20, 20), btn_rect,
                         width=1, border_radius=4)
        btn_label = font_r.render(
            "Stopping..." if stopping else "Stop", True, btn_fg)
        self.screen.blit(btn_label, btn_label.get_rect(center=btn_rect.center))

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
