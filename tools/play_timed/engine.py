"""Engine: wraps MCTS in a worker thread so pygame stays responsive.

Phase 1 usage:
  eng = Engine(model_path, device="cpu")
  eng.start_move_search(game, sim_budget=200)
  # ... every pygame frame:
  if eng.is_done():
      result = eng.take_result()
      # result: {move, search_snapshot}

Phase 2 will add ponder_start / ponder_stop / promote_and_continue methods
that leverage mcts_search's prev_root / return_root parameters.

Design notes:
  - We measure sims/sec ONCE at startup so the main loop can convert a
    time budget (seconds) into a sim count without re-measuring per move.
  - The worker thread never blocks pygame — main loop polls is_done().
  - Search runs with noise disabled and tactical_shortcuts enabled
    (deterministic, will instantly play a mate-in-1).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402

from extinction_chess import ExtinctionChess  # noqa: E402
from alphazero import (  # noqa: E402
    AlphaZeroNet, AlphaZeroEvaluator, mcts_search, move_to_index,
)


# Placeholder — the user reported ~10 sims/sec measured on their laptop.
# We use this only until we've done our own warmup measurement.
DEFAULT_SIMS_PER_SECOND = 10.0


class Engine:
    """MCTS runner + measured throughput. One Engine per model per session."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)

        # Load the model.
        self.model, meta = AlphaZeroNet.load_checkpoint(model_path, migrate=True)
        self.model = self.model.to(self.device).eval()
        self.iteration = int(meta.get("iteration", -1))
        self.evaluator = AlphaZeroEvaluator(self.model, device=self.device)

        # Measured at first use; None until then.
        self.sims_per_second: Optional[float] = None

        # Worker thread state.
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    # ── Sim/time conversion ────────────────────────────────────────────

    def sims_for_time_budget(self, seconds: float) -> int:
        """Convert a time budget to a sim budget. Rounds DOWN and enforces
        a minimum of 1 so we never call MCTS with 0 sims."""
        rate = self.sims_per_second or DEFAULT_SIMS_PER_SECOND
        return max(1, int(seconds * rate))

    def warmup(self, game: ExtinctionChess, sample_sims: int = 50) -> float:
        """Measure sims/sec on a real position. Blocks; do this at match
        start (before the clock is running). Returns the measured rate.

        50 sims is small enough to be quick but large enough for a stable
        measurement. If the position is terminal / has no legal moves,
        we fall back to the default and log a note.
        """
        if game.game_over or not game.get_legal_moves():
            self.sims_per_second = DEFAULT_SIMS_PER_SECOND
            return self.sims_per_second

        t0 = time.monotonic()
        move_visits, _root_value = mcts_search(
            game, self.evaluator,
            num_simulations=sample_sims,
            dirichlet_alpha=0.0,
            noise_weight=0.0,
            tactical_shortcuts=False,     # measure raw rate, not a shortcut hit
        )
        elapsed = max(1e-6, time.monotonic() - t0)
        self.sims_per_second = sample_sims / elapsed
        return self.sims_per_second

    # ── Worker thread lifecycle ────────────────────────────────────────

    def start_move_search(self, game: ExtinctionChess, sim_budget: int) -> None:
        """Kick off MCTS for the model's move in a background thread.

        The main loop should poll is_done() and take_result() when true.
        Only ONE search at a time; calling again while one is pending
        raises RuntimeError.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Engine already has a search in flight")

        with self._lock:
            self._result = None

        # Snapshot the game because ExtinctionChess isn't safe across threads
        # while being mutated. Deep copy via reconstruction from board state.
        game_snapshot = self._snapshot_game(game)

        def _worker():
            t0 = time.monotonic()
            move_visits, root_value = mcts_search(
                game_snapshot, self.evaluator,
                num_simulations=sim_budget,
                c_puct=2.5,
                dirichlet_alpha=0.0,     # deterministic play
                noise_weight=0.0,
                tactical_shortcuts=True, # instantly play mate-in-1s
            )
            elapsed = time.monotonic() - t0

            if not move_visits:
                # No legal moves — should be caught before start_move_search
                # is called, but handle defensively.
                with self._lock:
                    self._result = {"move": None, "elapsed": elapsed,
                                    "search_snapshot": None}
                return

            # Pick the highest-visit move (argmax — deterministic play).
            best_move, best_visits = max(move_visits, key=lambda x: x[1])
            total_visits = sum(v for _, v in move_visits)

            # Build a compact snapshot for Phase 3 review.
            # Sort by visits desc so top moves are first.
            sorted_moves = sorted(move_visits, key=lambda x: x[1], reverse=True)
            snapshot = {
                "sim_count": total_visits,
                "root_value": float(root_value),
                "top_moves": [
                    {
                        "from": [m.from_pos.rank, m.from_pos.file],
                        "to":   [m.to_pos.rank, m.to_pos.file],
                        "promotion": m.promotion.value if m.promotion else None,
                        "visits": v,
                        "prob": v / total_visits if total_visits else 0.0,
                    }
                    for m, v in sorted_moves
                ],
                "elapsed_seconds": elapsed,
            }

            with self._lock:
                self._result = {
                    "move": best_move,
                    "elapsed": elapsed,
                    "search_snapshot": snapshot,
                }

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def is_done(self) -> bool:
        with self._lock:
            return self._result is not None

    def take_result(self) -> Optional[Dict[str, Any]]:
        """Return the result and clear it. Call after is_done() → True."""
        with self._lock:
            r = self._result
            self._result = None
        self._thread = None
        return r

    # ── Utilities ──────────────────────────────────────────────────────

    def _snapshot_game(self, game: ExtinctionChess) -> ExtinctionChess:
        """Deep-copy the game via _copy_game (in alphazero.py). MCTS mutates
        node.game internally; we don't want that touching the main-thread
        game object."""
        # ExtinctionChess deep copy in the C++ backend is a plain constructor
        # copy — the Python fallback needs a bit more care. Easiest cross-
        # backend: create fresh and replay move history if we tracked it,
        # OR just rely on that our C++ Game copy works cleanly.
        # For safety and portability we do a manual field copy.
        gc = ExtinctionChess()
        gc.board = game.board.copy()
        gc.current_player = game.current_player
        gc.game_over = game.game_over
        # winner attribute exists on Python fallback; may or may not on C++.
        if hasattr(game, "winner"):
            gc.winner = game.winner
        return gc
