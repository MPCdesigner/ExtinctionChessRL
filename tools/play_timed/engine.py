"""Engine: MCTS runner with continuous pondering (Phase 2).

Key idea: the engine maintains a persistent MCTS root that grows in a
background thread. When either side plays a move, the engine descends
into that move's subtree via `descend_root` (from alphazero.py, shipped
with tree reuse Aug 3). No search work is thrown away between turns.

State machine (managed by the worker thread):
    IDLE       → start_from(game)   → SEARCHING
    SEARCHING  → descend(move)      → SEARCHING (new root = subtree)
    SEARCHING  → stop()             → IDLE

Main loop interaction:
    engine.start_from(game)              # at game start / new game
    # ... during user's turn: engine keeps pondering ...
    engine.descend(user_move)            # user makes move
    # ... engine keeps pondering from new root ...
    time.sleep(model_time_budget_sec)    # let engine work during model's turn
    move, snap = engine.get_current_result()   # main takes the current best
    engine.descend(model_move)           # model plays; keep pondering
    # ... etc ...

Warmup: measures sims/sec on the STARTING position via a bounded
mcts_search before pondering begins. Phase 1 tests showed ~10 sims/sec
on CPU for this model, so time budgets are mapped via that rate.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402

from extinction_chess import ExtinctionChess, Move  # noqa: E402
from alphazero import (  # noqa: E402
    AlphaZeroNet, AlphaZeroEvaluator, mcts_search,
)


def _descend_root_structural(root, played_move):
    """Local replacement for alphazero.descend_root that compares moves by
    (from, to, promo) tuple instead of Python identity.

    Bug: alphazero.descend_root uses `child.move == played_move` which falls
    back to identity comparison (Move class doesn't implement __eq__). That
    works for self-play because both sides pull the same Move object from
    the MCTS tree. It BREAKS for UI-driven descent, where the user's Move
    object comes from a fresh get_legal_moves() call and doesn't
    identity-match any Move in the tree — so descend_root ALWAYS returns
    None and the tree gets discarded on every move.

    This local version matches by field values, which is what we want for
    Phase 2 pondering to actually accumulate across turns.
    """
    if root is None:
        return None
    for child in root.children:
        cm = child.move
        pm = played_move
        if (cm.from_pos.rank == pm.from_pos.rank
                and cm.from_pos.file == pm.from_pos.file
                and cm.to_pos.rank == pm.to_pos.rank
                and cm.to_pos.file == pm.to_pos.file
                and cm.promotion == pm.promotion):
            if not child.is_expanded or child.visit_count == 0:
                return None
            return child
    return None


DEFAULT_SIMS_PER_SECOND = 10.0
PONDER_CHUNK_SIMS = 30   # how many extra sims to target per mcts_search chunk


class Engine:
    """Persistent MCTS worker with a growing root tree.

    Thread safety:
        - The worker thread OWNS `self._root` and `self._root_game`.
        - The main thread communicates via a lock-guarded request queue.
        - `get_current_result()` reads a lock-guarded snapshot; safe to call
          from main at any time.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = torch.device(device)

        self.model, meta = AlphaZeroNet.load_checkpoint(model_path, migrate=True)
        self.model = self.model.to(self.device).eval()
        self.iteration = int(meta.get("iteration", -1))
        self.evaluator = AlphaZeroEvaluator(self.model, device=self.device)

        self.sims_per_second: Optional[float] = None

        # ── Worker thread state ────────────────────────────────────────
        # Held only while modifying request state.
        self._req_lock = threading.Lock()
        # Pending requests: list of ("start", game) or ("descend", move)
        # or ("stop",). Consumed by the worker between MCTS chunks.
        self._pending_requests: List[Tuple[str, Any]] = []
        # Set to signal worker to exit entirely (on tool shutdown).
        self._exit_flag = threading.Event()
        # Set to signal the CURRENT mcts_search chunk to stop early
        # (used when a descend request arrives mid-chunk).
        self._interrupt_chunk = threading.Event()

        # Held only while modifying result state.
        self._result_lock = threading.Lock()
        # Latest visible-to-main snapshot of the current root.
        # Only meaningful when SEARCHING (not IDLE).
        self._latest_visits: List[Tuple[Move, int]] = []
        self._latest_root_value: float = 0.0
        self._latest_root_sim_count: int = 0

        # Worker's own state (only worker thread reads/writes these).
        self._w_state: str = "IDLE"
        self._w_game: Optional[ExtinctionChess] = None
        self._w_root = None  # MCTSNode from alphazero — Python-side type

        # Start the worker thread.
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # ── Sim/time conversion ────────────────────────────────────────────

    def sims_for_time_budget(self, seconds: float) -> int:
        rate = self.sims_per_second or DEFAULT_SIMS_PER_SECOND
        return max(1, int(seconds * rate))

    def warmup(self, game: ExtinctionChess, sample_sims: int = 30) -> float:
        """Blocking measurement of sims/sec on a live position. Call once
        before start_from() so the clock isn't running yet."""
        if game.game_over or not game.get_legal_moves():
            self.sims_per_second = DEFAULT_SIMS_PER_SECOND
            return self.sims_per_second

        game_copy = self._snapshot_game(game)
        t0 = time.monotonic()
        mcts_search(
            game_copy, self.evaluator,
            num_simulations=sample_sims,
            dirichlet_alpha=0.0,
            noise_weight=0.0,
            tactical_shortcuts=False,
        )
        elapsed = max(1e-6, time.monotonic() - t0)
        self.sims_per_second = sample_sims / elapsed
        return self.sims_per_second

    # ── Requests from main (thread-safe) ───────────────────────────────

    def start_from(self, game: ExtinctionChess) -> None:
        """Begin pondering from `game`. Discards any prior root."""
        with self._req_lock:
            self._pending_requests.append(("start", self._snapshot_game(game)))
        self._interrupt_chunk.set()

    def descend(self, played_move: Move) -> None:
        """A move was played — descend the root and keep pondering.

        If the current root doesn't have a subtree for this move (rare —
        e.g., we're mid-warmup or the tree hasn't expanded far enough),
        the worker falls back to a fresh search from the new position.
        """
        with self._req_lock:
            self._pending_requests.append(("descend", played_move))
        self._interrupt_chunk.set()

    def stop(self) -> None:
        """Stop searching. Engine transitions to IDLE."""
        with self._req_lock:
            self._pending_requests.append(("stop", None))
        self._interrupt_chunk.set()

    def shutdown(self) -> None:
        """Terminate the worker thread (on tool exit)."""
        self._exit_flag.set()
        self._interrupt_chunk.set()
        self._thread.join(timeout=2.0)

    # ── Reading current result (thread-safe) ───────────────────────────

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Cheap read of the engine's live state — for UI display.

        Returns: {sim_count, top_move_visits_str, state}.
        `state` is what the worker thread is currently doing:
          "SEARCHING" — MCTS chunks running against a persistent root
          "IDLE"     — waiting for a start_from() request
        (This is a lock-guarded read of the worker's internal state field;
        safe to call every frame from main.)
        """
        with self._result_lock:
            sim_count = self._latest_root_sim_count
            top_visits = None
            if self._latest_visits:
                _, top_visits = max(self._latest_visits, key=lambda x: x[1])
        state = self._w_state
        return {
            "sim_count": sim_count,
            "top_visits": top_visits,
            "state": state,
        }

    def get_current_result(self) -> Optional[Dict[str, Any]]:
        """Snapshot of the current root's search state. Returns None if
        the engine is IDLE (nothing to report yet). Otherwise:
            {
              "move":         Move,           # highest-visit child of root
              "search_snapshot": {
                "sim_count":  int,            # visits at root
                "root_value": float,          # from-current-player perspective
                "top_moves":  [{from, to, promotion, visits, prob}, ...],
                "elapsed_seconds": 0.0,       # unused in ponder mode
              },
            }
        Returns None if no legal move exists at the current root (game over).
        """
        with self._result_lock:
            visits = list(self._latest_visits)
            root_value = self._latest_root_value
            total = self._latest_root_sim_count

        if not visits:
            return None

        best_move, _best_visits = max(visits, key=lambda x: x[1])
        sorted_moves = sorted(visits, key=lambda x: x[1], reverse=True)
        snap = {
            "sim_count": total,
            "root_value": float(root_value),
            "top_moves": [
                {
                    "from": [m.from_pos.rank, m.from_pos.file],
                    "to":   [m.to_pos.rank, m.to_pos.file],
                    "promotion": m.promotion.value if m.promotion else None,
                    "visits": v,
                    "prob": v / max(1, total),
                }
                for m, v in sorted_moves
            ],
            "elapsed_seconds": 0.0,
        }
        return {"move": best_move, "search_snapshot": snap}

    # ── Worker thread body ─────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Continuously run MCTS chunks against the current root, servicing
        requests between chunks. All root/game mutations happen here."""
        while not self._exit_flag.is_set():
            # Drain any pending requests.
            self._process_pending_requests()
            if self._exit_flag.is_set():
                break

            if self._w_state != "SEARCHING" or self._w_game is None:
                time.sleep(0.03)  # idle sleep
                continue

            # If game is over at the current position, nothing to search.
            if self._w_game.game_over or not self._w_game.get_legal_moves():
                self._w_state = "IDLE"
                self._w_root = None
                self._w_game = None
                self._publish_result([], 0.0, 0)
                continue

            # Do one chunk of MCTS.
            current_sims = (self._w_root.visit_count
                            if self._w_root is not None else 0)
            target = current_sims + PONDER_CHUNK_SIMS

            self._interrupt_chunk.clear()
            try:
                move_visits, root_value, new_root = mcts_search(
                    self._w_game, self.evaluator,
                    num_simulations=target,
                    c_puct=2.5,
                    dirichlet_alpha=0.0,
                    noise_weight=0.0,
                    tactical_shortcuts=True,
                    prev_root=self._w_root,
                    return_root=True,
                    should_stop=self._interrupt_chunk.is_set,
                )
            except Exception as e:
                # Log and drop back to IDLE. Main will notice via
                # get_current_result() returning None (empty visits).
                print(f"[engine] mcts_search raised: "
                      f"{type(e).__name__}: {e}", flush=True)
                self._w_state = "IDLE"
                self._w_root = None
                self._w_game = None
                self._publish_result([], 0.0, 0)
                continue

            self._w_root = new_root
            self._publish_result(move_visits, root_value, new_root.visit_count)

    def _process_pending_requests(self) -> None:
        """Consume all pending requests. Called between chunks."""
        with self._req_lock:
            reqs = self._pending_requests
            self._pending_requests = []

        for kind, payload in reqs:
            if kind == "start":
                self._w_game = payload         # already-copied game
                self._w_root = None            # fresh root
                self._w_state = "SEARCHING"
                self._publish_result([], 0.0, 0)   # reset visible state
            elif kind == "descend":
                move = payload
                if self._w_game is None or self._w_root is None:
                    # Nothing to descend into — apply move to game and search fresh.
                    if self._w_game is not None:
                        applied = self._w_game.make_move(move)
                        if not applied:
                            # Illegal from-engine-perspective; nothing we can do.
                            self._w_state = "IDLE"
                            self._w_root = None
                            self._w_game = None
                            self._publish_result([], 0.0, 0)
                    continue

                # First apply the move to the game snapshot.
                new_game_copy = self._snapshot_game(self._w_game)
                applied = new_game_copy.make_move(move)
                if not applied:
                    print(f"[engine] descend: make_move failed for "
                          f"{move}", flush=True)
                    self._w_state = "IDLE"
                    self._w_root = None
                    self._w_game = None
                    self._publish_result([], 0.0, 0)
                    continue

                # Then descend the tree to the corresponding subtree.
                # Uses our structural comparison — NOT alphazero.descend_root
                # (which would fail on identity comparison since UI moves
                # are fresh Move instances, not the ones in the tree).
                promoted = _descend_root_structural(self._w_root, move)
                self._w_game = new_game_copy
                self._w_root = promoted   # may be None → fresh search next chunk
                self._w_state = "SEARCHING"
                # Reset visible state so main doesn't see stale visits.
                self._publish_result([], 0.0, 0)
            elif kind == "stop":
                self._w_state = "IDLE"
                self._w_root = None
                self._w_game = None
                self._publish_result([], 0.0, 0)

    def _publish_result(self, move_visits, root_value: float,
                        sim_count: int) -> None:
        """Copy the latest snapshot to a lock-guarded field that main
        threads can read via get_current_result()."""
        with self._result_lock:
            self._latest_visits = list(move_visits)
            self._latest_root_value = float(root_value)
            self._latest_root_sim_count = int(sim_count)

    # ── Game copy (safe from cross-thread mutation) ────────────────────

    def _snapshot_game(self, game: ExtinctionChess) -> ExtinctionChess:
        gc = ExtinctionChess()
        gc.board = game.board.copy()
        gc.current_player = game.current_player
        gc.game_over = game.game_over
        if hasattr(game, "winner"):
            gc.winner = game.winner
        return gc
