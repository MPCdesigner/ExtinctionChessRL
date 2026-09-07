"""Match state: game position, clocks, move history.

Owns everything needed to track an in-progress timed game and enough of
its history that Phase 3 can reconstruct + review it position-by-position.

Clocks tick in real time via monotonic clock. Only ONE clock runs at a
time (the side to move). Elapsed since the clock's last start is
subtracted lazily on read — no threading required.

Phase 1 does NOT run MCTS during the opponent's turn (no pondering).
Phase 2 will add that; the state shape here is designed to accommodate
it (see `search_history` field).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from extinction_chess import Color, ExtinctionChess, Move  # noqa: E402


# ── Time controls ──────────────────────────────────────────────────────────

@dataclass
class TimeControl:
    """Time control for one side.

    base_seconds: initial main-clock time budget.
    increment_seconds: added to the main clock at the END of each move.
      (Standard "Fischer" increment — same rule Lichess uses.)
    delay_seconds: "simple delay" grace period at the START of each turn.
      A delay-only timer counts down first. If the player moves within
      delay_seconds, no main-clock time is consumed. Only after the
      delay expires does the main clock start ticking. Unlike increment,
      unused delay does NOT bank onto the main clock — it just protects
      each move up to delay_seconds. Set delay_seconds=0 to disable.
    """
    base_seconds: float
    increment_seconds: float
    delay_seconds: float = 0.0

    def label(self) -> str:
        """Human-readable label like '5+3' or '5+3(d5)' (with delay)."""
        mins = int(self.base_seconds // 60)
        secs = int(self.base_seconds - mins * 60)
        base_str = f"{mins}" if secs == 0 else f"{mins}:{secs:02d}"
        s = f"{base_str}+{int(self.increment_seconds)}"
        if self.delay_seconds > 0:
            s += f"(d{int(self.delay_seconds)})"
        return s


# ── Outcome enum ───────────────────────────────────────────────────────────

# String constants (simpler than Enum for JSON serialization if we ever want it).
OUTCOME_ONGOING          = "ongoing"
OUTCOME_YOU_WIN_EXTINCT  = "you_win_extinction"
OUTCOME_MODEL_WIN_EXTINCT = "model_win_extinction"
OUTCOME_YOU_FLAGGED       = "you_flagged"
OUTCOME_MODEL_FLAGGED     = "model_flagged"
OUTCOME_DRAW              = "draw"


# ── Per-move record ────────────────────────────────────────────────────────

@dataclass
class MoveRecord:
    """One move by one side. Enough info for Phase 3 review to reconstruct
    the position AFTER this move plus the search context that produced it
    (for the model's moves only)."""
    ply: int                                    # 0-indexed half-move
    side: Color                                 # WHITE or BLACK
    move: Move                                  # the move played
    clock_before_seconds: float                 # mover's remaining time BEFORE
    clock_after_seconds: float                  # mover's remaining time AFTER (post-increment)
    thinking_time_seconds: float                # actual wall time spent on this move
    # Filled in only for model moves (None for user moves).
    # Phase 3 will use this to reveal what the model was thinking.
    search_snapshot: Optional[dict] = None      # {sim_count, top_moves: [(move, visits, prob)], root_value}


# ── Match state ────────────────────────────────────────────────────────────

class MatchState:
    """Holds a single game's live state: position, clocks, history.

    Colors:
        user_color: WHITE or BLACK — which side the human plays.
        model_color: the other one.

    Clocks:
        Two independent clocks. Whichever side's turn it is has their clock
        RUNNING (clock_start_monotonic is set); the other side's clock is
        paused. When a move is played:
          - Stop the mover's clock (subtract elapsed since start).
          - Add increment to the mover's clock.
          - Start the OTHER side's clock (record now as clock_start_monotonic).

    History:
        Every completed move goes into `moves`. Positions are reconstructable
        by replaying (or we can save them, but for extinction chess the
        replay is fast; we avoid duplicating state).
    """

    def __init__(self, model_path: str, user_color: Color,
                 user_tc: TimeControl, model_tc: TimeControl):
        self.model_path = model_path
        self.user_color = user_color
        self.model_color = Color.BLACK if user_color == Color.WHITE else Color.WHITE
        self.user_tc = user_tc
        self.model_tc = model_tc

        self.game = ExtinctionChess()
        self.user_remaining_seconds = user_tc.base_seconds
        self.model_remaining_seconds = model_tc.base_seconds

        # Whichever side is WHITE moves first; that side's clock starts on
        # game start (call start()).
        self.moves: List[MoveRecord] = []
        self.outcome: str = OUTCOME_ONGOING
        self.outcome_detail: str = ""             # e.g., "White Queen extinct"

        # Monotonic timestamp of when the currently-ticking clock last started.
        # None means no clock is ticking (game not started or game ended).
        self._active_clock_start: Optional[float] = None

        # Ready to receive start() call.

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the game — start the clock of whoever moves first."""
        self._active_clock_start = time.monotonic()

    def is_ongoing(self) -> bool:
        return self.outcome == OUTCOME_ONGOING

    def is_user_turn(self) -> bool:
        return self.is_ongoing() and self.game.current_player == self.user_color

    def is_model_turn(self) -> bool:
        return self.is_ongoing() and self.game.current_player == self.model_color

    # ── Clock accessors (lazy elapsed subtraction) ──────────────────────

    def _elapsed_active(self) -> float:
        """Wall seconds since the current turn started (both delay + main)."""
        if self._active_clock_start is None:
            return 0.0
        return time.monotonic() - self._active_clock_start

    def _main_time_consumed(self, elapsed: float, delay: float) -> float:
        """Given total elapsed this turn and the side's delay budget,
        how much time has been drawn from the MAIN clock so far?
        Zero while inside the delay window; positive after it expires."""
        return max(0.0, elapsed - delay)

    def user_clock_display(self) -> float:
        """Seconds remaining on user's MAIN clock RIGHT NOW (includes tick).

        During the delay window at the start of the user's turn, this
        stays constant — main clock hasn't started ticking yet.
        """
        if self.is_user_turn():
            used = self._main_time_consumed(self._elapsed_active(),
                                            self.user_tc.delay_seconds)
            return max(0.0, self.user_remaining_seconds - used)
        return self.user_remaining_seconds

    def model_clock_display(self) -> float:
        """Seconds remaining on model's MAIN clock RIGHT NOW."""
        if self.is_model_turn():
            used = self._main_time_consumed(self._elapsed_active(),
                                            self.model_tc.delay_seconds)
            return max(0.0, self.model_remaining_seconds - used)
        return self.model_remaining_seconds

    def user_delay_remaining(self) -> float:
        """Seconds left in the user's delay window this turn. Returns 0.0
        if past the delay, if delay=0, or if it's not the user's turn."""
        if self.is_user_turn() and self._active_clock_start is not None:
            return max(0.0, self.user_tc.delay_seconds - self._elapsed_active())
        return 0.0

    def model_delay_remaining(self) -> float:
        """Seconds left in the model's delay window this turn."""
        if self.is_model_turn() and self._active_clock_start is not None:
            return max(0.0, self.model_tc.delay_seconds - self._elapsed_active())
        return 0.0

    # ── Applying moves ──────────────────────────────────────────────────

    def apply_move(self, move: Move, search_snapshot: Optional[dict] = None) -> bool:
        """Apply the move whose turn it currently is. Returns True on success.

        Stops the mover's clock, applies increment, starts the other side's
        clock. Records a MoveRecord. Updates outcome if the game ended.

        search_snapshot is only meaningful when the model is moving.
        """
        if not self.is_ongoing():
            return False

        mover_side = self.game.current_player
        is_user_move = (mover_side == self.user_color)

        # Elapsed since turn start (covers both delay window + main-clock use).
        thinking = self._elapsed_active()

        # Simple-delay semantics: the first `delay_seconds` are "free" — no
        # main-clock deduction. Only the excess counts against remaining.
        tc = self.user_tc if is_user_move else self.model_tc
        time_from_main = self._main_time_consumed(thinking, tc.delay_seconds)

        # Time forfeit check: only fires if the excess-over-delay drained
        # the main clock. Moves completed within the delay window can NEVER
        # forfeit (used=0).
        if is_user_move:
            new_remaining = self.user_remaining_seconds - time_from_main
            if new_remaining <= 0:
                self.user_remaining_seconds = 0.0
                self._active_clock_start = None
                self.outcome = OUTCOME_YOU_FLAGGED
                self.outcome_detail = "You ran out of time"
                return False
            clock_before = self.user_remaining_seconds
            self.user_remaining_seconds = new_remaining + self.user_tc.increment_seconds
            clock_after = self.user_remaining_seconds
        else:
            new_remaining = self.model_remaining_seconds - time_from_main
            if new_remaining <= 0:
                self.model_remaining_seconds = 0.0
                self._active_clock_start = None
                self.outcome = OUTCOME_MODEL_FLAGGED
                self.outcome_detail = "Model ran out of time"
                return False
            clock_before = self.model_remaining_seconds
            self.model_remaining_seconds = new_remaining + self.model_tc.increment_seconds
            clock_after = self.model_remaining_seconds

        # Apply on the underlying game (assumes caller already validated legality).
        ok = self.game.make_move(move)
        if not ok:
            # Roll back the clock deduction — this shouldn't happen if caller
            # validated the move.
            if is_user_move:
                self.user_remaining_seconds = clock_before
            else:
                self.model_remaining_seconds = clock_before
            return False

        # Record.
        self.moves.append(MoveRecord(
            ply=len(self.moves),
            side=mover_side,
            move=move,
            clock_before_seconds=clock_before,
            clock_after_seconds=clock_after,
            thinking_time_seconds=thinking,
            search_snapshot=search_snapshot,
        ))

        # Game-end check (extinction).
        if self.game.game_over:
            self._active_clock_start = None
            if self.game.winner is None:
                self.outcome = OUTCOME_DRAW
                self.outcome_detail = getattr(self.game, "draw_reason", "") or "Draw"
            elif self.game.winner == self.user_color:
                self.outcome = OUTCOME_YOU_WIN_EXTINCT
                self.outcome_detail = "You extinct a piece type"
            else:
                self.outcome = OUTCOME_MODEL_WIN_EXTINCT
                self.outcome_detail = "Model extinct a piece type"
            return True

        # Swap clocks — start the other side's.
        self._active_clock_start = time.monotonic()
        return True

    # ── Takeback ────────────────────────────────────────────────────────

    def take_back(self, n_plies: int = 2) -> bool:
        """Undo the last `n_plies` moves. Returns True on success.

        Rebuilds the game position, restores per-side clocks to their
        post-move-N state, clears any terminal outcome, and starts a
        fresh timing window for whoever is now to move. The reason the
        default is 2 is so that a user-then-model pair collapses back
        to the user's original decision point.

        Fails (returns False) if n_plies is out of range.
        """
        if n_plies < 1 or n_plies > len(self.moves):
            return False
        # Index of the ply we want to be "post-" after taking back.
        # E.g. 5 moves total, n=2 → target ply-index = 2 (post-move-2 =
        # after 3 moves; the remaining moves list has length 3).
        target_idx = len(self.moves) - 1 - n_plies

        # Rebuild game state and read clocks BEFORE popping so the
        # existing self.moves indices are still valid.
        rebuilt = self.reconstruct_at(target_idx)
        user_t, model_t = self.clocks_at(target_idx)

        del self.moves[-n_plies:]
        self.game = rebuilt
        self.user_remaining_seconds = user_t
        self.model_remaining_seconds = model_t
        self.outcome = OUTCOME_ONGOING
        self.outcome_detail = ""
        # Restart the currently-to-move side's clock right now.
        self._active_clock_start = time.monotonic()
        return True

    # ── Passive flag check ──────────────────────────────────────────────

    def check_flag(self) -> bool:
        """If it's someone's turn and their MAIN clock has run out RIGHT
        NOW (accounting for the delay window), register a time forfeit and
        return True. Otherwise False. Cheap; call every frame.

        The player is protected during the delay window — flag can only
        fall AFTER delay_seconds have elapsed."""
        if not self.is_ongoing() or self._active_clock_start is None:
            return False
        elapsed = self._elapsed_active()
        if self.is_user_turn():
            used = self._main_time_consumed(elapsed, self.user_tc.delay_seconds)
            if self.user_remaining_seconds - used <= 0:
                self.user_remaining_seconds = 0.0
                self._active_clock_start = None
                self.outcome = OUTCOME_YOU_FLAGGED
                self.outcome_detail = "You ran out of time"
                return True
        else:  # model turn
            used = self._main_time_consumed(elapsed, self.model_tc.delay_seconds)
            if self.model_remaining_seconds - used <= 0:
                self.model_remaining_seconds = 0.0
                self._active_clock_start = None
                self.outcome = OUTCOME_MODEL_FLAGGED
                self.outcome_detail = "Model ran out of time"
                return True
        return False

    # ── Review-mode helpers (Phase 3) ───────────────────────────────────

    def reconstruct_at(self, ply_index: int) -> ExtinctionChess:
        """Rebuild the board state AFTER move index `ply_index` was played.

        ply_index=-1 means "before any moves" (initial position).
        ply_index=0 means "after move 0 (the first move) was played".
        ply_index >= len(moves) is clamped to the last move.

        Runs in O(ply_index) since we replay moves from scratch. For
        extinction chess games (~40 moves), this is a few milliseconds —
        cheap enough to run on every review-navigation click.
        """
        g = ExtinctionChess()
        stop_at = min(ply_index, len(self.moves) - 1)
        for i in range(stop_at + 1):
            g.make_move(self.moves[i].move)
        return g

    def clocks_at(self, ply_index: int) -> "tuple[float, float]":
        """(user_clock, model_clock) as they were AFTER move ply_index.

        Reconstructs by walking the move history. Only mover's clock is
        modified per move (thinking_time deducted then increment added).
        ply_index=-1 means "before any moves" — starting clocks.
        """
        user_t = self.user_tc.base_seconds
        model_t = self.model_tc.base_seconds
        stop_at = min(ply_index, len(self.moves) - 1)
        for i in range(stop_at + 1):
            rec = self.moves[i]
            if rec.side == self.user_color:
                user_t = rec.clock_after_seconds
            else:
                model_t = rec.clock_after_seconds
        return user_t, model_t

    # ── Convenience for engine budget ───────────────────────────────────

    def model_thinking_budget_seconds(self) -> float:
        """Time management: what should the model spend on this move?

        Heuristic: `remaining / expected_moves_left + increment`, with:
          - Ceiling scaling to time control (never > 30s on short games;
            longer games get proportionally more headroom)
          - Safety guard when time is critically low (never risk more
            than 30% of remaining on a single move once <10s left)
          - NO artificial floor — with pondering + tree reuse, the
            model can play near-instantly from an accumulated tree.
            Floor removed Aug 30.
          - Delay is added on top as FREE time — spending it doesn't
            eat the main clock, so we should always use it when available.
        """
        if not self.is_model_turn():
            return 0.0
        remaining = self.model_clock_display()
        delay = self.model_tc.delay_seconds

        # Base: expected 30 moves left.
        base_budget = remaining / 30.0 + self.model_tc.increment_seconds

        # Ceiling scales with time control.
        hard_cap = max(30.0, self.model_tc.base_seconds / 4.0)
        budget = min(base_budget, hard_cap)

        # Time-trouble safety: never risk more than 30% of remaining if
        # main clock is low. Applied BEFORE adding delay so delay stays free.
        if remaining < 10.0:
            budget = min(budget, remaining * 0.3)

        # Delay is free time — add on top of the main-clock budget so the
        # model uses it every turn without depleting reserves.
        return budget + delay
