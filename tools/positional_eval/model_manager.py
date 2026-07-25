"""Model loader and evaluator for the positional evaluation tool.

Loads N models at startup (fixed set — user picks them once). Provides a
uniform interface for evaluating a position at either raw NN (1 sim) or
MCTS at a given sim count.

MCTS is run deterministically for this tool:
  - dirichlet_alpha=0, noise_weight=0   -> reproducible results
  - tactical_shortcuts=False            -> raw model signal (no auto-take)

mcts_search returns a SEARCH-REFINED Q-value (root.value_sum / root.visit_count
converted to current-player perspective), not the raw NN value. So value
does change with sim count. The move distribution is also search-refined
(visit counts).

Evaluations take a `progress_callback` so the UI can display live progress
without freezing. Format:
    progress_callback(stage, model_label, sim_count, sims_done, sims_total)
where stage is one of "start", "progress", "done", "all_done".
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Add src/ to sys.path so we can import project modules
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from alphazero import (  # noqa: E402
    AlphaZeroNet, AlphaZeroEvaluator, mcts_search, move_to_index,
)
from extinction_chess import Color, ExtinctionChess, Move  # noqa: E402


# Sim options the tool exposes. 1 = raw NN (no search).
SIM_OPTIONS: Tuple[int, ...] = (1, 20, 50, 100, 200, 400, 800)


@dataclass
class EvalResult:
    """One evaluation output.

    Attributes:
        value: scalar position value in WHITE'S PERSPECTIVE. Range [-1, +1].
               +1 means White is winning; -1 means Black is winning; 0 is
               even. Does NOT depend on whose turn it is — this is the
               human-readable "who is ahead in this position" convention.
               Internally the model and MCTS both natively output values
               from current-player perspective; we flip the sign for
               positions where Black is to move so callers see a consistent
               white-perspective number.
        moves: list of (Move, probability, visit_count) sorted by probability
               desc. Illegal moves are excluded. Probabilities are normalized
               to sum to 1 across the legal moves shown. visit_count is the
               raw MCTS visit count for the move (0 for raw NN output where
               there is no search — display should hide zero counts).
    """
    value: float
    moves: List[Tuple[Move, float, int]] = field(default_factory=list)


class LoadedModel:
    """One loaded model + its evaluator + friendly label."""

    def __init__(self, path: str, device: str = "cpu"):
        self.path = os.path.abspath(path)
        self.filename = os.path.basename(path)
        self.label = self._parse_label(self.filename)

        # migrate=True handles the 14→115 channel migration for older checkpoints
        model, meta = AlphaZeroNet.load_checkpoint(self.path, migrate=True)
        self.model = model
        self.meta = meta or {}
        self.iteration = self.meta.get("iteration")
        self.win_rate = self.meta.get("win_rate")
        self.evaluator = AlphaZeroEvaluator(model, device=device)

    @staticmethod
    def _parse_label(filename: str) -> str:
        """Extract a friendly label from a checkpoint filename.

        Handles both `az_iter_770_100pct.pt` and `az_iter770.pt` naming.
        Falls back to the filename stem.
        """
        m = re.search(r"iter[_]?(\d+)", filename)
        if m:
            return f"iter {m.group(1)}"
        return filename.rsplit(".", 1)[0]

    # ── Evaluation ──────────────────────────────────────────────────────────

    def _terminal_result(self, game: ExtinctionChess) -> EvalResult:
        """Return value in WHITE'S perspective for a terminal game."""
        if game.winner == Color.WHITE:
            v = 1.0
        elif game.winner == Color.BLACK:
            v = -1.0
        else:
            v = 0.0
        return EvalResult(value=v, moves=[])

    @staticmethod
    def _to_white_perspective(value: float, game: ExtinctionChess) -> float:
        """Convert a current-player-perspective value to White's perspective.

        The network and MCTS both output values from the perspective of the
        player to move. For display we want a single sign convention that
        doesn't flip with whose turn it is: positive = White winning.
        """
        return value if game.current_player == Color.WHITE else -value

    def evaluate_raw(self, game: ExtinctionChess) -> EvalResult:
        """Raw NN value + policy prior, filtered to legal moves and renormalized.

        Value is converted to White's perspective (positive = White winning).
        """
        if game.game_over:
            return self._terminal_result(game)

        legal = game.get_legal_moves()
        if not legal:
            return EvalResult(value=0.0, moves=[])

        policy_logits, value = self.evaluator.evaluate_with_policy(game)
        # Convert current-player-perspective NN output to White's perspective.
        value = self._to_white_perspective(value, game)

        # Mask illegal moves: take logits only at legal indices, softmax, sort.
        move_indices = [move_to_index(m) for m in legal]
        move_logits = np.array([policy_logits[i] for i in move_indices],
                               dtype=np.float64)
        move_logits -= move_logits.max()  # for numerical stability
        probs = np.exp(move_logits)
        probs /= (probs.sum() + 1e-12)

        # Raw NN has no visit counts (no search performed) — mark with 0.
        pairs: List[Tuple[Move, float, int]] = [
            (m, p, 0) for m, p in zip(legal, probs.tolist())
        ]
        pairs.sort(key=lambda t: -t[1])
        return EvalResult(value=float(value), moves=pairs)

    def _result_from(self, move_visits, value: float) -> EvalResult:
        """Build an EvalResult from raw (move, visit_count) list + value."""
        total = sum(count for _, count in move_visits)
        pairs: List[Tuple[Move, float, int]] = []
        if total > 0:
            for move, count in move_visits:
                if count > 0:
                    pairs.append((move, count / total, int(count)))
        pairs.sort(key=lambda t: -t[1])
        return EvalResult(value=float(value), moves=pairs)

    def evaluate_mcts_multi(self, game: ExtinctionChess,
                            sim_counts: List[int],
                            sim_progress_callback=None
                            ) -> Dict[int, EvalResult]:
        """Run ONE MCTS to max(sim_counts), capturing snapshots at each
        intermediate checkpoint.

        Returns {sim_count: EvalResult} for every sim count requested.
        This is much faster than running mcts_search from scratch for each
        sim count — a single 800-sim run replaces separate 20+50+100+200+
        400+800 = 1570 sims of redundant work.
        """
        if game.game_over:
            terminal = self._terminal_result(game)
            return {s: terminal for s in sim_counts}

        if not sim_counts:
            return {}

        max_sims = max(sim_counts)
        captured: Dict[int, EvalResult] = {}

        def _on_checkpoint(sim_count_reached, move_visits, value):
            # Match this checkpoint to the closest requested sim count that
            # we haven't already captured. Convert value to White's
            # perspective before storing so display sign is consistent.
            if sim_count_reached in sim_counts and sim_count_reached not in captured:
                w_value = self._to_white_perspective(value, game)
                captured[sim_count_reached] = self._result_from(
                    move_visits, w_value)

        move_visits, root_value = mcts_search(
            game, self.evaluator,
            num_simulations=max_sims,
            dirichlet_alpha=0.0,
            noise_weight=0.0,
            tactical_shortcuts=False,
            progress_callback=sim_progress_callback,
            checkpoint_sims=sim_counts,
            checkpoint_callback=_on_checkpoint,
        )
        # Final: always capture at max_sims (in case the checkpoint fired
        # slightly early due to batch granularity).
        captured[max_sims] = self._result_from(
            move_visits, self._to_white_perspective(root_value, game))

        # Fill in any missing sim counts with the closest earlier snapshot
        # (rare — happens if a batch size never triggered exactly at the mark).
        prev = None
        for s in sorted(sim_counts):
            if s in captured:
                prev = s
            elif prev is not None:
                captured[s] = captured[prev]
        return {s: captured[s] for s in sim_counts if s in captured}


class ModelManager:
    """Owns a fixed list of loaded models.

    Constructed once at startup with the paths the user selected. Later,
    the UI decides which subset (via `model_indices`) and which sim counts
    to actually evaluate per click.
    """

    def __init__(self, model_paths: List[str],
                 device: Optional[str] = None,
                 verbose: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.models: List[LoadedModel] = []
        for path in model_paths:
            if verbose:
                print(f"[model_manager] loading {os.path.basename(path)}...",
                      flush=True)
            self.models.append(LoadedModel(path, device=device))
        if verbose:
            print(f"[model_manager] loaded {len(self.models)} model(s) "
                  f"on {device}", flush=True)

    # ── Accessors ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.models)

    def get_labels(self) -> List[str]:
        return [m.label for m in self.models]

    def get_model(self, index: int) -> LoadedModel:
        return self.models[index]

    # ── Batch evaluation ────────────────────────────────────────────────────

    def evaluate(self, game: ExtinctionChess,
                 model_indices: List[int],
                 sim_counts: List[int],
                 progress_callback=None
                 ) -> Dict[int, Dict[int, EvalResult]]:
        """Evaluate a position across selected models and sim counts.

        Args:
            game: the position to evaluate (an ExtinctionChess instance).
            model_indices: which loaded models to run.
            sim_counts: which sim counts to run per model. 1 = raw NN.
            progress_callback: optional callable of shape
                (stage, model_label, sim_count, sims_done, sims_total) -> None.
                stage is one of "start", "progress", "done", "all_done".
                "start" fires just before evaluating one cell; "progress"
                fires periodically during MCTS (sims_done < sims_total);
                "done" fires when the cell result is ready; "all_done" once
                everything finishes.

        Returns:
            Nested dict: results[model_index][sim_count] = EvalResult.
        """
        results: Dict[int, Dict[int, EvalResult]] = {}

        # Split requested sims into raw NN (1) vs MCTS (everything else).
        # MCTS is done as ONE run per model with checkpoints, so the progress
        # bar shows continuous progress from 0 to max_sims within a model.
        mcts_sims = sorted(s for s in sim_counts if s > 1)
        want_raw = (1 in sim_counts)

        for mi in model_indices:
            model = self.models[mi]
            per_sim: Dict[int, EvalResult] = {}

            # 1) Raw NN (fast, essentially instant)
            if want_raw:
                if progress_callback:
                    progress_callback("start", model.label, 1, 0, 1)
                per_sim[1] = model.evaluate_raw(game)
                if progress_callback:
                    progress_callback("done", model.label, 1, 1, 1)

            # 2) MCTS: single run to max_sims, capturing snapshots at each
            #    requested intermediate sim count.
            if mcts_sims:
                max_sims = max(mcts_sims)
                if progress_callback:
                    progress_callback("start", model.label, max_sims,
                                      0, max_sims)

                if progress_callback:
                    def _prog(done, total, _label=model.label,
                              _max=max_sims):
                        progress_callback(
                            "progress", _label, _max, done, total)
                else:
                    _prog = None

                mcts_results = model.evaluate_mcts_multi(
                    game, mcts_sims, sim_progress_callback=_prog)
                per_sim.update(mcts_results)

                if progress_callback:
                    progress_callback("done", model.label, max_sims,
                                      max_sims, max_sims)

            results[mi] = per_sim

        if progress_callback:
            progress_callback("all_done", "", 0, 0, 0)
        return results
