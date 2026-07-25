"""Measure MCTS subtree reuse savings.

Plays a single deterministic self-play game with reuse enabled, then reports
how many sims per move end up in the subtree that gets promoted (i.e., the
sims we would save on the NEXT move).

For each move M(i):
  - MCTS runs num_simulations sims from the current position
  - The chosen child C has visit_count V — those V sims descended into it
  - When we promote C as the new root for M(i+1), we inherit those V visits
  - So the "savings" on M(i+1) = V

Aggregate stats across a game show the expected speedup from reuse.

Usage:
    python tools/measure_reuse_savings.py --model models/az_iter850.pt
    python tools/measure_reuse_savings.py --model models/az_iter100.pt --sims 800 --moves 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Add src/ to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from extinction_chess import ExtinctionChess, Color  # noqa: E402
from alphazero import (  # noqa: E402
    AlphaZeroNet, AlphaZeroEvaluator, mcts_search, descend_root,
)


def play_and_measure(evaluator, num_sims, max_moves, verbose=False):
    """Play a self-play game with reuse enabled. Return per-move stats."""
    game = ExtinctionChess()
    move_records = []   # list of (move_num, move, top_visits, total_visits, reused_flag)
    prev_root = None
    prev_move = None

    while not game.game_over and len(move_records) < max_moves:
        # Try to reuse subtree
        promoted = None
        if prev_root is not None and prev_move is not None:
            promoted = descend_root(prev_root, [prev_move])

        result = mcts_search(
            game, evaluator,
            num_simulations=num_sims,
            dirichlet_alpha=0.0, noise_weight=0.0,
            tactical_shortcuts=False,
            prev_root=promoted,
            return_root=True,
        )
        move_visits, root_value, this_root = result

        if not move_visits:
            break

        best_move, best_visits = max(move_visits, key=lambda x: x[1])
        total_visits = sum(v for _, v in move_visits)

        move_num = len(move_records) + 1
        reused = (promoted is not None)
        move_records.append((move_num, best_move, best_visits, total_visits, reused))

        if verbose:
            reuse_marker = "R" if reused else "F"
            saves_pct = 100.0 * best_visits / num_sims
            print(f"  Move {move_num:>2} [{reuse_marker}] {best_move} | "
                  f"top_visits={best_visits}/{total_visits} | "
                  f"next move saves ~{best_visits}/{num_sims} sims ({saves_pct:.1f}%)",
                  flush=True)

        game.make_move(best_move)
        prev_root = this_root
        prev_move = best_move

    return move_records


def summarize(records, num_sims):
    """Print a summary of reuse savings."""
    if len(records) < 2:
        print(f"\nOnly {len(records)} moves played — need at least 2 to measure savings")
        return

    # Move i's top_visits becomes savings on move i+1
    # So savings-applicable moves are indexed 1..N-1 (0-indexed)
    # The "savings for move k+1" = top_visits at move k = records[k][2]
    savings = [r[2] for r in records[:-1]]

    avg = sum(savings) / len(savings)
    median = sorted(savings)[len(savings) // 2]
    minimum = min(savings)
    maximum = max(savings)

    remaining = num_sims - avg
    speedup = num_sims / remaining if remaining > 0 else float('inf')

    print(f"\n{'='*60}")
    print(f"REUSE SAVINGS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total moves played:     {len(records)}")
    print(f"  Moves with reuse:       {len(savings)}")
    print(f"  Sims per move:          {num_sims}")
    print(f"")
    print(f"  Savings distribution (sims inherited per reuse):")
    print(f"    Min:                  {minimum}   ({100*minimum/num_sims:.1f}% of budget)")
    print(f"    Median:               {median}   ({100*median/num_sims:.1f}% of budget)")
    print(f"    Average:              {avg:.1f}  ({100*avg/num_sims:.1f}% of budget)")
    print(f"    Max:                  {maximum}   ({100*maximum/num_sims:.1f}% of budget)")
    print(f"")
    print(f"  Average new sims/move:  {remaining:.1f} (vs {num_sims} without reuse)")
    print(f"  Expected speedup:       {speedup:.2f}x")

    # Print distribution histogram
    print(f"\n  Per-move savings (chronological):")
    for r in records[:-1]:
        move_num, move, top, total, reused = r
        marker = "R" if reused else "F"
        bar = "#" * int(50 * top / num_sims)
        print(f"    Move {move_num:>2} [{marker}]: {top:>4} sims saved  |{bar:<50}|")


def main():
    parser = argparse.ArgumentParser(description="Measure MCTS subtree reuse savings")
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint (relative to project root)")
    parser.add_argument("--sims", type=int, default=800,
                        help="Simulations per move (default: 800, matches training)")
    parser.add_argument("--moves", type=int, default=30,
                        help="Max moves to play in the game")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each move as it's played")
    args = parser.parse_args()

    model_path = os.path.join(_PROJECT_ROOT, args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_path}...")
    model, meta = AlphaZeroNet.load_checkpoint(model_path)
    evaluator = AlphaZeroEvaluator(model, device=args.device)
    print(f"  iteration = {meta.get('iteration', '?')}, device = {args.device}")

    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\nPlaying self-play game with reuse enabled ({args.sims} sims/move, "
          f"up to {args.moves} moves)...")
    start = time.time()
    records = play_and_measure(evaluator, args.sims, args.moves, verbose=args.verbose)
    elapsed = time.time() - start
    print(f"\n  Game done in {elapsed:.1f}s")

    summarize(records, args.sims)


if __name__ == "__main__":
    main()
