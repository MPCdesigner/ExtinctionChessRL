"""Test MCTS subtree reuse correctness.

Plays a short deterministic self-play game two ways — with subtree reuse OFF
and with subtree reuse ON — and compares outcomes at each move. Because MCTS
is deterministic under dirichlet_alpha=0, noise_weight=0, tactical_shortcuts=
False, and the ties in PUCT are broken by strict-greater-than, the two runs
should produce IDENTICAL trees. Any divergence indicates a bug in the reuse
implementation.

Runs entirely offline against a local checkpoint. No cluster required.

Usage:
    python tools/test_reuse.py
    python tools/test_reuse.py --sims 200 --moves 30 --model models/az_iter850.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# Add src/ to sys.path so we can import project modules
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
import alphazero  # to patch BATCH_SIZE_MCTS  # noqa: E402


def play_game(evaluator, num_sims, reuse_enabled, max_moves, verbose=False):
    """Play a deterministic self-play game.

    Returns a list of move records: each is (move, top_visits, total_visits,
    root_value, root_visit_count).
    """
    game = ExtinctionChess()
    records = []
    prev_root = None
    move_history = []

    while not game.game_over and len(records) < max_moves:
        promoted = None
        if reuse_enabled and prev_root is not None and len(move_history) >= 1:
            # Self-play: 1-ply descent (we made the last move, we're moving again next turn)
            promoted = descend_root(prev_root, [move_history[-1]])

        result = mcts_search(
            game, evaluator,
            num_simulations=num_sims,
            dirichlet_alpha=0.0, noise_weight=0.0,
            tactical_shortcuts=False,
            prev_root=promoted,
            return_root=True,
        )
        if len(result) != 3:
            raise RuntimeError(f"Expected 3-tuple from mcts_search, got {len(result)}")
        move_visits, root_value, this_root = result

        if not move_visits:
            break

        best_move, best_visits = max(move_visits, key=lambda x: x[1])
        total_visits = sum(v for _, v in move_visits)
        rvc = this_root.visit_count

        records.append((best_move, best_visits, total_visits, root_value, rvc))
        move_history.append(best_move)
        game.make_move(best_move)
        prev_root = this_root

        if verbose:
            reuse_marker = "R" if (promoted is not None) else "F"
            print(f"  Move {len(records):2d} [{reuse_marker}] "
                  f"{best_move} | visits {best_visits}/{total_visits} | "
                  f"root_visits={rvc} | root_q={root_value:+.4f}", flush=True)

    return records


def compare_records(records_fresh, records_reuse):
    """Return (all_match, per_move_matches) list.

    Move objects are compared via str() representation since the Move class
    doesn't implement __eq__ (default identity comparison would fail for
    equal moves from different game runs).
    """
    matches = []
    for i, (rf, rr) in enumerate(zip(records_fresh, records_reuse)):
        m_f, v_f, tv_f, rv_f, rvc_f = rf
        m_r, v_r, tv_r, rv_r, rvc_r = rr

        # Compare Move objects by their string representation (Move has no __eq__)
        same_move = (str(m_f) == str(m_r))
        same_top_visits = (v_f == v_r)
        same_total = (tv_f == tv_r)
        same_rv = abs(rv_f - rv_r) < 1e-6
        same_rvc = (rvc_f == rvc_r)

        match = same_move and same_top_visits and same_total and same_rv and same_rvc
        matches.append((match, same_move, same_top_visits, same_total, same_rv, same_rvc))
    length_match = (len(records_fresh) == len(records_reuse))
    return length_match and all(m[0] for m in matches), matches


def main():
    parser = argparse.ArgumentParser(description="Test MCTS subtree reuse correctness")
    parser.add_argument("--model", default="models/az_iter850.pt",
                        help="Path to model checkpoint (relative to project root)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--moves", type=int, default=20,
                        help="Max moves per game")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each move as it's played")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override BATCH_SIZE_MCTS in alphazero (default: 8). "
                             "Set to 1 to eliminate virtual-loss cross-sim effects "
                             "and prove reuse produces bit-exact-identical trees.")
    args = parser.parse_args()

    if args.batch_size is not None:
        print(f"Overriding BATCH_SIZE_MCTS: {alphazero.BATCH_SIZE_MCTS} -> {args.batch_size}")
        alphazero.BATCH_SIZE_MCTS = args.batch_size

    model_path = os.path.join(_PROJECT_ROOT, args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_path}...")
    model, meta = AlphaZeroNet.load_checkpoint(model_path)
    evaluator = AlphaZeroEvaluator(model, device=args.device)
    print(f"  iteration = {meta.get('iteration', '?')}, device = {args.device}")

    # Determinism: MCTS is deterministic given fixed model & inputs. But numpy
    # RNG might be seeded from prior imports. Seed explicitly for repeatability.
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\n== Fresh MCTS ({args.sims} sims/move, {args.moves} moves) ==")
    records_fresh = play_game(evaluator, args.sims, reuse_enabled=False,
                              max_moves=args.moves, verbose=args.verbose)

    np.random.seed(42)
    torch.manual_seed(42)

    print(f"\n== Reused MCTS ({args.sims} sims/move, {args.moves} moves) ==")
    records_reuse = play_game(evaluator, args.sims, reuse_enabled=True,
                              max_moves=args.moves, verbose=args.verbose)

    print("\n" + "=" * 76)
    print(f"{'Move':>4} {'Fresh':>32} {'Reuse':>32} {'Match':>6}")
    print("=" * 76)

    all_match, per_move = compare_records(records_fresh, records_reuse)

    n_shown = max(len(records_fresh), len(records_reuse))
    for i in range(n_shown):
        rf = records_fresh[i] if i < len(records_fresh) else None
        rr = records_reuse[i] if i < len(records_reuse) else None

        def fmt(r):
            if r is None:
                return "—"
            m, v, tv, rv, rvc = r
            return f"{m} v={v}/{tv} rvc={rvc} rv={rv:+.3f}"

        marker = "OK"
        if i >= len(per_move) or not per_move[i][0]:
            marker = "FAIL"

        print(f"{i+1:>4} {fmt(rf):>32} {fmt(rr):>32} {marker:>6}")

    print("=" * 76)
    if all_match:
        print("RESULT: PASS — reuse produces identical outcomes to fresh MCTS.")
        print("        Subtree reuse is correctness-preserving in this test.")
        sys.exit(0)
    else:
        print("RESULT: FAIL — reuse produced different outcomes.")
        print("        Investigate before enabling in benchmarks.")
        print("        Divergence details (per move):")
        for i, m in enumerate(per_move):
            if not m[0]:
                match, sm, stv, stot, srv, srvc = m
                print(f"          Move {i+1}: same_move={sm}, "
                      f"same_top_visits={stv}, same_total={stot}, "
                      f"same_root_value={srv}, same_root_vc={srvc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
