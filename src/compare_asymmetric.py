"""
Asymmetric head-to-head comparison between two AlphaZero checkpoints.

Unlike compare_extensive.py, this pairs the two models at DIFFERENT sim
counts. Purpose: measure whether an older (usually weaker at equal sims)
model can outperform a newer one when given a depth advantage. If yes,
that quantifies untapped MCTS-extractable value in the older model —
which is the motivating signal for potentially injecting "expert" games
(older model + higher sims) into the training data pipeline.

Convention:
    --m1 = the "shallower" side (typically the newer/currently-strongest model)
    --m2 = the "deeper" side  (typically the older/reference model)

For each SIM_PAIR (m1_sims, m2_sims), plays 2 games per opening position
(m1=W then m2=W), so 20 openings × 2 = 40 games per pair.

Standard SIM_PAIRS shifts everything up by one preset tier:
    [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800)]
The last pair (400 vs 800) is beyond compare_extensive.py's cap.

Usage:
    python3 compare_asymmetric.py --m1 az_iter_1020_100pct.pt \
                                  --m2 az_iter_970_100pct.pt \
                                  --device cuda
    python3 compare_asymmetric.py --m1 az_iter_1020_100pct.pt \
                                  --m2 az_iter_970_100pct.pt \
                                  --sim-pairs 100,200 200,400 400,800

MCTS parameters match compare_extensive.py (dirichlet=0, noise=0,
tactical_shortcuts=False, c_puct=2.5) so results are reproducible and
comparable to normal benchmark output.
"""
import argparse
import os
import time

import torch  # noqa: F401  (required to init CUDA before AlphaZero imports)
import numpy as np  # noqa: F401  (used by AlphaZero internals)

from extinction_chess import ExtinctionChess, Color, Move  # noqa: F401
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search


def _copy_game(game):
    """Copy a game state (C++ objects can't be deepcopied)."""
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DEFAULT_SIM_PAIRS = [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800)]
MAX_MOVES = 300


def load_model(path, device="cpu"):
    model, meta = AlphaZeroNet.load_checkpoint(path)
    evaluator = AlphaZeroEvaluator(model, device=device)
    iteration = meta.get("iteration", "?")
    return evaluator, iteration


def play_game(game_state,
              white_eval, white_sims, white_label,
              black_eval, black_sims, black_label):
    """Play a game from a given position with per-side sim counts.

    Returns (result_from_white_perspective, num_moves):
      +1 white wins, -1 black wins, 0 draw.
    """
    game = _copy_game(game_state)
    moves = 0
    while not game.game_over and moves < MAX_MOVES:
        side = "W" if game.current_player == Color.WHITE else "B"
        if game.current_player == Color.WHITE:
            evaluator, sims, label = white_eval, white_sims, white_label
        else:
            evaluator, sims, label = black_eval, black_sims, black_label
        mv, _ = mcts_search(
            game, evaluator,
            num_simulations=sims,
            dirichlet_alpha=0, noise_weight=0,
            tactical_shortcuts=False,
        )
        if not mv:
            break
        best = max(mv, key=lambda x: x[1])[0]
        moves += 1
        print(f"        {moves:>3}. {side} ({label}, {sims}s) {best}", flush=True)
        game.make_move(best)

    if game.winner == Color.WHITE:
        print(f"        -> White ({white_label} @ {white_sims}s) wins in {moves} moves",
              flush=True)
        return 1, moves
    elif game.winner == Color.BLACK:
        print(f"        -> Black ({black_label} @ {black_sims}s) wins in {moves} moves",
              flush=True)
        return -1, moves
    print(f"        -> Draw ({moves} moves)", flush=True)
    return 0, moves


def _parse_sim_pairs(pair_strings):
    """Parse CLI-provided sim pairs like ['100,200', '200,400']."""
    pairs = []
    for s in pair_strings:
        parts = s.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"--sim-pairs entry must be 'm1_sims,m2_sims', got {s!r}")
        try:
            pairs.append((int(parts[0]), int(parts[1])))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--sim-pairs entry must be two ints, got {s!r}")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Asymmetric head-to-head comparison at different sim counts")
    parser.add_argument("--m1", required=True,
                        help="Model 1 filename (in models/ dir) — 'shallower' side")
    parser.add_argument("--m2", required=True,
                        help="Model 2 filename (in models/ dir) — 'deeper' side")
    parser.add_argument("--sim-pairs", nargs="+", type=str,
                        default=[f"{a},{b}" for a, b in DEFAULT_SIM_PAIRS],
                        help=("Sim pairs as 'm1,m2' strings. Default: "
                              f"{' '.join(f'{a},{b}' for a, b in DEFAULT_SIM_PAIRS)}"))
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    sim_pairs = _parse_sim_pairs(args.sim_pairs)

    path1 = os.path.join(MODELS_DIR, args.m1)
    path2 = os.path.join(MODELS_DIR, args.m2)

    print("Loading models...")
    eval1, iter1 = load_model(path1, args.device)
    eval2, iter2 = load_model(path2, args.device)
    label1 = f"iter {iter1}"
    label2 = f"iter {iter2}"
    print(f"  M1 ('shallower'): {label1} ({args.m1})")
    print(f"  M2 ('deeper'):    {label2} ({args.m2})")
    print(f"  Device: {args.device}")

    # Openings
    start_game = ExtinctionChess()
    opening_moves = start_game.get_legal_moves()
    print(f"\n{len(opening_moves)} opening moves, {len(sim_pairs)} sim pairs")
    total_games = len(opening_moves) * len(sim_pairs) * 2
    print(f"Total games: {total_games}\n")

    # Results per sim pair
    pair_results = {pair: {"m1": 0.0, "m2": 0.0,
                           "m1w": 0, "m2w": 0, "draws": 0}
                    for pair in sim_pairs}

    game_count = 0
    t_start = time.time()

    for m1_sims, m2_sims in sim_pairs:
        print(f"\n{'='*60}")
        print(f"  SIM PAIR: {label1} @ {m1_sims}  vs  {label2} @ {m2_sims}")
        print(f"  Depth advantage: {label2} gets {m2_sims/m1_sims:.2f}x more sims")
        print(f"{'='*60}")

        for move in opening_moves:
            game_count += 2
            print(f"\n  [{game_count}/{total_games}] Opening: 1. {move}", flush=True)

            post_opening = _copy_game(start_game)
            post_opening.make_move(move)

            m1_score = 0.0
            m2_score = 0.0

            # Game 1: M1 is white (@ m1_sims), M2 is black (@ m2_sims)
            print(f"      Game 1: {label1}(W,{m1_sims}s) vs {label2}(B,{m2_sims}s)",
                  flush=True)
            r1, _ = play_game(post_opening,
                              eval1, m1_sims, label1,
                              eval2, m2_sims, label2)

            # Game 2: M2 is white (@ m2_sims), M1 is black (@ m1_sims)
            print(f"      Game 2: {label2}(W,{m2_sims}s) vs {label1}(B,{m1_sims}s)",
                  flush=True)
            r2, _ = play_game(post_opening,
                              eval2, m2_sims, label2,
                              eval1, m1_sims, label1)

            # Score from M1's perspective
            if r1 == 1:
                m1_score += 1
            elif r1 == -1:
                m2_score += 1
            else:
                m1_score += 0.5
                m2_score += 0.5

            if r2 == 1:
                m2_score += 1
            elif r2 == -1:
                m1_score += 1
            else:
                m1_score += 0.5
                m2_score += 0.5

            pr = pair_results[(m1_sims, m2_sims)]
            pr["m1"] += m1_score
            pr["m2"] += m2_score
            if m1_score > m2_score:
                pr["m1w"] += 1
            elif m2_score > m1_score:
                pr["m2w"] += 1
            else:
                pr["draws"] += 1

            print(f"      Result: {label1} {m1_score}-{m2_score} {label2}", flush=True)

        # Per-pair summary
        pr = pair_results[(m1_sims, m2_sims)]
        max_pts_pair = len(opening_moves) * 2
        print(f"\n  Summary ({label1}@{m1_sims} vs {label2}@{m2_sims}):")
        print(f"    {label1}: {pr['m1']:.1f} / {max_pts_pair}  ({pr['m1']/max_pts_pair*100:.1f}%)")
        print(f"    {label2}: {pr['m2']:.1f} / {max_pts_pair}  ({pr['m2']/max_pts_pair*100:.1f}%)")
        print(f"    Match wins: {label1} {pr['m1w']} - {pr['draws']} - {pr['m2w']} {label2}")

    # Overall summary
    elapsed = time.time() - t_start
    total_m1 = sum(pr["m1"] for pr in pair_results.values())
    total_m2 = sum(pr["m2"] for pr in pair_results.values())
    max_pts = len(opening_moves) * len(sim_pairs) * 2

    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"  {label1} ('shallower'): {total_m1:.1f} / {max_pts}  ({total_m1/max_pts*100:.1f}%)")
    print(f"  {label2} ('deeper'):    {total_m2:.1f} / {max_pts}  ({total_m2/max_pts*100:.1f}%)")
    print(f"\n  Per sim pair:")
    print(f"  {'M1 sims':>8}  {'M2 sims':>8}  {label1:>12}  {label2:>12}  {'M1 W-D-L':>10}")
    print(f"  {'-'*62}")
    for m1_sims, m2_sims in sim_pairs:
        pr = pair_results[(m1_sims, m2_sims)]
        print(f"  {m1_sims:>8}  {m2_sims:>8}  {pr['m1']:>12.1f}  {pr['m2']:>12.1f}  "
              f"{pr['m1w']:>2}-{pr['draws']}-{pr['m2w']}")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Human-readable verdict hint
    print(f"\n{'='*60}")
    print(f"  INTERPRETATION")
    print(f"{'='*60}")
    for m1_sims, m2_sims in sim_pairs:
        pr = pair_results[(m1_sims, m2_sims)]
        max_pts_pair = len(opening_moves) * 2
        m2_pct = pr['m2'] / max_pts_pair * 100
        ratio = m2_sims / m1_sims
        if m2_pct > 55:
            verdict = f"{label2} STRONGER with {ratio:.1f}x depth"
        elif m2_pct < 45:
            verdict = f"{label1} STRONGER even with {ratio:.1f}x depth disadvantage"
        else:
            verdict = f"TIE (within ±5% noise floor)"
        print(f"  @ {m1_sims:>4}s vs {m2_sims:>4}s ({ratio:.1f}x): {label2} = {m2_pct:.1f}%  → {verdict}")


if __name__ == "__main__":
    main()
