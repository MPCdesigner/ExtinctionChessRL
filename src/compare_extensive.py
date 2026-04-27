"""
Extensive head-to-head comparison between two AlphaZero checkpoints.

For each of white's 20 legal opening moves, plays a match (2 games: each
model as white) at each sim setting. With 5 sim settings, that's
20 openings × 5 sims × 2 games = 200 games total.

Usage:
    python3 compare_extensive.py --m1 az_iter_100_100pct.pt --m2 az_iter_160_100pct.pt
    python3 compare_extensive.py --m1 az_iter_100_100pct.pt --m2 az_iter_160_100pct.pt --device cuda
"""
import argparse
import os
import time

import torch
import numpy as np

from extinction_chess import ExtinctionChess, Color, Move
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
SIM_OPTIONS = [20, 50, 100, 200, 400]
MAX_MOVES = 300


def load_model(path, device="cpu"):
    model, meta = AlphaZeroNet.load_checkpoint(path)
    evaluator = AlphaZeroEvaluator(model, device=device)
    iteration = meta.get("iteration", "?")
    return evaluator, iteration


def play_game(game_state, white_eval, black_eval, sims,
              white_label="W", black_label="B"):
    """Play a game from a given position. Returns +1 white wins, -1 black wins, 0 draw."""
    game = _copy_game(game_state)
    moves = 0
    while not game.game_over and moves < MAX_MOVES:
        side = "W" if game.current_player == Color.WHITE else "B"
        evaluator = white_eval if game.current_player == Color.WHITE else black_eval
        label = white_label if game.current_player == Color.WHITE else black_label
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
        print(f"        {moves:>3}. {side} ({label}) {best}", flush=True)
        game.make_move(best)

    if game.winner == Color.WHITE:
        print(f"        -> White ({white_label}) wins in {moves} moves", flush=True)
        return 1, moves
    elif game.winner == Color.BLACK:
        print(f"        -> Black ({black_label}) wins in {moves} moves", flush=True)
        return -1, moves
    print(f"        -> Draw ({moves} moves)", flush=True)
    return 0, moves


def main():
    parser = argparse.ArgumentParser(description="Extensive head-to-head comparison")
    parser.add_argument("--m1", required=True, help="Model 1 filename (in models/ dir)")
    parser.add_argument("--m2", required=True, help="Model 2 filename (in models/ dir)")
    parser.add_argument("--sims", nargs="+", type=int, default=SIM_OPTIONS,
                        help=f"Sim settings (default: {SIM_OPTIONS})")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    path1 = os.path.join(MODELS_DIR, args.m1)
    path2 = os.path.join(MODELS_DIR, args.m2)

    print("Loading models...")
    eval1, iter1 = load_model(path1, args.device)
    eval2, iter2 = load_model(path2, args.device)
    label1 = f"iter {iter1}"
    label2 = f"iter {iter2}"
    print(f"  M1: {label1} ({args.m1})")
    print(f"  M2: {label2} ({args.m2})")

    # Get all 20 legal opening moves
    start_game = ExtinctionChess()
    opening_moves = start_game.get_legal_moves()
    print(f"\n{len(opening_moves)} opening moves, {len(args.sims)} sim settings")
    total_games = len(opening_moves) * len(args.sims) * 2
    print(f"Total games: {total_games}\n")

    # Results tracking per sim setting
    # For each sim: m1_score, m2_score, m1_wins, m2_wins, draws
    sim_results = {s: {"m1": 0.0, "m2": 0.0, "m1w": 0, "m2w": 0, "draws": 0}
                   for s in args.sims}
    # Per-opening results
    opening_results = {}  # (opening_move, sims) -> (m1_score, m2_score)

    game_count = 0
    t_start = time.time()

    for sims in args.sims:
        print(f"\n{'='*60}")
        print(f"  SIM SETTING: {sims}")
        print(f"{'='*60}")

        for move in opening_moves:
            game_count += 2
            print(f"\n  [{game_count}/{total_games}] Opening: 1. {move} ({sims} sims)",
                  flush=True)

            # Create position after opening move
            post_opening = _copy_game(start_game)
            post_opening.make_move(move)

            m1_score = 0.0
            m2_score = 0.0

            # Game 1: M1 is white (but black to move since opening already played)
            # Wait — M1 played the opening move as white, now M2 responds as black
            # Actually: we want each model to play both sides from this position.
            # The opening move was played by "white". Now it's black's turn.
            # Game 1: M1=white, M2=black (M2 moves next)
            # Game 2: M2=white, M1=black (M1 moves next)
            # But the opening move already committed white's first move.
            # So Game 1: M1 "was" white, M2 responds. Game 2: M2 "was" white, M1 responds.

            print(f"      Game 1: {label1}(W) vs {label2}(B)", flush=True)
            r1, moves1 = play_game(post_opening, eval1, eval2, sims,
                                   white_label=label1, black_label=label2)

            print(f"      Game 2: {label2}(W) vs {label1}(B)", flush=True)
            r2, moves2 = play_game(post_opening, eval2, eval1, sims,
                                   white_label=label2, black_label=label1)

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

            sim_results[sims]["m1"] += m1_score
            sim_results[sims]["m2"] += m2_score
            if m1_score > m2_score:
                sim_results[sims]["m1w"] += 1
            elif m2_score > m1_score:
                sim_results[sims]["m2w"] += 1
            else:
                sim_results[sims]["draws"] += 1

            opening_results[(str(move), sims)] = (m1_score, m2_score)
            print(f"      Result: {label1} {m1_score}-{m2_score} {label2}", flush=True)

        # Print sim summary
        r = sim_results[sims]
        print(f"\n  Summary ({sims} sims):")
        print(f"    {label1}: {r['m1']:.1f} pts ({r['m1w']}W {r['draws']}D {r['m2w']}L)")
        print(f"    {label2}: {r['m2']:.1f} pts ({r['m2w']}W {r['draws']}D {r['m1w']}L)")

    # Overall summary
    elapsed = time.time() - t_start
    total_m1 = sum(r["m1"] for r in sim_results.values())
    total_m2 = sum(r["m2"] for r in sim_results.values())
    max_pts = len(opening_moves) * len(args.sims) * 2  # 2 points per opening per sim

    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"  {label1}: {total_m1:.1f} / {max_pts}  ({total_m1/max_pts*100:.1f}%)")
    print(f"  {label2}: {total_m2:.1f} / {max_pts}  ({total_m2/max_pts*100:.1f}%)")
    print(f"\n  Per sim setting:")
    print(f"  {'Sims':>6}  {label1:>12}  {label2:>12}  {'M1 W-D-L':>10}")
    print(f"  {'-'*48}")
    for sims in args.sims:
        r = sim_results[sims]
        print(f"  {sims:>6}  {r['m1']:>12.1f}  {r['m2']:>12.1f}  "
              f"{r['m1w']:>2}-{r['draws']}-{r['m2w']}")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
