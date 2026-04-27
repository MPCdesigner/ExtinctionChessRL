"""
Win-taking test: play a head-to-head match between two AlphaZero checkpoints
and track how often each model misses an instant win (a legal move that
immediately ends the game by extinction).

Runs WITHOUT pygame — prints moves to stdout with visit counts.
Designed to run on Google Colab (CPU-only is fine for small sim counts).

Usage (local):
    python win_taking_test.py --m1 az_iter190.pt --m2 az_iter100.pt --sims 200

Colab setup:
    1. Upload src/ folder and model .pt files
    2. !pip install torch numpy pybind11
    3. !cd src && python setup.py build_ext --inplace
    4. !python win_taking_test.py --m1 az_iter190.pt --m2 az_iter100.pt --sims 200
"""

import argparse
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from extinction_chess import ExtinctionChess, Color
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search

MAX_MOVES = 300


def has_instant_win(game):
    """Check if the current player has any move that immediately wins."""
    current = game.current_player
    winning_moves = []
    for m in game.get_legal_moves():
        gc = ExtinctionChess()
        gc.board = game.board.copy()
        gc.current_player = game.current_player
        gc.game_over = game.game_over
        gc.winner = game.winner
        if gc.make_move(m) and gc.game_over and gc.winner == current:
            winning_moves.append(m)
    return winning_moves


def play_game(white_eval, black_eval, sims, white_label, black_label):
    """
    Play a single game, printing each move with visit counts.
    Flags missed instant wins with *** MISSED WIN ***.
    Returns (result, moves_played, missed_wins_white, missed_wins_black).
    result: +1 white wins, -1 black wins, 0 draw.
    """
    game = ExtinctionChess()
    moves = 0
    missed_w = 0
    missed_b = 0

    while not game.game_over and moves < MAX_MOVES:
        is_white = game.current_player == Color.WHITE
        side = "W" if is_white else "B"
        evaluator = white_eval if is_white else black_eval
        label = white_label if is_white else black_label

        # Check for instant wins before MCTS
        winning_moves = has_instant_win(game)

        # Run MCTS (no noise, deterministic)
        mv, root_val = mcts_search(
            game, evaluator,
            num_simulations=sims,
            dirichlet_alpha=0, noise_weight=0,
            tactical_shortcuts=False,
        )
        if not mv:
            break

        # Sort by visits descending for display
        mv_sorted = sorted(mv, key=lambda x: x[1], reverse=True)
        best_move = mv_sorted[0][0]
        best_visits = mv_sorted[0][1]
        moves += 1

        # Print move with top visit counts
        top_moves = mv_sorted[:5]  # show top 5 candidates
        visits_str = ", ".join(f"{m}:{v}" for m, v in top_moves)
        print(f"  {moves:>3}. {side} ({label}) {best_move}  "
              f"[{visits_str}]", flush=True)

        # Check if this was a missed win
        if winning_moves:
            best_is_winning = any(
                str(best_move) == str(wm) for wm in winning_moves
            )
            if not best_is_winning:
                win_strs = ", ".join(str(wm) for wm in winning_moves)
                print(f"       *** MISSED WIN *** "
                      f"Winning move(s): {win_strs}", flush=True)
                if is_white:
                    missed_w += 1
                else:
                    missed_b += 1

        game.make_move(best_move)

    # Result
    if game.winner == Color.WHITE:
        print(f"  -> White ({white_label}) wins in {moves} moves", flush=True)
        result = 1
    elif game.winner == Color.BLACK:
        print(f"  -> Black ({black_label}) wins in {moves} moves", flush=True)
        result = -1
    else:
        print(f"  -> Draw ({moves} moves)", flush=True)
        result = 0

    return result, moves, missed_w, missed_b


def load_model(path, device="cpu"):
    model, meta = AlphaZeroNet.load_checkpoint(path)
    evaluator = AlphaZeroEvaluator(model, device=device)
    iteration = meta.get("iteration", "?")
    return evaluator, iteration


def main():
    parser = argparse.ArgumentParser(description="Win-taking test")
    parser.add_argument("--m1", required=True, help="Model 1 checkpoint path")
    parser.add_argument("--m2", required=True, help="Model 2 checkpoint path")
    parser.add_argument("--sims", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")
    parser.add_argument("--device", default="cpu",
                        help="Device for inference (default: cpu)")
    args = parser.parse_args()

    print("Loading models...")
    eval1, iter1 = load_model(args.m1, args.device)
    eval2, iter2 = load_model(args.m2, args.device)
    label1 = f"iter {iter1}"
    label2 = f"iter {iter2}"
    print(f"  M1: {label1} ({os.path.basename(args.m1)})")
    print(f"  M2: {label2} ({os.path.basename(args.m2)})")
    print(f"  Sims: {args.sims}")
    print(f"  Device: {args.device}")
    print()

    total_missed_m1 = 0
    total_missed_m2 = 0

    # Game 1: M1 as white, M2 as black
    print(f"{'='*60}")
    print(f"  Game 1: {label1} (White) vs {label2} (Black)")
    print(f"{'='*60}")
    r1, moves1, mw1, mb1 = play_game(eval1, eval2, args.sims, label1, label2)
    missed_m1_g1 = mw1  # M1 was white
    missed_m2_g1 = mb1  # M2 was black
    total_missed_m1 += missed_m1_g1
    total_missed_m2 += missed_m2_g1
    print(f"  Missed wins: {label1}={missed_m1_g1}, {label2}={missed_m2_g1}")
    print()

    # Game 2: M2 as white, M1 as black
    print(f"{'='*60}")
    print(f"  Game 2: {label2} (White) vs {label1} (Black)")
    print(f"{'='*60}")
    r2, moves2, mw2, mb2 = play_game(eval2, eval1, args.sims, label2, label1)
    missed_m2_g2 = mw2  # M2 was white
    missed_m1_g2 = mb2  # M1 was black
    total_missed_m1 += missed_m1_g2
    total_missed_m2 += missed_m2_g2
    print(f"  Missed wins: {label2}={missed_m2_g2}, {label1}={missed_m1_g2}")
    print()

    # Score from M1's perspective
    m1_score = 0
    m2_score = 0
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

    # Summary
    print(f"{'='*60}")
    print(f"  SUMMARY ({args.sims} sims)")
    print(f"{'='*60}")
    print(f"  Score: {label1} {m1_score}-{m2_score} {label2}")
    print(f"  Game 1: {label1}(W) vs {label2}(B) — "
          f"{'White wins' if r1==1 else 'Black wins' if r1==-1 else 'Draw'} "
          f"in {moves1} moves")
    print(f"  Game 2: {label2}(W) vs {label1}(B) — "
          f"{'White wins' if r2==1 else 'Black wins' if r2==-1 else 'Draw'} "
          f"in {moves2} moves")
    print()
    print(f"  Missed instant wins:")
    print(f"    {label1}: {total_missed_m1} total "
          f"({missed_m1_g1} as W, {missed_m1_g2} as B)")
    print(f"    {label2}: {total_missed_m2} total "
          f"({missed_m2_g1} as B, {missed_m2_g2} as W)")


if __name__ == "__main__":
    main()
