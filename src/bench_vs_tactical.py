"""
Headless benchmark: model vs advanced tactical random opponent.

The tactical random opponent always takes instant wins and never blunders
into instant losses (unless forced). This tests whether the model can win
through strategic play, not just by exploiting blunders.

Usage:
    python3 bench_vs_tactical.py --model az_iter_320_100pct.pt --device cuda
    python3 bench_vs_tactical.py --model az_iter_320_100pct.pt --sims 50 100 200 --games 20
"""
import argparse
import os
import random
import time

import torch
import numpy as np

from extinction_chess import ExtinctionChess, Color, Move
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
SIM_OPTIONS = [20, 50, 100, 200, 400]
MAX_MOVES = 300


def _copy_game(game):
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


def tactical_random_move(game):
    """Advanced tactical random: takes instant wins, avoids instant losses."""
    legal_moves = game.get_legal_moves()
    if not legal_moves:
        return None

    current = game.current_player

    # Check for instant wins
    winning_moves = []
    for move in legal_moves:
        gc = _copy_game(game)
        gc.make_move(move)
        if gc.game_over and gc.winner == current:
            winning_moves.append(move)
    if winning_moves:
        return random.choice(winning_moves)

    # Filter out moves that give the opponent an instant win
    safe_moves = []
    for move in legal_moves:
        gc = _copy_game(game)
        gc.make_move(move)
        if gc.game_over:
            # This move ends the game but isn't a win for us
            continue
        opponent_has_win = False
        for opp_move in gc.get_legal_moves():
            gc2 = _copy_game(gc)
            gc2.make_move(opp_move)
            if gc2.game_over and gc2.winner != current:
                opponent_has_win = True
                break
        if not opponent_has_win:
            safe_moves.append(move)
    if safe_moves:
        return random.choice(safe_moves)

    # All moves lose — pick any
    return random.choice(legal_moves)


def play_game(evaluator, sims, model_is_white, label="model"):
    """Play one game. Returns +1 model wins, -1 model loses, 0 draw."""
    game = ExtinctionChess()
    moves = 0
    while not game.game_over and moves < MAX_MOVES:
        is_model_turn = (game.current_player == Color.WHITE) == model_is_white
        side = "W" if game.current_player == Color.WHITE else "B"

        if is_model_turn:
            mv, _ = mcts_search(
                game, evaluator,
                num_simulations=sims,
                dirichlet_alpha=0, noise_weight=0,
                tactical_shortcuts=False,
            )
            if not mv:
                break
            best = max(mv, key=lambda x: x[1])[0]
            who = label
        else:
            best = tactical_random_move(game)
            if best is None:
                break
            who = "tactical"

        moves += 1
        print(f"    {moves:>3}. {side} ({who}) {best}", flush=True)
        game.make_move(best)

    # Determine result from model's perspective
    if game.winner is not None:
        model_color = Color.WHITE if model_is_white else Color.BLACK
        if game.winner == model_color:
            side_str = "White" if model_is_white else "Black"
            print(f"    -> Model ({side_str}) wins in {moves} moves", flush=True)
            return 1, moves
        else:
            side_str = "White" if model_is_white else "Black"
            print(f"    -> Model ({side_str}) loses in {moves} moves", flush=True)
            return -1, moves
    print(f"    -> Draw ({moves} moves)", flush=True)
    return 0, moves


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model vs advanced tactical random")
    parser.add_argument("--model", required=True, help="Model filename (in models/ dir)")
    parser.add_argument("--sims", nargs="+", type=int, default=SIM_OPTIONS,
                        help=f"Sim settings (default: {SIM_OPTIONS})")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per sim setting (half as white, half as black; default: 20)")
    parser.add_argument("--device", default="cpu", help="Device (default: cpu)")
    args = parser.parse_args()

    path = os.path.join(MODELS_DIR, args.model)
    print("Loading model...")
    model, meta = AlphaZeroNet.load_checkpoint(path)
    evaluator = AlphaZeroEvaluator(model, device=args.device)
    iteration = meta.get("iteration", "?")
    label = f"iter {iteration}"
    print(f"  Model: {label} ({args.model})")
    print(f"  Opponent: advanced tactical random (win-taking + loss avoidance)")
    print(f"  Sims: {args.sims}")
    print(f"  Games per sim: {args.games} ({args.games // 2} as W, {args.games // 2} as B)")
    total_games = len(args.sims) * args.games
    print(f"  Total games: {total_games}\n")

    games_as_white = args.games // 2
    games_as_black = args.games - games_as_white

    all_results = {}
    grand_wins, grand_losses, grand_draws = 0, 0, 0
    t_start = time.time()

    for sims in args.sims:
        print(f"{'='*60}")
        print(f"  Sims: {sims}")
        print(f"{'='*60}")
        wins, losses, draws = 0, 0, 0
        total_moves = 0

        # Model as white
        for g in range(games_as_white):
            print(f"\n  Game {g+1}/{games_as_white} (model=White, sims={sims})")
            result, move_count = play_game(evaluator, sims, model_is_white=True, label=label)
            total_moves += move_count
            if result > 0: wins += 1
            elif result < 0: losses += 1
            else: draws += 1

        # Model as black
        for g in range(games_as_black):
            print(f"\n  Game {g+1}/{games_as_black} (model=Black, sims={sims})")
            result, move_count = play_game(evaluator, sims, model_is_white=False, label=label)
            total_moves += move_count
            if result > 0: wins += 1
            elif result < 0: losses += 1
            else: draws += 1

        total = wins + losses + draws
        win_rate = wins / total * 100 if total > 0 else 0
        avg_moves = total_moves / total if total > 0 else 0
        all_results[sims] = {"wins": wins, "losses": losses, "draws": draws,
                             "win_rate": win_rate, "avg_moves": avg_moves}
        grand_wins += wins
        grand_losses += losses
        grand_draws += draws

        print(f"\n  Sims {sims}: W={wins} L={losses} D={draws} "
              f"({win_rate:.1f}% win rate, avg {avg_moves:.0f} moves)")

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {label} vs advanced tactical random")
    print(f"{'='*60}")
    print(f"  {'Sims':>6}  {'W':>3}  {'L':>3}  {'D':>3}  {'Win%':>6}  {'Avg Moves':>9}")
    print(f"  {'-'*40}")
    for sims in args.sims:
        r = all_results[sims]
        print(f"  {sims:>6}  {r['wins']:>3}  {r['losses']:>3}  {r['draws']:>3}  "
              f"{r['win_rate']:>5.1f}%  {r['avg_moves']:>9.0f}")
    grand_total = grand_wins + grand_losses + grand_draws
    grand_rate = grand_wins / grand_total * 100 if grand_total > 0 else 0
    print(f"  {'-'*40}")
    print(f"  {'Total':>6}  {grand_wins:>3}  {grand_losses:>3}  {grand_draws:>3}  "
          f"{grand_rate:>5.1f}%")
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
