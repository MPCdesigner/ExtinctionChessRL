"""
Round-robin tournament between AlphaZero checkpoints.

Each pair plays one match (2 games: each side as white) per sim setting.
With temperature=0, games are deterministic so one match per setting suffices.

Usage:
    python tournament.py
    python tournament.py --models az_iter100.pt az_iter120.pt az_iter130.pt
    python tournament.py --sims 50 100 200
"""
import argparse
import glob
import os
import sys
import time
from copy import deepcopy
from itertools import combinations

from extinction_chess import ExtinctionChess, Color
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MAX_MOVES = 300


def load_model(path, device="cpu"):
    model, meta = AlphaZeroNet.load_checkpoint(path)
    evaluator = AlphaZeroEvaluator(model, device=device)
    iteration = meta.get("iteration", "?")
    return evaluator, iteration


def play_game(white_eval, black_eval, sims, white_label="W", black_label="B"):
    """Play a single game. Returns +1 if white wins, -1 if black wins, 0 for draw."""
    game = ExtinctionChess()
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
        print(f"      {moves:>3}. {side} ({label}) {best}", flush=True)
        game.make_move(best)

    if game.winner == Color.WHITE:
        print(f"      -> White ({white_label}) wins in {moves} moves", flush=True)
        return 1, moves
    elif game.winner == Color.BLACK:
        print(f"      -> Black ({black_label}) wins in {moves} moves", flush=True)
        return -1, moves
    print(f"      -> Draw ({moves} moves)", flush=True)
    return 0, moves


def play_match(eval_a, eval_b, sims, label_a="A", label_b="B"):
    """Play a match: A as white then B as white. Returns (a_score, b_score, details)."""
    # Game 1: A is white
    print(f"    Game 1: {label_a}(W) vs {label_b}(B)", flush=True)
    r1, m1 = play_game(eval_a, eval_b, sims, white_label=label_a, black_label=label_b)
    # Game 2: B is white
    print(f"    Game 2: {label_b}(W) vs {label_a}(B)", flush=True)
    r2, m2 = play_game(eval_b, eval_a, sims, white_label=label_b, black_label=label_a)

    a_score = 0
    b_score = 0
    details = []

    # Game 1: A white vs B black
    if r1 == 1:
        a_score += 1
        details.append(f"G1: A(W) wins in {m1}")
    elif r1 == -1:
        b_score += 1
        details.append(f"G1: B(B) wins in {m1}")
    else:
        a_score += 0.5
        b_score += 0.5
        details.append(f"G1: draw ({m1} moves)")

    # Game 2: B white vs A black
    if r2 == 1:
        b_score += 1
        details.append(f"G2: B(W) wins in {m2}")
    elif r2 == -1:
        a_score += 1
        details.append(f"G2: A(B) wins in {m2}")
    else:
        a_score += 0.5
        b_score += 0.5
        details.append(f"G2: draw ({m2} moves)")

    return a_score, b_score, details


def main():
    parser = argparse.ArgumentParser(description="Round-robin tournament")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model filenames (in models/ dir). Default: all .pt files")
    parser.add_argument("--sims", nargs="+", type=int, default=[20, 50, 100, 200, 400],
                        help="Sim settings to test (default: 20 50 100 200 400)")
    parser.add_argument("--device", default="cpu", help="Device for inference (default: cpu)")
    args = parser.parse_args()

    # Discover models
    if args.models:
        model_paths = [os.path.join(MODELS_DIR, m) for m in args.models]
    else:
        model_paths = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))

    if len(model_paths) < 2:
        print("Need at least 2 models for a tournament.")
        return

    # Load all models
    print("Loading models...")
    models = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        evaluator, iteration = load_model(path, args.device)
        label = f"iter {iteration}"
        models.append({"eval": evaluator, "label": label, "path": path})
        print(f"  Loaded {label} ({os.path.basename(path)})")

    if len(models) < 2:
        print("Need at least 2 valid models.")
        return

    pairs = list(combinations(range(len(models)), 2))
    total_games = len(pairs) * len(args.sims) * 2
    print(f"\n{len(models)} models, {len(pairs)} pairs, {len(args.sims)} sim settings")
    print(f"Total games: {total_games}\n")

    n = len(models)
    # Track results per sim setting and overall
    # h2h[sims][(i,j)] = score of i against j in that sim setting (out of 2)
    overall_scores = {i: 0.0 for i in range(n)}
    overall_h2h = {}  # (i,j) -> total score of i vs j across all sims
    game_count = 0

    for sims in args.sims:
        print(f"\n{'='*60}")
        print(f"  SIM SETTING: {sims}")
        print(f"{'='*60}")

        sim_scores = {i: 0.0 for i in range(n)}
        sim_h2h = {}  # (i,j) -> score of i vs j

        for a_idx, b_idx in pairs:
            a_label = models[a_idx]["label"]
            b_label = models[b_idx]["label"]
            game_count += 2
            print(f"\n  [{game_count}/{total_games}] {a_label} vs {b_label} ({sims} sims)...",
                  flush=True)

            t0 = time.time()
            a_score, b_score, details = play_match(
                models[a_idx]["eval"], models[b_idx]["eval"], sims,
                label_a=a_label, label_b=b_label
            )
            elapsed = time.time() - t0

            sim_scores[a_idx] += a_score
            sim_scores[b_idx] += b_score
            overall_scores[a_idx] += a_score
            overall_scores[b_idx] += b_score

            sim_h2h[(a_idx, b_idx)] = a_score
            sim_h2h[(b_idx, a_idx)] = b_score
            overall_h2h[(a_idx, b_idx)] = overall_h2h.get((a_idx, b_idx), 0) + a_score
            overall_h2h[(b_idx, a_idx)] = overall_h2h.get((b_idx, a_idx), 0) + b_score

            result_str = f"{a_score}-{b_score}"
            print(f"{result_str}  ({elapsed:.1f}s)")
            for d in details:
                print(f"    {d}")

        # Print standings + head-to-head matrix for this sim setting
        ranking = sorted(range(n), key=lambda i: sim_scores[i], reverse=True)
        max_sim = (n - 1) * 2  # max possible score per sim (2 points per opponent)
        print(f"\n  Standings ({sims} sims):")
        print(f"  {'Model':<12} {'Score':>6} / {max_sim}")
        print(f"  {'-'*25}")
        for i in ranking:
            print(f"  {models[i]['label']:<12} {sim_scores[i]:>6.1f}")

        _print_h2h_matrix(models, ranking, sim_h2h)

    # Overall standings
    print(f"\n{'='*60}")
    print(f"  OVERALL STANDINGS (all sim settings combined)")
    print(f"{'='*60}")
    max_score = (n - 1) * 2 * len(args.sims)
    ranking = sorted(range(n), key=lambda i: overall_scores[i], reverse=True)
    print(f"  {'Model':<12} {'Score':>7} / {max_score}  {'Win%':>6}")
    print(f"  {'-'*35}")
    for i in ranking:
        pct = overall_scores[i] / max_score * 100 if max_score > 0 else 0
        print(f"  {models[i]['label']:<12} {overall_scores[i]:>7.1f}  {pct:>6.1f}%")

    print(f"\n  Head-to-head (all sims combined, each cell out of {len(args.sims) * 2}):")
    _print_h2h_matrix(models, ranking, overall_h2h)


def _print_h2h_matrix(models, ranking, h2h):
    """Print a head-to-head results matrix."""
    n = len(models)
    # Column width adapts to label length
    labels = [models[i]["label"] for i in ranking]
    cw = max(len(l) for l in labels) + 2  # column width

    # Header row
    header = " " * (cw + 2)
    for j in ranking:
        header += f"{models[j]['label']:>{cw}}"
    print(f"\n{header}")
    print(f"  {'-' * (cw + cw * n)}")

    for i in ranking:
        row = f"  {models[i]['label']:<{cw}}"
        for j in ranking:
            if i == j:
                row += f"{'—':>{cw}}"
            else:
                score = h2h.get((i, j), 0)
                row += f"{score:>{cw}.1f}"
        print(row)


if __name__ == "__main__":
    main()
