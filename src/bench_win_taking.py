"""
Headless win-taking benchmark: generates random positions where an instant win
exists, then tests whether each model (at various sim counts) finds a winning capture.

Supports the same filters as the GUI version (direction, distance, piece types).

Usage:
    python bench_win_taking.py --models az_iter360.pt az_iter350.pt \
        --sims 20 50 100 200 400 --positions 100 \
        --directions backward sideways --min-distance 4
"""

import argparse
import os
import sys
import random
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from extinction_chess import ExtinctionChess, Position, Color, PieceType, Piece
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


# ═══════════════════════════════════════════════════════════════════════════
# Game helpers
# ═══════════════════════════════════════════════════════════════════════════

def copy_game(game):
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


def capture_direction(move, player_color):
    dr = move.to_pos.rank - move.from_pos.rank
    if player_color == Color.BLACK:
        dr = -dr
    if dr > 0:
        return 'forward'
    elif dr < 0:
        return 'backward'
    else:
        return 'sideways'


def capture_distance(move):
    dr = abs(move.to_pos.rank - move.from_pos.rank)
    df = abs(move.to_pos.file - move.from_pos.file)
    return max(dr, df)


def find_winning_moves(game, target_piece_types=None):
    current = game.current_player
    opponent = Color.BLACK if current == Color.WHITE else Color.WHITE
    winners = []
    for m in game.get_legal_moves():
        gc = copy_game(game)
        captured_piece = game.board.get_piece(m.to_pos)
        if gc.make_move(m) and gc.game_over and gc.winner == current:
            captured_type = None
            if captured_piece and captured_piece.color == opponent:
                captured_type = captured_piece.piece_type
            if target_piece_types is None or captured_type in target_piece_types:
                winners.append((m, captured_type))
    return winners


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def run_test(models, sim_counts, num_positions, target_piece_types=None,
             target_directions=None, min_capture_distance=1, hard_only=False,
             verbose=True):
    """Run headless win-taking test. Returns aggregate results dict."""

    # Aggregate: {model_label: {sims: {"hits": N, "misses": N}}}
    aggregate = {}
    for label, _ in models:
        aggregate[label] = {}
        for s in sim_counts:
            aggregate[label][s] = {"hits": 0, "misses": 0}

    # Per-position results: list of dicts with board, side, winners, and per-model hits
    per_position = []

    positions_tested = 0
    games_played = 0
    game = ExtinctionChess()

    t0 = time.time()

    while positions_tested < num_positions:
        # Generate a position with qualifying winning moves
        if game.game_over:
            game = ExtinctionChess()
            games_played += 1

        legal = game.get_legal_moves()
        if not legal:
            game = ExtinctionChess()
            games_played += 1
            continue

        current = game.current_player
        all_winners = find_winning_moves(game)

        def passes_filters(m, pt):
            if hard_only:
                # A move is "hard" if it's strictly backward OR distance >= min
                # A move FAILS (is easy) if it's (forward or sideways) AND short-range
                d = capture_direction(m, current)
                dist = capture_distance(m)
                if target_directions and d in target_directions:
                    return True  # direction match is always hard
                if min_capture_distance > 1 and dist >= min_capture_distance:
                    return True  # long-range is always hard
                return False
            # AND logic (original)
            if target_piece_types is not None and pt not in target_piece_types:
                return False
            if target_directions is not None:
                d = capture_direction(m, current)
                if d not in target_directions:
                    return False
            if min_capture_distance > 1:
                if capture_distance(m) < min_capture_distance:
                    return False
            return True

        winners = [(m, pt) for m, pt in all_winners if passes_filters(m, pt)]
        non_matching = [(m, pt) for m, pt in all_winners if not passes_filters(m, pt)]
        if non_matching:
            winners = []

        if winners:
            winning_moves = [m for m, _ in winners]
            positions_tested += 1

            side_str = "White" if current == Color.WHITE else "Black"
            win_info = [
                (str(m), pt.value if pt else '?',
                 capture_direction(m, current), capture_distance(m))
                for m, pt in winners
            ]

            if verbose:
                win_strs = ", ".join(
                    f"{wm}({pv},{wd[0]},d{wdist})"
                    for wm, pv, wd, wdist in win_info
                )
                print(f"\nPos {positions_tested}/{num_positions} | {side_str} | wins: {win_strs}")

            # Snapshot board state as text rows (rank 8 down to rank 1)
            piece_char = {
                PieceType.KING: 'K', PieceType.QUEEN: 'Q', PieceType.ROOK: 'R',
                PieceType.BISHOP: 'B', PieceType.KNIGHT: 'N', PieceType.PAWN: 'P',
            }
            board_rows = []
            for rank in range(7, -1, -1):
                row = ""
                for file in range(8):
                    piece = game.board.get_piece(Position(rank, file))
                    if piece:
                        c = piece_char[piece.piece_type]
                        if piece.color == Color.BLACK:
                            c = c.lower()
                        row += c
                    else:
                        row += "."
                board_rows.append(row)

            pos_record = {
                "index": positions_tested,
                "side": side_str,
                "board_rows": board_rows,
                "winners": win_info,
                "hits": {label: {} for label, _ in models},
            }

            any_hit = False
            for label, evaluator in models:
                for sims in sim_counts:
                    game_copy = copy_game(game)
                    mv, root_val = mcts_search(
                        game_copy, evaluator,
                        num_simulations=sims,
                        dirichlet_alpha=0, noise_weight=0,
                        tactical_shortcuts=False,
                    )

                    if mv:
                        mv_sorted = sorted(mv, key=lambda x: x[1], reverse=True)
                        best_move = mv_sorted[0][0]
                        is_hit = any(str(best_move) == str(wm) for wm in winning_moves)

                        if is_hit:
                            aggregate[label][sims]["hits"] += 1
                            any_hit = True
                            marker = "HIT"
                        else:
                            aggregate[label][sims]["misses"] += 1
                            marker = "MISS"

                        pos_record["hits"][label][sims] = is_hit

                        if verbose:
                            top3 = mv_sorted[:3]
                            top3_str = ", ".join(f"{m}:{v}" for m, v in top3)
                            print(f"  {label} @{sims}: {best_move} [{top3_str}] — {marker}")
                    else:
                        aggregate[label][sims]["misses"] += 1
                        pos_record["hits"][label][sims] = False
                        if verbose:
                            print(f"  {label} @{sims}: no moves — MISS")

            per_position.append(pos_record)

            # Start fresh game if any model found the win
            if any_hit:
                game = ExtinctionChess()
                games_played += 1
            else:
                legal = game.get_legal_moves()
                if legal:
                    game.make_move(random.choice(legal))
        else:
            # No qualifying position yet — random move
            move = random.choice(legal)
            game.make_move(move)

    elapsed = time.time() - t0
    return aggregate, positions_tested, games_played, elapsed, per_position


def print_tough_positions(per_position, models, sim_counts):
    """Print positions where at least half the models completely failed
    (missed at every sim count)."""
    num_models = len(models)
    threshold = (num_models + 1) // 2  # at least half (rounded up)

    tough = []
    for pos in per_position:
        complete_failures = []
        for label, _ in models:
            hits = pos["hits"].get(label, {})
            if hits and not any(hits.values()):
                complete_failures.append(label)
        if len(complete_failures) >= threshold:
            tough.append((pos, complete_failures))

    if not tough:
        print(f"\n{'='*60}")
        print(f"No positions where >= {threshold}/{num_models} models completely failed.")
        print(f"{'='*60}")
        return

    print(f"\n{'='*60}")
    print(f"Tough Positions ({len(tough)} where >= {threshold}/{num_models} models completely failed)")
    print(f"{'='*60}")

    for pos, failed in tough:
        win_strs = ", ".join(
            f"{wm}({pv},{wd[0]},d{wdist})"
            for wm, pv, wd, wdist in pos["winners"]
        )
        print(f"\n--- Pos {pos['index']} | {pos['side']} to move ---")
        print(f"Winning moves: {win_strs}")
        print(f"Failed completely: {', '.join(failed)}")
        # Board printing preserved below — uncomment if you want full diagrams
        # print()
        # print("    a b c d e f g h")
        # for i, row in enumerate(pos["board_rows"]):
        #     rank = 8 - i
        #     spaced = " ".join(row)
        #     print(f"  {rank} {spaced} {rank}")
        # print("    a b c d e f g h")


def print_summary(aggregate, sim_counts, models, positions_tested, games_played, elapsed):
    print(f"\n{'='*60}")
    print(f"Win-Taking Test Summary ({positions_tested} positions, {games_played} games, {elapsed:.1f}s)")
    print(f"{'='*60}")

    # Header
    header = f"{'Model':<14}"
    for s in sim_counts:
        header += f"{'@'+str(s):>10}"
    print(header)
    print("-" * (14 + 10 * len(sim_counts)))

    for label, _ in models:
        row = f"{label:<14}"
        for s in sim_counts:
            h = aggregate[label][s]["hits"]
            mi = aggregate[label][s]["misses"]
            total = h + mi
            if total > 0:
                pct = h / total * 100
                row += f"{h}/{total}({pct:.0f}%)".rjust(10)
            else:
                row += f"{'—':>10}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description="Headless win-taking benchmark")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model filenames (e.g. az_iter360.pt) or full paths")
    parser.add_argument("--sims", nargs="+", type=int, default=[20, 50, 100, 200, 400],
                        help="Sim counts to test (default: 20 50 100 200 400)")
    parser.add_argument("--positions", type=int, default=50,
                        help="Number of positions to test (default: 50)")
    parser.add_argument("--directions", nargs="*", default=None,
                        choices=["forward", "backward", "sideways"],
                        help="Filter by capture direction (default: all)")
    parser.add_argument("--min-distance", type=int, default=1,
                        help="Minimum capture distance (default: 1 = no filter)")
    parser.add_argument("--pieces", nargs="*", default=None,
                        choices=["K", "Q", "R", "B", "N", "P"],
                        help="Filter by captured piece type (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference (default: cuda if available)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save results to JSON file")
    parser.add_argument("--hard-only", action="store_true",
                        help="Only positions where all wins are hard (backward, sideways, or distance>=3)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary, not per-position details")
    args = parser.parse_args()

    # Resolve model paths
    model_paths = []
    for m in args.models:
        if os.path.isfile(m):
            model_paths.append(m)
        else:
            full = os.path.join(MODELS_DIR, m)
            if os.path.isfile(full):
                model_paths.append(full)
            else:
                print(f"ERROR: Model not found: {m} (tried {full})")
                sys.exit(1)

    # Load models
    models = []
    for path in model_paths:
        model, meta = AlphaZeroNet.load_checkpoint(path)
        evaluator = AlphaZeroEvaluator(model, device=args.device)
        iteration = meta.get("iteration", "?")
        label = f"iter {iteration}"
        models.append((label, evaluator))
        print(f"Loaded {os.path.basename(path)} ({label}) on {args.device}")

    # Parse piece type filter
    piece_map = {
        "K": PieceType.KING, "Q": PieceType.QUEEN, "R": PieceType.ROOK,
        "B": PieceType.BISHOP, "N": PieceType.KNIGHT, "P": PieceType.PAWN,
    }
    target_pieces = None
    if args.pieces:
        target_pieces = set(piece_map[p] for p in args.pieces)

    # Parse direction filter
    target_dirs = None
    if args.directions:
        target_dirs = set(args.directions)

    sim_counts = sorted(args.sims)

    print(f"\nConfig: {args.positions} positions, sims={sim_counts}")
    if args.hard_only:
        print(f"  Hard-only: yes (no forward short-range wins)")
    if target_dirs:
        print(f"  Directions: {target_dirs}")
    if target_pieces:
        print(f"  Piece types: {args.pieces}")
    if args.min_distance > 1:
        print(f"  Min distance: {args.min_distance}")
    print()

    # Run
    aggregate, positions_tested, games_played, elapsed, per_position = run_test(
        models, sim_counts, args.positions,
        target_piece_types=target_pieces,
        target_directions=target_dirs,
        min_capture_distance=args.min_distance,
        hard_only=args.hard_only,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary(aggregate, sim_counts, models, positions_tested, games_played, elapsed)

    # Print tough positions where >= half of models completely failed
    print_tough_positions(per_position, models, sim_counts)

    # Save JSON
    if args.save_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"win_taking_results_{timestamp}.json"
        output = {
            "timestamp": timestamp,
            "models": [os.path.basename(p) for p in model_paths],
            "sim_counts": sim_counts,
            "positions_tested": positions_tested,
            "games_played": games_played,
            "elapsed_seconds": elapsed,
            "filters": {
                "directions": list(target_dirs) if target_dirs else "all",
                "pieces": args.pieces if args.pieces else "all",
                "min_distance": args.min_distance,
            },
            "aggregate": {label: {str(s): v for s, v in sims.items()}
                          for label, sims in aggregate.items()},
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
