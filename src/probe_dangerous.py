"""
Probe: generate positions where at least one legal move is a blunder
(opponent has an instant winning response), then run MCTS at varying
sim counts to see how the model evaluates them.

Goal: check whether higher sim counts actually identify the threats
and concentrate visits on safe moves. If yes, we have a basis for
deeper-search defensive drilling.

Usage:
    python probe_dangerous.py --model az_iter_400_100pct.pt --positions 10 \
        --sims 800 3200
"""

import argparse
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from extinction_chess import ExtinctionChess, Position, Color, PieceType
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def copy_game(game):
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


def _move_key(m):
    """Hashable identifier for a move (works across separate get_legal_moves calls)."""
    promo = getattr(m, 'promotion', None)
    return (m.from_pos.rank, m.from_pos.file, m.to_pos.rank, m.to_pos.file, promo)


def find_blunder_keys(game):
    """Return set of move keys that allow opponent to win in 1 ply."""
    current = game.current_player
    blunder_keys = set()
    for m in game.get_legal_moves():
        gc = copy_game(game)
        if not gc.make_move(m):
            continue
        if gc.game_over:
            continue
        for om in gc.get_legal_moves():
            gc2 = copy_game(gc)
            if gc2.make_move(om) and gc2.game_over and gc2.winner != current:
                blunder_keys.add(_move_key(m))
                break
    return blunder_keys


def board_to_string(game):
    """Render board as text rows (rank 8 at top)."""
    piece_char = {
        PieceType.KING: 'K', PieceType.QUEEN: 'Q', PieceType.ROOK: 'R',
        PieceType.BISHOP: 'B', PieceType.KNIGHT: 'N', PieceType.PAWN: 'P',
    }
    lines = ["    a b c d e f g h"]
    for rank in range(7, -1, -1):
        row = ""
        for file in range(8):
            piece = game.board.get_piece(Position(rank, file))
            if piece:
                c = piece_char[piece.piece_type]
                if piece.color == Color.BLACK:
                    c = c.lower()
                row += c + " "
            else:
                row += ". "
        lines.append(f"  {rank+1} {row.rstrip()} {rank+1}")
    lines.append("    a b c d e f g h")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model filename or path")
    parser.add_argument("--positions", type=int, default=10,
                        help="Number of dangerous positions to test (default 10)")
    parser.add_argument("--sims", nargs="+", type=int, default=[800, 3200],
                        help="Sim counts to test at (default: 800 3200)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-random-moves", type=int, default=200,
                        help="Max random moves per game to find a position")
    parser.add_argument("--min-blunder-ratio", type=float, default=0.7,
                        help="Minimum fraction of legal moves that must be blunders "
                             "(default 0.7 = position must be 'under threat')")
    parser.add_argument("--min-root-value", type=float, default=-0.5,
                        help="Skip positions where root_value at highest sim < this "
                             "(default -0.5 = filter out hopeless positions)")
    args = parser.parse_args()

    # Resolve model path
    if os.path.isfile(args.model):
        path = args.model
    else:
        path = os.path.join(MODELS_DIR, args.model)
        if not os.path.isfile(path):
            print(f"ERROR: Model not found: {args.model}")
            sys.exit(1)

    # Load model
    model, meta = AlphaZeroNet.load_checkpoint(path)
    evaluator = AlphaZeroEvaluator(model, device=args.device)
    label = f"iter {meta.get('iteration', '?')}"
    print(f"Loaded {os.path.basename(path)} ({label}) on {args.device}\n")

    positions_found = 0
    game = ExtinctionChess()

    while positions_found < args.positions:
        if game.game_over:
            game = ExtinctionChess()

        legal = game.get_legal_moves()
        if not legal:
            game = ExtinctionChess()
            continue

        blunder_keys = find_blunder_keys(game)
        blunders = [m for m in legal if _move_key(m) in blunder_keys]
        safe = [m for m in legal if _move_key(m) not in blunder_keys]
        # Want positions "under threat": most moves blunder, but at least one safe move exists
        blunder_ratio = len(blunders) / len(legal) if legal else 0
        is_under_threat = (
            blunders
            and safe
            and blunder_ratio >= args.min_blunder_ratio
        )
        if is_under_threat:
            positions_found += 1
            current = game.current_player
            side = "White" if current == Color.WHITE else "Black"

            print(f"\n{'='*70}")
            print(f"Position {positions_found}/{args.positions} | {side} to move")
            print(f"{'='*70}")
            print(board_to_string(game))
            print(f"\nLegal moves: {len(legal)} | Blunders: {len(blunders)} ({blunder_ratio*100:.0f}%) | Safe: {len(safe)}")
            print(f"Safe moves (model must find these): {', '.join(str(s) for s in safe)}")

            for sims in args.sims:
                game_copy = copy_game(game)
                mv, root_value = mcts_search(
                    game_copy, evaluator,
                    num_simulations=sims,
                    dirichlet_alpha=0, noise_weight=0,
                    tactical_shortcuts=False,
                )
                if not mv:
                    print(f"\n  @{sims} sims: no moves returned")
                    continue

                # Sort by visits, top 5
                mv_sorted = sorted(mv, key=lambda x: x[1], reverse=True)
                total_visits = sum(v for _, v in mv)
                blunder_visits = sum(v for m, v in mv if _move_key(m) in blunder_keys)
                blunder_pct = (blunder_visits / total_visits * 100) if total_visits else 0

                print(f"\n  @{sims} sims | root_value={root_value:+.3f} | "
                      f"blunder share={blunder_pct:.1f}%")
                for m, v in mv_sorted[:5]:
                    pct = (v / total_visits * 100) if total_visits else 0
                    tag = " BLUNDER" if _move_key(m) in blunder_keys else ""
                    print(f"    {m}: {v} visits ({pct:.1f}%){tag}")

            # Apply random move and continue
            game.make_move(random.choice(legal))
        else:
            # Not a useful position - random move
            move = random.choice(legal)
            game.make_move(move)


if __name__ == "__main__":
    main()
