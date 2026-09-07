"""Verify Board.copy() still behaves correctly after the __new__ fix.

Plays random games, checks at every ply that:
  1. All 5 attributes on the copy match the original
  2. Mutating the copy doesn't affect the original
  3. position_key(current_player) matches between copy and original

Run:
    python verify_board_copy.py
"""
import random, sys
sys.path.insert(0, 'src')
from extinction_chess import ExtinctionChess

random.seed(42)
BOARD_ATTRS = ('en_passant_target', 'halfmove_clock',
               'fullmove_number', 'position_history')


def grids_match(g1, g2):
    for r in range(8):
        for f in range(8):
            p1, p2 = g1[r][f], g2[r][f]
            if (p1 is None) != (p2 is None):
                return False
            if p1 is not None and (p1.piece_type != p2.piece_type
                                    or p1.color != p2.color
                                    or p1.has_moved != p2.has_moved):
                return False
    return True


def check_copy(board, current_player, ply):
    c = board.copy()
    # 1. Attribute equality on the fresh copy
    assert grids_match(board.grid, c.grid), f"ply {ply}: grid mismatch"
    for a in BOARD_ATTRS:
        assert getattr(board, a) == getattr(c, a), \
            f"ply {ply}: {a} mismatch (orig={getattr(board, a)!r} "\
            f"copy={getattr(c, a)!r})"

    # 2. Independence — mutations on copy must not leak into original
    old_ep = board.en_passant_target
    old_hc = board.halfmove_clock
    old_hist_len = len(board.position_history)
    c.en_passant_target = "MUTATED"
    c.halfmove_clock = 999
    c.position_history.append("SENTINEL")
    # grid: clear a corner square and confirm original still has whatever
    orig_corner = board.grid[0][0]
    c.grid[0][0] = None
    assert board.en_passant_target == old_ep, f"ply {ply}: en_passant leaked"
    assert board.halfmove_clock == old_hc, f"ply {ply}: halfmove leaked"
    assert len(board.position_history) == old_hist_len, \
        f"ply {ply}: position_history leaked (len {len(board.position_history)} vs {old_hist_len})"
    assert board.grid[0][0] is orig_corner, f"ply {ply}: grid leaked"

    # 3. Fresh copy's position_key must match original
    c2 = board.copy()
    k_orig = board.get_position_key(current_player)
    k_copy = c2.get_position_key(current_player)
    assert k_orig == k_copy, \
        f"ply {ply}: position_key mismatch\n  orig={k_orig}\n  copy={k_copy}"


def main():
    n_games = 5
    n_plies_target = 60
    for gi in range(n_games):
        g = ExtinctionChess()
        last_ply = 0
        for ply in range(n_plies_target):
            check_copy(g.board, g.current_player, ply)
            moves = g.get_legal_moves()
            if not moves or g.game_over:
                break
            g.make_move(random.choice(moves))
            last_ply = ply
        print(f"game {gi}: OK through ply {last_ply + 1}"
              + (" (game ended early)" if g.game_over or not moves else ""))
    print(f"\nAll {n_games} games passed all checks. C1 fix looks correct.")


if __name__ == "__main__":
    main()
