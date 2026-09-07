"""Stress-test the engine.descend race fix (C3).

Loads a small checkpoint on CPU, lets the engine ponder a starting
position, then rapidly descends + polls in a loop. Any time
get_current_result() returns a non-None move, we assert it is legal
in the CURRENT game position. A stale result (from before the last
descend) would have a move belonging to the opposite side and fail.

Runs 30 rounds. Each round: pick a random legal move, descend into it,
immediately poll get_current_result, check the returned move for
legality. Also polls repeatedly for ~0.5s per round to exercise the
race window while the worker is processing.

Uses the smallest checkpoint (models/az_iter30.pt) for fast startup.
CPU-only, no GPU needed.

Run:
    python verify_engine_race.py

Pass criteria: 0 stale-move errors across all rounds.
Failure symptom: assertion "returned move X is not legal in current
position" — that's the exact bug this test is designed to catch.
"""
import os, random, sys, time
sys.path.insert(0, 'src')

from extinction_chess import ExtinctionChess
from tools.play_timed.engine import Engine

CHECKPOINT = os.path.join('models', 'az_iter30.pt')
N_ROUNDS = 30
PONDER_MS_BEFORE_DESCEND = 200    # let the tree grow first
POLL_MS_AFTER_DESCEND = 500       # window where the race could fire


def move_is_legal(move, legal_moves):
    for m in legal_moves:
        if (m.from_pos.rank == move.from_pos.rank
                and m.from_pos.file == move.from_pos.file
                and m.to_pos.rank == move.to_pos.rank
                and m.to_pos.file == move.to_pos.file
                and m.promotion == move.promotion):
            return True
    return False


def move_str(m):
    ff = chr(ord('a') + m.from_pos.file) + str(m.from_pos.rank + 1)
    tt = chr(ord('a') + m.to_pos.file) + str(m.to_pos.rank + 1)
    return f"{ff}-{tt}"


def main():
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: {CHECKPOINT} not found. Adjust CHECKPOINT in the script.")
        sys.exit(1)

    random.seed(1337)
    print(f"Loading {CHECKPOINT} on CPU (~5-15s)...")
    engine = Engine(CHECKPOINT, device="cpu")
    game = ExtinctionChess()
    engine.warmup(game, sample_sims=10)
    engine.start_from(game)
    print(f"Loaded iter {engine.iteration}, {engine.sims_per_second:.1f} sims/sec")
    print(f"Running {N_ROUNDS} descend-then-poll rounds...\n")

    stale_count = 0
    total_polls = 0
    for r in range(N_ROUNDS):
        # Let engine ponder for a bit so the tree has depth to reuse.
        time.sleep(PONDER_MS_BEFORE_DESCEND / 1000.0)

        legal = game.get_legal_moves()
        if not legal or game.game_over:
            print(f"round {r}: game ended, restarting")
            game = ExtinctionChess()
            engine.stop()
            engine.start_from(game)
            continue

        chosen = random.choice(legal)
        game.make_move(chosen)
        engine.descend(chosen)

        # Poll rapidly during the race window
        deadline = time.monotonic() + POLL_MS_AFTER_DESCEND / 1000.0
        current_legal = game.get_legal_moves()
        stale_this_round = 0
        polls_this_round = 0
        while time.monotonic() < deadline:
            result = engine.get_current_result()
            polls_this_round += 1
            if result is None or result.get("move") is None:
                continue
            m = result["move"]
            if not move_is_legal(m, current_legal):
                stale_this_round += 1
                if stale_this_round == 1:
                    print(f"  STALE at round {r}: engine returned "
                          f"{move_str(m)}, not legal for "
                          f"{'W' if game.current_player.value else 'B'} to move")

        stale_count += stale_this_round
        total_polls += polls_this_round
        status = f"{stale_this_round} stale" if stale_this_round else "OK"
        print(f"round {r:2d}: descended {move_str(chosen)}, "
              f"{status} ({polls_this_round} polls)")

    engine.shutdown()
    print(f"\n{stale_count} stale-move events across {total_polls} polls "
          f"in {N_ROUNDS} rounds.")
    if stale_count == 0:
        print("PASS: C3 fix is holding — no stale moves observed.")
        sys.exit(0)
    else:
        print(f"FAIL: {stale_count} stale-move events. C3 race is NOT closed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
