"""Build a value-drilling .npz from the local value_targets.json.

Reads `dataset/value_targets.json` (managed by the positional eval tool),
optionally filters by session or tag, reconstructs each position via
PositionState.from_dict — which REPLAYS move_history so history planes,
castling rights, and en passant are correct — encodes via StateEncoder,
and writes an .npz file in the exact format the cluster replay buffer
expects.

Output format (matches src/alphazero.py:1957-1962 self-play writes):
    boards:   uint8    shape (N, 115, 8, 8)
    policies: float32  shape (N, 4864)   ALL ZEROS (mask via cross-entropy)
    values:   float32  shape (N,)         values in {-1, 0, +1}

Zero policies work as a natural training mask: the cross-entropy loss
in alphazero.py:2109 is `-sum(target * log_softmax(pred))`, and when
target is all zeros the sample contributes zero to policy gradient.
Only the value MSE term hits. See commit b77d20a docstring for details.

Usage (from project root):
    python tools/build_value_drill_npz.py
    python tools/build_value_drill_npz.py -o dataset/handset_batch1.npz
    python tools/build_value_drill_npz.py --tags forced_win
    python tools/build_value_drill_npz.py --session 2026-08-05T22:03:00
    python tools/build_value_drill_npz.py --dry-run

After building, upload to cluster:
    scp -i C:\\Users\\henry\\.ssh\\id_ed25519 \\
        "c:/Users/henry/Desktop/Extinction Chess RL/dataset/iter_9999_valdrill.npz" \\
        h74liang@wato-login1.ext.watonomous.ca:~/extinction-chess/replay_buffer/

Filename `iter_9999_valdrill.npz` is chosen so the buffer regex accepts
it (`^iter_(\\d+)(?:_([A-Za-z0-9]+))?\\.npz$`) and the very high iter
number 9999 means it never triggers the K=5 eviction cleanup. The
`_valdrill` tag surfaces in the loader's per-source breakdown log.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

# Path setup so imports work regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_TOOL_PKG = os.path.join(_HERE, "positional_eval")
for p in (_SRC_DIR, _TOOL_PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

# From src/
from state_encoder import StateEncoder  # noqa: E402
from alphazero import POLICY_SIZE  # noqa: E402

# From tools/positional_eval/
from position_state import PositionState  # noqa: E402
from value_dataset import ValueDataset, default_dataset_path  # noqa: E402


DEFAULT_OUTPUT = os.path.join(_PROJECT_ROOT, "dataset", "iter_9999_valdrill.npz")


def filter_entries(entries: List[dict], tags: List[str],
                   session: str) -> List[dict]:
    """Apply --tags AND --session filters. Empty filter list = pass-through."""
    filtered = entries
    if tags:
        tag_set = set(tags)
        # AND semantics: entry must have ALL requested tags. (OR would be
        # cheap to add but AND is what a targeted export usually wants.)
        filtered = [e for e in filtered
                    if tag_set.issubset(set(e.get("tags", [])))]
    if session:
        filtered = [e for e in filtered if e.get("session_id") == session]
    return filtered


def encode_entries(entries: List[dict]) -> tuple:
    """Encode each entry → (boards, policies, values) numpy arrays.

    Skips entries whose position can't be reconstructed and prints a
    warning; the remaining valid entries still get exported. Returns
    empty arrays if nothing was encodable.
    """
    encoder = StateEncoder(num_channels=115)
    boards, values, skipped = [], [], 0

    for i, entry in enumerate(entries):
        pos_dict = entry.get("position")
        if pos_dict is None:
            print(f"[warn] entry {i}: no 'position' field, skipping")
            skipped += 1
            continue
        try:
            ps = PositionState.from_dict(pos_dict)
        except Exception as e:
            print(f"[warn] entry {i}: from_dict failed "
                  f"({type(e).__name__}: {e}), skipping")
            skipped += 1
            continue
        try:
            board = encoder.encode_board(ps.game)  # (115, 8, 8) float32
        except Exception as e:
            print(f"[warn] entry {i}: encode_board failed "
                  f"({type(e).__name__}: {e}), skipping")
            skipped += 1
            continue

        if board.shape != (115, 8, 8):
            print(f"[warn] entry {i}: unexpected board shape {board.shape}, "
                  f"skipping")
            skipped += 1
            continue

        boards.append(board)
        values.append(float(entry["value"]))

    if skipped:
        print(f"[build_value_drill_npz] {skipped} entries skipped due to errors")

    n = len(boards)
    if n == 0:
        return (np.zeros((0, 115, 8, 8), dtype=np.uint8),
                np.zeros((0, POLICY_SIZE), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))

    # Match cluster's self-play write: uint8 boards, float32 policies+values.
    # Zero-fill policies so cross-entropy contributes zero gradient per sample.
    boards_arr = np.array(boards, dtype=np.uint8)
    policies_arr = np.zeros((n, POLICY_SIZE), dtype=np.float32)
    values_arr = np.array(values, dtype=np.float32)
    return boards_arr, policies_arr, values_arr


def main():
    p = argparse.ArgumentParser(
        description="Build value-drilling .npz from local value_targets.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 2)[-1] if __doc__ else "")
    p.add_argument("-i", "--input", default=default_dataset_path(),
                   help=f"input JSON (default: {default_dataset_path()})")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                   help=f"output .npz (default: {DEFAULT_OUTPUT})")
    p.add_argument("--tags", nargs="+", default=[],
                   help="filter to entries carrying ALL these tags")
    p.add_argument("--session", default="",
                   help="filter to entries from a specific session_id")
    p.add_argument("--dry-run", action="store_true",
                   help="print counts without writing")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ds = ValueDataset(args.input)
    print(f"[build_value_drill_npz] loaded {ds.total_count()} entries "
          f"across {len(ds.session_summary())} sessions")

    filtered = filter_entries(ds.entries, args.tags, args.session)
    print(f"[build_value_drill_npz] after filters (tags={args.tags} "
          f"session={args.session or 'ALL'}): {len(filtered)} entries")

    if args.dry_run:
        # Print a per-value breakdown so the user can sanity-check the filter
        buckets = {-1: 0, 0: 0, 1: 0}
        for e in filtered:
            v = int(e["value"])
            if v in buckets:
                buckets[v] += 1
        print(f"[dry-run] value breakdown: "
              f"+1: {buckets[1]}, 0: {buckets[0]}, -1: {buckets[-1]}")
        return

    if len(filtered) == 0:
        print("[error] no entries matched the filters; nothing to write.",
              file=sys.stderr)
        sys.exit(1)

    boards, policies, values = encode_entries(filtered)
    if len(boards) == 0:
        print("[error] all entries failed to encode; check warnings above.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Atomic write via tmp + os.replace (matches alphazero.py's
    # atomic_savez_compressed pattern — see src/alphazero.py:68-73).
    tmp = args.output + ".tmp"
    np.savez_compressed(tmp, boards=boards, policies=policies, values=values)
    os.replace(tmp, args.output)

    print(f"[build_value_drill_npz] wrote {len(boards)} positions -> "
          f"{args.output}")
    print(f"[build_value_drill_npz] value breakdown: "
          f"+1={int((values > 0.5).sum())}, "
          f"0={int((np.abs(values) < 0.5).sum())}, "
          f"-1={int((values < -0.5).sum())}")


if __name__ == "__main__":
    main()
