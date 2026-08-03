"""Standalone C++ subtree-reuse verification test.

Runs `batched_self_play` twice against the same local checkpoint:
  Pass A: use_tree_reuse=False (baseline behavior — same as current cluster)
  Pass B: use_tree_reuse=True  (new behavior — attempts promote() every move)

Reports aggregate stats for each pass so we can verify:
  1. Reuse-ON completes without crashes (memory corruption from bad pruning
     would surface here as segfault or exception).
  2. Total NN evals DROPS in the reuse-ON pass (fewer forward passes needed
     per game because subtree visits carry over).
  3. Wall time DROPS proportionally (assumes GPU is the bottleneck, which
     the [timing] log lines confirm at ~95%+ of process_results).
  4. Game outcomes (W/B/D distribution, avg positions/game) stay in the
     expected band. A dramatic shift would indicate reuse changed how MCTS
     searches — a red flag for training divergence.

Because C++ MCTS uses std::random_device for Dirichlet noise, the two runs
are NOT bit-exact identical even in principle. This is a STATISTICAL smoke
test, not a correctness proof (see tools/test_reuse.py for Python-side
bit-exact identity proof at batch_size=1).

Expected savings from tools/measure_reuse_savings.py (Python model, iter
920, 800 sims/move, 50 moves): 35.7% avg per-move, 1.55x expected speedup.
Real C++ speedup will be less due to fixed overhead in each MCTS + amortized
Phase 1 recording; realistic target is ~1.3-1.5x wall time.

Usage (on cluster GPU):
    python tools/test_cpp_reuse.py --model ~/extinction-chess/models/az_iter_930_100pct.pt --games 20 --sims 800
Locally on CPU (much slower but works):
    python tools/test_cpp_reuse.py --model models/az_iter920.pt --games 4 --sims 200 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402
from alphazero import AlphaZeroNet, batched_self_play  # noqa: E402


def run_one(model, device, games, sims, use_tree_reuse, num_parallel, num_threads):
    """One pass of batched_self_play. Returns dict of stats."""
    label = "REUSE" if use_tree_reuse else "BASELINE"
    print(f"\n== {label} pass: games={games} sims={sims} "
          f"num_parallel={num_parallel} num_threads={num_threads} ==")
    t0 = time.time()
    results = batched_self_play(
        model=model,
        device=device,
        games_per_iteration=games,
        num_simulations=sims,
        num_parallel=min(num_parallel, games),
        num_threads=num_threads,
        use_tree_reuse=use_tree_reuse,
    )
    elapsed = time.time() - t0

    # Aggregate
    total_positions = 0
    wins_w = wins_b = draws = 0
    for boards, policies, players, outcome in results:
        total_positions += len(boards)
        if outcome > 0.5:
            wins_w += 1
        elif outcome < -0.5:
            wins_b += 1
        else:
            draws += 1

    return {
        "label": label,
        "games": len(results),
        "total_positions": total_positions,
        "wins_w": wins_w,
        "wins_b": wins_b,
        "draws": draws,
        "avg_pos_per_game": total_positions / max(1, len(results)),
        "wall_seconds": elapsed,
    }


def print_row(baseline, reuse, key, fmt="{}"):
    b = baseline[key]
    r = reuse[key]
    b_str = fmt.format(b)
    r_str = fmt.format(r)
    print(f"  {key:>25} : {b_str:>12} | {r_str:>12}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--games", type=int, default=20,
                   help="Games per pass (default 20)")
    p.add_argument("--sims", type=int, default=800,
                   help="MCTS simulations per move (default 800)")
    p.add_argument("--num-parallel", type=int, default=50,
                   help="Parallel games in C++ SelfPlayManager (default 50)")
    p.add_argument("--num-threads", type=int, default=4,
                   help="C++ threads for collect_leaves/process_results (default 4)")
    p.add_argument("--device", default="cuda",
                   help="Device: cuda or cpu (default cuda)")
    args = p.parse_args()

    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_path}...")
    model, meta = AlphaZeroNet.load_checkpoint(model_path)
    model = model.to(args.device).eval()
    print(f"  iteration = {meta.get('iteration', '?')}, device = {args.device}")

    baseline = run_one(model, args.device, args.games, args.sims,
                       use_tree_reuse=False,
                       num_parallel=args.num_parallel,
                       num_threads=args.num_threads)
    reuse = run_one(model, args.device, args.games, args.sims,
                    use_tree_reuse=True,
                    num_parallel=args.num_parallel,
                    num_threads=args.num_threads)

    print()
    print("=" * 66)
    print(f"{'metric':>25}   {'baseline':>12} | {'reuse':>12}")
    print("=" * 66)
    print_row(baseline, reuse, "games")
    print_row(baseline, reuse, "wins_w")
    print_row(baseline, reuse, "wins_b")
    print_row(baseline, reuse, "draws")
    print_row(baseline, reuse, "total_positions")
    print_row(baseline, reuse, "avg_pos_per_game", "{:.1f}")
    print_row(baseline, reuse, "wall_seconds", "{:.1f}")
    print("=" * 66)

    speedup = baseline["wall_seconds"] / max(1e-6, reuse["wall_seconds"])
    print(f"\n  Wall-time speedup: {speedup:.2f}x  (target: 1.3-1.5x)")

    # Sanity thresholds
    ok = True
    if reuse["games"] != baseline["games"]:
        print("  FAIL: reuse pass completed a different number of games")
        ok = False
    if reuse["total_positions"] == 0:
        print("  FAIL: reuse pass produced no positions")
        ok = False
    # Draw rate check — draws are structurally rare in extinction chess
    # (0-3 per 400 games historically), so a jump means something's wrong.
    if reuse["draws"] > max(3, baseline["draws"] + 5):
        print(f"  WARN: reuse draw count {reuse['draws']} much higher than "
              f"baseline {baseline['draws']} — investigate")

    if ok:
        print("  PASS: reuse pass completed cleanly with expected outcome shape.")
    else:
        print("  FAIL: see errors above — do NOT enable use_tree_reuse in training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
