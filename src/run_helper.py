"""
Helper job: generate self-play games using the current model checkpoint
and write the result atomically to a specified output path.

Main training launches helpers via sbatch at the start of each iter's
self-play. Each helper generates ~200 games, writes them, and exits.
Main consumes the helper files at training time (and deletes them).

Usage (from helper.sh, which sets up sbatch + working dir):
    python3 run_helper.py --model-path <path> --output-path <path>
                         [--num-games 200] [--num-simulations 800]
                         [--num-parallel 50] [--num-threads 4]
"""

import argparse
import multiprocessing
import os
import time

import numpy as np
import torch

from alphazero import (
    AlphaZeroNet,
    HAS_CPP_SELFPLAY,
    atomic_savez_compressed,
    batched_self_play,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to az_latest.pt (or a versioned checkpoint).")
    parser.add_argument("--output-path", required=True,
                        help="Path to write the helper .npz to (atomic).")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--num-simulations", type=int, default=800)
    parser.add_argument("--num-parallel", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    if not HAS_CPP_SELFPLAY:
        raise RuntimeError("Helper requires C++ self-play extension (_ext_chess).")

    print(f"[helper] Loading model from {args.model_path}", flush=True)
    model, meta = AlphaZeroNet.load_checkpoint(args.model_path, migrate=True)
    iter_num = meta.get("iteration", -1)
    print(f"[helper] Model iteration: {iter_num}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[helper] Device: {device}", flush=True)
    model = model.to(device)
    model.eval()

    t0 = time.time()
    print(f"[helper] Generating {args.num_games} games "
          f"(sims={args.num_simulations}, threads={args.num_threads})",
          flush=True)

    game_results = batched_self_play(
        model, device, args.num_games,
        num_simulations=args.num_simulations,
        temp_threshold=30,
        num_parallel=min(args.num_parallel, args.num_games),
        max_batch=512,
        num_threads=args.num_threads,
        use_tree_reuse=True,  # Aug 24: match main training. Without this,
                              # helpers ran ~1.2x slower than reuse-enabled
                              # main and timed out at the 2h45m SLURM limit
                              # on trpro-slurm1 (8 consecutive iters, 985-988).
                              # See helper_595207.log for the smoking gun:
                              # tree_reuse=off + avg_process=916us/batch
                              # (vs main's 513us on 2080 Ti) = ~3h/200 games.
    )

    # Flatten games into position-level arrays
    all_boards = []
    all_policies = []
    all_values = []
    wins_w = wins_b = draws = 0
    for boards, policies, players, outcome in game_results:
        for b, pi, player in zip(boards, policies, players):
            value = outcome if player == 0 else -outcome
            all_boards.append(b)
            all_policies.append(pi)
            all_values.append(value)
        if outcome > 0.5:
            wins_w += 1
        elif outcome < -0.5:
            wins_b += 1
        else:
            draws += 1

    gen_time = time.time() - t0
    print(f"[helper] {len(game_results)} games | {len(all_boards)} positions "
          f"| W={wins_w} B={wins_b} D={draws} | gen={gen_time:.1f}s",
          flush=True)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    print(f"[helper] Writing {args.output_path}", flush=True)
    atomic_savez_compressed(
        args.output_path,
        boards=np.array(all_boards, dtype=np.uint8),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
        num_games=np.int32(len(game_results)),
        iter_num=np.int32(iter_num),
    )
    print(f"[helper] Done.", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
