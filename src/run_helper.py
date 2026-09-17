"""
Helper job: generate self-play games using the current model checkpoint
and write the result atomically to a specified output path.

Main training launches helpers via sbatch at the start of each iter's
self-play. Each helper generates up to --num-games games in
--milestone-games chunks, checking $SLURM_JOB_END_TIME between chunks.
If the next chunk wouldn't fit before wall, the helper writes whatever
it already has and exits early — so slow nodes deliver partial data
instead of getting SIGKILL'd with nothing (deployed Sep 17 2026 as
Phase 2, after 6 consecutive iter 1108-1113 trpro-slurm1 helpers timed
out at exact 3h wall with zero output).

Usage (from helper.sh, which sets up sbatch + working dir):
    python3 run_helper.py --model-path <path> --output-path <path>
                         [--num-games 300] [--milestone-games 100]
                         [--num-simulations 800]
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


# Reserve this many seconds before SLURM_JOB_END_TIME for the atomic .npz
# write + Python cleanup. Empirically the write is <10s but we're generous.
DEADLINE_SAFETY_SECONDS = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to az_latest.pt (or a versioned checkpoint).")
    parser.add_argument("--output-path", required=True,
                        help="Path to write the helper .npz to (atomic).")
    parser.add_argument("--num-games", type=int, default=300,
                        help="Target total games. Actual output may be less "
                             "if wall-time approaches.")
    parser.add_argument("--milestone-games", type=int, default=100,
                        help="Games per milestone chunk. Helper checks "
                             "deadline between chunks.")
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

    # Deadline setup: use SLURM_JOB_END_TIME if available (a unix timestamp
    # SLURM sets to when it will SIGKILL). Subtract safety margin for the
    # .npz atomic write. If not running under SLURM, deadline is None and
    # the helper runs to completion regardless of time.
    job_end_ts_env = os.environ.get("SLURM_JOB_END_TIME", "0")
    try:
        job_end_ts = int(job_end_ts_env)
    except ValueError:
        job_end_ts = 0
    if job_end_ts > 0:
        deadline_ts = job_end_ts - DEADLINE_SAFETY_SECONDS
        now = time.time()
        print(f"[helper] SLURM deadline: {deadline_ts - now:.0f}s from now "
              f"(job ends at {job_end_ts}, safety {DEADLINE_SAFETY_SECONDS}s)",
              flush=True)
    else:
        deadline_ts = None
        print(f"[helper] no SLURM deadline detected — will run to completion",
              flush=True)

    t0 = time.time()
    print(f"[helper] Generating up to {args.num_games} games in "
          f"{args.milestone_games}-game milestones "
          f"(sims={args.num_simulations}, threads={args.num_threads})",
          flush=True)

    # Milestone loop: run batched_self_play in chunks, check deadline
    # between chunks, save whatever we have if we can't fit another chunk.
    all_game_results = []
    milestone_num = 0
    last_milestone_time = 0.0

    while len(all_game_results) < args.num_games:
        remaining = args.num_games - len(all_game_results)
        batch_size = min(args.milestone_games, remaining)
        milestone_num += 1

        # Pre-batch deadline check. Skip on first milestone — we always
        # attempt at least one chunk (if we can't do 100 games in 3h,
        # the helper is hopeless anyway; SIGKILL costs the same either way).
        if deadline_ts is not None and milestone_num > 1:
            now = time.time()
            time_remaining = deadline_ts - now
            # 1.1x safety factor: assume next milestone might be a bit slower
            estimated_next = last_milestone_time * 1.1
            if estimated_next > time_remaining:
                print(f"[helper] deadline check: {time_remaining:.0f}s left, "
                      f"next milestone estimated {estimated_next:.0f}s — "
                      f"stopping early with {len(all_game_results)} "
                      f"of {args.num_games} games",
                      flush=True)
                break

        milestone_t0 = time.time()
        print(f"[helper] milestone {milestone_num}: generating {batch_size} "
              f"games (target cumulative {len(all_game_results) + batch_size}"
              f"/{args.num_games})",
              flush=True)
        batch_results = batched_self_play(
            model, device, batch_size,
            num_simulations=args.num_simulations,
            temp_threshold=30,
            num_parallel=min(args.num_parallel, batch_size),
            max_batch=512,
            num_threads=args.num_threads,
            use_tree_reuse=True,  # Aug 24: match main training. Without this,
                                  # helpers ran ~1.2x slower than reuse-enabled
                                  # main and timed out at the 2h45m SLURM limit
                                  # on trpro-slurm1 (8 consecutive iters, 985-988).
                                  # See helper_595207.log for the smoking gun:
                                  # tree_reuse=off + avg_process=916us/batch
                                  # (vs main's 513us on 2080 Ti) = ~3h/200 games.
            # Aug 29: align exploration params with main. Prior to this,
            # helper silently ran with batched_self_play's paper-tuned
            # defaults (dirichlet_alpha=0.3, noise_weight=0.25) while main
            # had been bumped Aug 8 → so ~half of training data per iter
            # used narrower exploration than the other half. Diverging
            # exploration is bad for signal isolation when we're trying to
            # measure the bump's effect. Keep helper values EXACTLY MATCHED
            # to run_training.py's — if you change one, change the other.
            dirichlet_alpha=2.0,
            noise_weight=0.75,
        )
        last_milestone_time = time.time() - milestone_t0
        all_game_results.extend(batch_results)
        print(f"[helper] milestone {milestone_num}: completed "
              f"{len(batch_results)} games in {last_milestone_time:.1f}s "
              f"(cumulative {len(all_game_results)}/{args.num_games})",
              flush=True)

    # If we never got any games out (extremely unlikely — deadline check
    # skips the first milestone), don't write an empty .npz that would
    # confuse main's consumer.
    if not all_game_results:
        print(f"[helper] no milestones completed — exiting without writing "
              f"output file", flush=True)
        return

    # Flatten games into position-level arrays
    all_boards = []
    all_policies = []
    all_values = []
    wins_w = wins_b = draws = 0
    for boards, policies, players, outcome in all_game_results:
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
    print(f"[helper] {len(all_game_results)} games | {len(all_boards)} "
          f"positions | W={wins_w} B={wins_b} D={draws} | gen={gen_time:.1f}s "
          f"| {milestone_num} milestone(s)",
          flush=True)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    print(f"[helper] Writing {args.output_path}", flush=True)
    atomic_savez_compressed(
        args.output_path,
        boards=np.array(all_boards, dtype=np.uint8),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
        num_games=np.int32(len(all_game_results)),
        iter_num=np.int32(iter_num),
    )
    print(f"[helper] Done.", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
