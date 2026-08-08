"""
AlphaZero-style training for Extinction Chess.

Full implementation:
 - ResNet with policy + value heads (20 blocks, 256 filters, ~24M params)
 - MCTS with network policy priors
 - Self-play producing (state, policy_target, value_target) training data
 - Training on both heads: cross-entropy(policy) + MSE(value)

The C++ engine (_ext_chess) handles all game logic and board encoding.
"""

import math
import random
import re
import time
import os
import json
import glob
import subprocess
from typing import List, Tuple, Optional
from multiprocessing import Process, Queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extinction_chess import ExtinctionChess, Color, PieceType, Position
from state_encoder import StateEncoder


# Replay-buffer filename convention. Accepts cluster's own 'iter_<N>.npz' and
# also externally-produced variants like 'iter_<N>_modalA1.npz' from the
# Modal helper pipeline. Tag suffix is optional; if present it must be
# non-empty alphanumeric (no dots, underscores in the tag itself, to keep the
# parse unambiguous). Behavior with zero external files == identical to the
# original strict pattern.
_REPLAY_ITER_RE = re.compile(r"^iter_(\d+)(?:_([A-Za-z0-9]+))?\.npz$")


def _replay_iter_num(fname):
    """Extract iter number from a replay-buffer filename. Returns None if
    the name doesn't match. Used both for the loader (which files to read)
    and cleanup (which files to delete when they age out of K-buffer)."""
    m = _REPLAY_ITER_RE.match(fname)
    return int(m.group(1)) if m else None


def _replay_file_tag(fname):
    """Extract the tag suffix from a replay-buffer filename, or None if
    untagged (cluster's own iter_<N>.npz). Tags come from external
    contributors like the Modal helper pipeline ('modalA1', 'eliteB3',
    etc.) and let the loader log where positions came from."""
    m = _REPLAY_ITER_RE.match(fname)
    return m.group(2) if m else None


# ═════════════════════════════════════════════════════════════════════════════
# Atomic file writes
#
# Both the model checkpoint and replay-buffer .npz files are read by other
# processes (auto-resubmitted training jobs, helper jobs). Write to a .tmp
# path first and rename: os.replace is atomic on POSIX, so readers either
# see the full previous file or the full new one — never a half-written one.
# ═════════════════════════════════════════════════════════════════════════════

def atomic_savez_compressed(path, **arrays):
    # Pass a file object instead of a path string so numpy doesn't auto-append
    # ".npz" to ".tmp" suffixes (it does that silently for string paths).
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp_path, path)


def atomic_torch_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


# ═════════════════════════════════════════════════════════════════════════════
# SLURM helper-job management
#
# Per iter, main can launch N helper sbatch jobs that each generate a batch
# of self-play games against az_latest.pt and write a .npz file. Main reads
# those files at training time and concatenates them with its own self-play
# data — a "recency injection" that does not enter the K-buffer.
#
# Helpers are tried on a primary GPU type (typically 2080 Ti on
# delta-slurm1, which has low queue contention) and fall back to a
# secondary GPU type (typically 3090 on trpro-slurm1) if the primary is
# pending. If neither slot frees within max_wait_seconds, the helper is
# skipped for that iter — main proceeds with degraded data (current
# pipeline behavior).
# ═════════════════════════════════════════════════════════════════════════════

def _slurm_job_state(job_id):
    """Returns the SLURM state of a job (RUNNING/PENDING/COMPLETED/...) or None."""
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _wait_until_running(job_id, max_seconds):
    """Poll squeue until job is RUNNING or a terminal state. Returns True if RUNNING."""
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        state = _slurm_job_state(job_id)
        if state == "RUNNING":
            return True
        if state in (None, "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            return False
        time.sleep(2)
    return False


def _sbatch_helper(helper_script_path, output_path, gres, nodelist):
    """Submit one helper job. Returns SLURM job ID or None on submission failure."""
    cmd = [
        "sbatch", "--parsable",
        f"--gres={gres}",
        f"--nodelist={nodelist}",
        helper_script_path,
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print(f"         [helper] sbatch failed ({gres} on {nodelist}): "
                  f"{result.stderr.strip()}")
            return None
        return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"         [helper] sbatch error: {e}")
        return None


def _cancel_job(job_id):
    if job_id is None:
        return
    try:
        subprocess.run(["scancel", str(job_id)], check=False,
                       capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def launch_helper_with_fallback(iter_num, helper_id, helper_script_path, helper_dir,
                                max_wait_seconds=25,
                                primary_gres="gpu:rtx_2080_ti:1",
                                primary_node="delta-slurm1",
                                fallback_gres="gpu:rtx_3090:1",
                                fallback_node="trpro-slurm1"):
    """Launch one helper job with a primary→fallback GPU strategy.

    Returns (job_id, output_path). If neither primary nor fallback could
    allocate within max_wait_seconds, returns (None, None) and main will
    proceed without this helper's data.
    """
    output_path = os.path.join(helper_dir, f"helper_v{iter_num}_id{helper_id}.npz")

    # Primary attempt
    job_id = _sbatch_helper(helper_script_path, output_path, primary_gres, primary_node)
    if job_id is not None and _wait_until_running(job_id, max_wait_seconds):
        print(f"         [helper {helper_id}] running on {primary_node} (job {job_id})")
        return job_id, output_path

    # Primary pending or failed — cancel and try fallback
    _cancel_job(job_id)
    time.sleep(2)

    job_id = _sbatch_helper(helper_script_path, output_path, fallback_gres, fallback_node)
    if job_id is not None and _wait_until_running(job_id, max_wait_seconds):
        print(f"         [helper {helper_id}] running on {fallback_node} (job {job_id})")
        return job_id, output_path

    # Both attempts failed — give up for this iter
    _cancel_job(job_id)
    print(f"         [helper {helper_id}] could not allocate a GPU, skipping")
    return None, None


def consume_helper_files(helper_specs):
    """Read all delivered helper .npz files and return concatenated arrays.

    helper_specs: list of (handle, output_path) tuples.
    - handle for SLURM is an int job_id (no .get() method) — we just check
      file presence at consume time.
    - handle for Modal is a FunctionCall-like object with .get(), which we
      call to wait for the helper to finish and refresh the shared volume.
    Tuples with output_path=None are skipped.
    Files are deleted after successful consumption.

    Returns (boards, policies, values, total_games). All None/0 if nothing.
    """
    # Phase 1: wait for each helper to complete (backend-specific via duck-typing)
    for handle, output_path in helper_specs:
        if output_path is None:
            continue
        if hasattr(handle, "get"):
            try:
                handle.get(timeout=600)
            except Exception as e:
                print(f"         [helper] wait failed for "
                      f"{os.path.basename(output_path)}: {e}")

    # Phase 2: load every file that exists, regardless of how it got there
    all_b, all_p, all_v = [], [], []
    total_games = 0
    for handle, output_path in helper_specs:
        if output_path is None:
            continue
        if not os.path.exists(output_path):
            print(f"         [helper] missing file: {os.path.basename(output_path)} "
                  f"(handle: {handle}) — proceeding without")
            continue
        try:
            with np.load(output_path) as data:
                all_b.append(data["boards"].astype(np.float32))
                all_p.append(np.asarray(data["policies"]))
                all_v.append(np.asarray(data["values"]))
                n_games = int(data["num_games"]) if "num_games" in data.files else 0
                n_positions = len(data["boards"])
            total_games += n_games
            print(f"         [helper] consumed {os.path.basename(output_path)} "
                  f"({n_games} games, {n_positions} positions)")
            os.remove(output_path)
        except Exception as e:
            print(f"         [helper] failed to load {output_path}: {e}")
    if not all_b:
        return None, None, None, 0
    return (np.concatenate(all_b), np.concatenate(all_p),
            np.concatenate(all_v), total_games)


def cleanup_stale_helper_files(current_iter_num, helper_dir):
    """Remove leftover helper_v*_id*.npz files from older iters."""
    if not os.path.isdir(helper_dir):
        return
    for fname in os.listdir(helper_dir):
        if not (fname.startswith("helper_v") and fname.endswith(".npz")):
            continue
        try:
            # parse "helper_v{N}_id{M}.npz"
            v_part = fname.split("_v")[1].split("_")[0]
            version = int(v_part)
            if version < current_iter_num:
                os.remove(os.path.join(helper_dir, fname))
        except (IndexError, ValueError):
            pass


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark battery — fire-and-forget test suite for milestone checkpoints
# ═════════════════════════════════════════════════════════════════════════════
# After main saves an az_iter_<N>_XXpct.pt at a multiple of 10, main sbatches
# 13 tests to trpro-slurm2 (RTX 4090) plus one aggregator job. All are
# non-blocking: main proceeds to iter N+1 immediately. The aggregator runs
# after all tests complete (--dependency=afterany) and writes a single
# summary file to ~/extinction-chess/benchmark_results/iter_<N>.txt.
#
# The 13 tests:
#   - 5 recent H2H (vs iter N-10, N-20, N-30, N-40, N-50)
#   - 5 distant H2H (vs iter N-110, N-120, N-130, N-140, N-150)
#   - Win-taking multi-model
#   - Tactical random
#   - vs iter 100
# Any test whose reference checkpoint is missing is silently skipped.

# Fixed historical anchors always included in win-taking (in addition to
# rolling recent + distant sets). Reflects checkpoints known to exist and
# be interesting for tactical comparison.
_WIN_TAKING_HISTORICAL_ANCHORS = [560, 550, 540, 520, 500, 480, 340, 100]


def _find_checkpoint(iter_num, models_dir):
    """Return filename (basename, not full path) of az_iter_<N>_XXpct.pt.

    Prefers _100pct.pt if multiple matches exist. Returns None if no
    checkpoint file for this iter is on disk.
    """
    pattern = os.path.join(models_dir, f"az_iter_{iter_num}_*pct.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    for m in matches:
        if "_100pct.pt" in m:
            return os.path.basename(m)
    return os.path.basename(matches[0])


def _sbatch_benchmark(iter_dir, log_name, wrap_cmd, gres="gpu:rtx_4090:1",
                     nodelist="trpro-slurm2", cpus=4, mem="16G", time_limit="4:00:00",
                     dependency=None):
    """Submit one benchmark test job. Returns SLURM JOBID or None on failure.

    Fire-and-forget: does NOT wait for the job to reach RUNNING. Main
    proceeds immediately.
    """
    cmd = ["sbatch", "--parsable", "--partition=compute"]
    if gres:
        cmd.append(f"--gres={gres}")
    if nodelist:
        cmd.append(f"--nodelist={nodelist}")
    cmd += [
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time_limit}",
        f"--output={os.path.join(iter_dir, log_name)}",
    ]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(f"--wrap={wrap_cmd}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print(f"         [benchmark] sbatch failed: {result.stderr.strip()}")
            return None
        return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        print(f"         [benchmark] sbatch error: {e}")
        return None


def launch_benchmark_battery(iter_num, models_dir,
                             benchmark_dir="~/benchmark_battery",
                             results_dir="~/extinction-chess/benchmark_results",
                             aggregator_script="~/extinction-chess/src/aggregate_benchmark.py",
                             src_cwd="~/extinction-chess/src"):
    """Sbatch a full benchmark battery for the given iter, non-blocking.

    Called from main's training loop after a milestone checkpoint save
    (multiples of 10). Main proceeds to iter N+1 immediately after this
    returns; tests run in background on trpro-slurm2 and results appear
    in results_dir/iter_<N>.txt when the aggregator completes.
    """
    current_ckpt = _find_checkpoint(iter_num, models_dir)
    if current_ckpt is None:
        print(f"         [benchmark] iter {iter_num}: no checkpoint file, skipping battery")
        return

    benchmark_dir = os.path.expanduser(benchmark_dir)
    results_dir = os.path.expanduser(results_dir)
    aggregator_script = os.path.expanduser(aggregator_script)
    src_cwd = os.path.expanduser(src_cwd)

    iter_dir = os.path.join(benchmark_dir, f"iter_{iter_num}")
    os.makedirs(iter_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"         [benchmark] launching battery for iter {iter_num}")

    def _submit_h2h(opp_iter, kind):
        """Submit one H2H test. Returns (opp_iter, jobid) or None if skipped."""
        opp_ckpt = _find_checkpoint(opp_iter, models_dir)
        if opp_ckpt is None:
            print(f"         [benchmark] {kind} iter {opp_iter}: no checkpoint, skipping")
            return None
        wrap = (
            "export PYTHONUNBUFFERED=1 && "
            f"cd {src_cwd} && "
            "python3 setup.py build_ext --inplace && "
            f"python3 compare_extensive.py --m1 {opp_ckpt} --m2 {current_ckpt} --device cuda"
        )
        log = f"compare_{iter_num}vs{opp_iter}_%j.log"
        jid = _sbatch_benchmark(iter_dir, log, wrap)
        if jid is not None:
            print(f"         [benchmark] {kind} H2H vs iter {opp_iter} → job {jid}")
        return jid

    job_ids = []

    # 5 recent H2H
    for opp in [iter_num - 10 * i for i in range(1, 6)]:
        jid = _submit_h2h(opp, "recent")
        if jid is not None:
            job_ids.append(jid)

    # 5 distant H2H
    for opp in [iter_num - 10 * i for i in range(11, 16)]:
        jid = _submit_h2h(opp, "distant")
        if jid is not None:
            job_ids.append(jid)

    # Win-taking multi-model
    wt_iters = (
        [iter_num] +
        [iter_num - 10 * i for i in range(1, 6)] +
        [iter_num - 10 * i for i in range(11, 16)] +
        _WIN_TAKING_HISTORICAL_ANCHORS
    )
    wt_models = []
    for i in wt_iters:
        if i <= 0:
            continue
        f = _find_checkpoint(i, models_dir)
        if f and f not in wt_models:
            wt_models.append(f)
    if wt_models:
        wrap = (
            "export PYTHONUNBUFFERED=1 && "
            f"cd {src_cwd} && "
            "python3 setup.py build_ext --inplace && "
            f"python3 bench_win_taking.py --models {' '.join(wt_models)} "
            "--sims 20 50 100 200 --positions 200 --hard-only --min-distance 5"
        )
        jid = _sbatch_benchmark(iter_dir, "bench_win_taking_%j.log", wrap,
                                time_limit="2:00:00")
        if jid is not None:
            job_ids.append(jid)
            print(f"         [benchmark] win-taking ({len(wt_models)} models) → job {jid}")

    # Tactical random
    wrap = (
        "export PYTHONUNBUFFERED=1 && "
        f"cd {src_cwd} && "
        "python3 setup.py build_ext --inplace && "
        f"python3 bench_vs_tactical.py --model {current_ckpt} --device cuda"
    )
    jid = _sbatch_benchmark(iter_dir, f"tactical_{iter_num}_%j.log", wrap)
    if jid is not None:
        job_ids.append(jid)
        print(f"         [benchmark] tactical random → job {jid}")

    # vs iter 100
    ckpt_100 = _find_checkpoint(100, models_dir)
    if ckpt_100:
        wrap = (
            "export PYTHONUNBUFFERED=1 && "
            f"cd {src_cwd} && "
            "python3 setup.py build_ext --inplace && "
            f"python3 compare_extensive.py --m1 {ckpt_100} --m2 {current_ckpt} --device cuda"
        )
        jid = _sbatch_benchmark(iter_dir, f"compare_{iter_num}vs100_%j.log", wrap)
        if jid is not None:
            job_ids.append(jid)
            print(f"         [benchmark] vs iter 100 → job {jid}")

    # Aggregator (waits for all tests via --dependency=afterany)
    if job_ids:
        dep = "afterany:" + ":".join(str(j) for j in job_ids)
        results_file = os.path.join(results_dir, f"iter_{iter_num}.txt")
        wrap = (
            "export PYTHONUNBUFFERED=1 && "
            f"python3 {aggregator_script} "
            f"--iter {iter_num} "
            f"--iter-dir {iter_dir} "
            f"--results-file {results_file}"
        )
        agg_jid = _sbatch_benchmark(
            iter_dir, "aggregator_%j.log", wrap,
            gres=None, nodelist=None, cpus=1, mem="2G",
            time_limit="00:30:00", dependency=dep,
        )
        if agg_jid is not None:
            print(f"         [benchmark] aggregator → job {agg_jid} "
                  f"(waits for {len(job_ids)} tests)")
    else:
        print(f"         [benchmark] no test jobs launched — skipping aggregator")


# ═════════════════════════════════════════════════════════════════════════════
# Move encoding  (Move ↔ policy index)
#
# 76 planes × 64 squares = 4864 total policy logits
#   Planes  0–55: Queen-type moves (8 directions × 7 distances)
#   Planes 56–63: Knight moves (8 offsets)
#   Planes 64–72: Underpromotions (knight/bishop/rook × 3 file deltas)
#   Planes 73–75: King promotions (3 file deltas)
#   Queen promotions are encoded as queen-type moves (no special plane).
# ═════════════════════════════════════════════════════════════════════════════

NUM_POLICY_PLANES = 76
POLICY_SIZE = NUM_POLICY_PLANES * 64   # 4864
NUM_INPUT_CHANNELS = 115  # 9 positions × 12 pieces + player + endangered + 4 castling + halfmove

# Direction table for queen-type moves
#   index → (dr, df)
QUEEN_DIRS = [
    ( 1,  0),  # N
    ( 1,  1),  # NE
    ( 0,  1),  # E
    (-1,  1),  # SE
    (-1,  0),  # S
    (-1, -1),  # SW
    ( 0, -1),  # W
    ( 1, -1),  # NW
]

# Knight offsets
KNIGHT_OFFSETS = [
    ( 2,  1), ( 2, -1), (-2,  1), (-2, -1),
    ( 1,  2), ( 1, -2), (-1,  2), (-1, -2),
]

# Build reverse lookup: (dr, df) → direction index
_DIR_LOOKUP = {}
for i, (dr, df) in enumerate(QUEEN_DIRS):
    for dist in range(1, 8):
        _DIR_LOOKUP[(dr * dist, df * dist)] = (i, dist)

_KNIGHT_LOOKUP = {}
for i, (dr, df) in enumerate(KNIGHT_OFFSETS):
    _KNIGHT_LOOKUP[(dr, df)] = i


def move_to_index(move) -> int:
    """Convert a Move to a policy index in [0, 4864)."""
    fr = move.from_pos
    to = move.to_pos
    from_sq = fr.rank * 8 + fr.file
    dr = to.rank - fr.rank
    df = to.file - fr.file

    # Underpromotions / king promotions
    promo = move.promotion
    if promo is not None and promo != PieceType.QUEEN:
        # File delta → direction index (0, 1, 2 for df = -1, 0, +1)
        df_idx = df + 1   # -1→0, 0→1, +1→2

        if promo == PieceType.KNIGHT:
            plane = 64 + df_idx
        elif promo == PieceType.BISHOP:
            plane = 64 + 3 + df_idx
        elif promo == PieceType.ROOK:
            plane = 64 + 6 + df_idx
        elif promo == PieceType.KING:
            plane = 73 + df_idx
        else:
            plane = 0  # fallback
        return plane * 64 + from_sq

    # Knight moves
    if (dr, df) in _KNIGHT_LOOKUP:
        plane = 56 + _KNIGHT_LOOKUP[(dr, df)]
        return plane * 64 + from_sq

    # Queen-type moves (includes queen promotions and castling king moves)
    if (dr, df) in _DIR_LOOKUP:
        direction, distance = _DIR_LOOKUP[(dr, df)]
        plane = direction * 7 + (distance - 1)
        return plane * 64 + from_sq

    # Fallback (shouldn't happen for valid moves)
    return 0


def index_to_move_info(index: int) -> Tuple[int, int, int, Optional[int]]:
    """Convert policy index back to (from_sq, to_sq, plane, promotion).
    Returns raw squares — caller maps to actual Move objects."""
    plane = index // 64
    from_sq = index % 64
    from_r, from_f = from_sq // 8, from_sq % 8

    promo = None

    if plane < 56:
        # Queen-type move
        direction = plane // 7
        distance = (plane % 7) + 1
        dr, df = QUEEN_DIRS[direction]
        to_r = from_r + dr * distance
        to_f = from_f + df * distance
    elif plane < 64:
        # Knight move
        ki = plane - 56
        dr, df = KNIGHT_OFFSETS[ki]
        to_r = from_r + dr
        to_f = from_f + df
    elif plane < 73:
        # Underpromotion
        sub = plane - 64
        piece_idx = sub // 3     # 0=knight, 1=bishop, 2=rook
        df_idx = sub % 3         # 0=left, 1=straight, 2=right
        df = df_idx - 1
        dr = 1 if from_r < 4 else -1  # white goes up, black goes down
        to_r = from_r + dr
        to_f = from_f + df
        promo = [PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK][piece_idx]
    else:
        # King promotion
        df_idx = plane - 73
        df = df_idx - 1
        dr = 1 if from_r < 4 else -1
        to_r = from_r + dr
        to_f = from_f + df
        promo = PieceType.KING

    to_sq = to_r * 8 + to_f
    return from_sq, to_sq, plane, promo


# ═════════════════════════════════════════════════════════════════════════════
# Network
# ═════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Conv → BN → ReLU → Conv + skip."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-scale network for Extinction Chess.

    Architecture (matches the original paper):
      Input:  115 × 8 × 8 (9 positions × 12 pieces + player + endangered + 4 castling + hmclock)
      Body:   Conv 3×3 → 20 residual blocks (256 filters)
      Policy: Conv 1×1 (2 filters) → BN → ReLU → FC → 4864 logits
      Value:  Conv 1×1 (1 filter)  → BN → ReLU → FC → 256 → ReLU → FC → 1 → tanh
    """

    def __init__(self, in_channels: int = NUM_INPUT_CHANNELS, num_filters: int = 256,
                 num_blocks: int = 20, policy_size: int = POLICY_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.policy_size = policy_size

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_blocks)]
        )

        # Policy head (AlphaZero paper: conv 1×1 → 2 filters → BN → ReLU → FC)
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, policy_size)

        # Value head (conv 1×1 → 1 filter → BN → ReLU → FC 256 → ReLU → FC 1 → tanh)
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(8 * 8, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        """Returns (policy_logits, value) where policy is raw logits (batch, 4864)."""
        # Body
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v

    def save_checkpoint(self, path, **metadata):
        metadata["model_type"] = "alphazero"
        metadata["in_channels"] = self.in_channels
        metadata["num_filters"] = self.num_filters
        metadata["num_blocks"] = self.num_blocks
        metadata["policy_size"] = self.policy_size
        atomic_torch_save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(cls, path, migrate: bool = False) -> Tuple["AlphaZeroNet", dict]:
        data = torch.load(path, weights_only=False, map_location="cpu")
        meta = data.get("metadata", {})
        old_in_channels = meta.get("in_channels", 14)

        if migrate and old_in_channels < NUM_INPUT_CHANNELS:
            # Training path: expand to current channel count
            net = cls(
                in_channels=NUM_INPUT_CHANNELS,
                num_filters=meta.get("num_filters", 256),
                num_blocks=meta.get("num_blocks", 20),
                policy_size=meta.get("policy_size", POLICY_SIZE),
            )
            state = data["state_dict"]
            old_conv_w = state["input_conv.weight"]  # [filters, old_ch, 3, 3]
            new_conv_w = torch.zeros_like(net.input_conv.weight)  # [filters, new_ch, 3, 3]
            # Old layout (14ch): 0-11 pieces, 12 current_player, 13 endangered
            # New layout (115ch): 0-11 pieces, 12-107 history, 108 player, 109 endangered, 110-113 castling, 114 halfmove
            new_conv_w[:, :12, :, :] = old_conv_w[:, :12, :, :]   # piece planes stay at 0-11
            new_conv_w[:, 108, :, :] = old_conv_w[:, 12, :, :]    # current player: 12 → 108
            new_conv_w[:, 109, :, :] = old_conv_w[:, 13, :, :]    # endangered: 13 → 109
            state["input_conv.weight"] = new_conv_w
            print(f"  Checkpoint migration: {old_in_channels} → {NUM_INPUT_CHANNELS} "
                  f"input channels (zero-init expansion, remapped ch12→108, ch13→109)")
        else:
            # Native path: load at checkpoint's own channel count (faster inference)
            net = cls(
                in_channels=old_in_channels,
                num_filters=meta.get("num_filters", 256),
                num_blocks=meta.get("num_blocks", 20),
                policy_size=meta.get("policy_size", POLICY_SIZE),
            )
            state = data["state_dict"]

        net.load_state_dict(state)
        return net, meta


# ═════════════════════════════════════════════════════════════════════════════
# Evaluator wrapper (for compatibility with existing code)
# ═════════════════════════════════════════════════════════════════════════════

class AlphaZeroEvaluator:
    """Wraps AlphaZeroNet for inference.  Returns value only (for backward compat)."""

    def __init__(self, model: AlphaZeroNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.in_channels = model.in_channels
        self.encoder = StateEncoder(num_channels=model.in_channels)
        self.model.eval()

    def evaluate(self, game: ExtinctionChess) -> float:
        if game.game_over:
            if game.winner == Color.WHITE:  return 1.0
            elif game.winner == Color.BLACK: return -1.0
            return 0.0
        board_tensor = self.encoder.encode_board(game)
        with torch.no_grad():
            t = torch.tensor(board_tensor, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            _, v = self.model(t)
            return v.item()

    def evaluate_with_policy(self, game: ExtinctionChess):
        """Return (policy_logits, value) for a single position."""
        board_tensor = self.encoder.encode_board(game)
        with torch.no_grad():
            t = torch.tensor(board_tensor, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
            p, v = self.model(t)
            return p.squeeze(0).cpu().numpy(), v.item()

    def batch_evaluate(self, games: List[ExtinctionChess]) -> np.ndarray:
        boards = np.stack([self.encoder.encode_board(g) for g in games])
        with torch.no_grad():
            t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            _, v = self.model(t)
            return v.cpu().numpy()

    def batch_evaluate_with_policy(self, games: List[ExtinctionChess]):
        """Return (policy_logits, values) for a batch of positions.
        policy_logits: numpy (N, 4864), values: numpy (N,)"""
        boards = np.stack([self.encoder.encode_board(g) for g in games])
        with torch.no_grad():
            t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            p, v = self.model(t)
            return p.cpu().numpy(), v.cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# MCTS with policy priors
# ═════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ('game', 'parent', 'move', 'prior',
                 'children', 'visit_count', 'value_sum', 'is_expanded',
                 'virtual_loss')

    def __init__(self, game, parent=None, move=None, prior=1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.value_sum = 0.0      # from WHITE's perspective
        self.is_expanded = False
        self.virtual_loss = 0

    def q_from_parent(self):
        vc = self.visit_count + self.virtual_loss
        if vc == 0:
            return 0.0
        # VL must reduce attractiveness from the PARENT's perspective. Since
        # value_sum is white-perspective, flip VL sign for black parents so
        # that after the -wq flip below, VL still penalizes.
        parent_is_white = self.parent.game.current_player == Color.WHITE
        vl_signed = self.virtual_loss if parent_is_white else -self.virtual_loss
        wq = (self.value_sum - vl_signed) / vc
        return wq if parent_is_white else -wq

    def ucb(self, c_puct):
        vc = self.visit_count + self.virtual_loss
        parent_vc = self.parent.visit_count + self.parent.virtual_loss
        return (self.q_from_parent()
                + c_puct * self.prior
                * math.sqrt(parent_vc) / (1 + vc))

    def best_child(self, c_puct):
        return max(self.children, key=lambda c: c.ucb(c_puct))


def _copy_game(game):
    """Make a copy of a Game for search."""
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    return gc


def _eval_white(evaluator: AlphaZeroEvaluator, game):
    """Evaluate from White's perspective."""
    if game.game_over:
        if game.winner == Color.WHITE:  return 1.0
        elif game.winner == Color.BLACK: return -1.0
        return 0.0
    v = evaluator.evaluate(game)
    return v if game.current_player == Color.WHITE else -v


def _expand_node(node, policy_logits):
    """Expand a node using policy logits. Returns value from white's perspective.

    Guards against double-expansion: if the same leaf is selected twice in a
    batch (rare with virtual loss but possible when priors are weak or the
    subtree is thin), we must skip re-expansion or the children list gets
    doubled and stats get corrupted. Matches the C++ MCTS fix (facb668)."""
    if node.is_expanded:
        return None
    child_legal = node.game.get_legal_moves()
    if not child_legal:
        node.is_expanded = True
        return _terminal_value(node.game)

    m_indices = [move_to_index(m) for m in child_legal]
    m_logits = np.array([policy_logits[i] for i in m_indices])
    m_logits -= m_logits.max()
    child_probs = np.exp(m_logits)
    child_probs /= child_probs.sum() + 1e-8

    for cm, cp in zip(child_legal, child_probs):
        gc = _copy_game(node.game)
        if gc.make_move(cm):
            node.children.append(MCTSNode(gc, parent=node, move=cm, prior=cp))
    node.is_expanded = True
    return None  # value comes from the evaluator


def _terminal_value(game):
    """Get value from white's perspective for a terminal game."""
    if game.game_over:
        if game.winner == Color.WHITE: return 1.0
        if game.winner == Color.BLACK: return -1.0
        return 0.0
    return 0.0


def _add_virtual_loss(node):
    """Add virtual loss along path from node to root."""
    n = node
    while n is not None:
        n.virtual_loss += 1
        n = n.parent


def _remove_virtual_loss(node):
    """Remove virtual loss along path from node to root."""
    n = node
    while n is not None:
        n.virtual_loss -= 1
        n = n.parent


def _backpropagate(node, white_value):
    """Backpropagate value from node to root."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += white_value
        node = node.parent


def descend_root(prev_root, moves):
    """Walk down prev_root along a sequence of played moves.

    Used by callers of mcts_search(return_root=True) to locate the subtree
    corresponding to the current game position, so they can pass it back as
    prev_root for subtree reuse.

    Args:
        prev_root: MCTSNode returned by an earlier mcts_search(return_root=True) call.
        moves: iterable of Move objects representing the moves played since
               prev_root was returned. For self-play (one model both sides),
               this is 1 ply (own move). For H2H between different models,
               this is 2 plies (own move + opponent's response).

    Returns:
        MCTSNode representing the current position (already expanded, with
        visits > 0), suitable for reuse; or None if the tree can't be
        traversed (move not in tree, node unexpanded, or unvisited).
    """
    node = prev_root
    for m in moves:
        found = None
        for child in node.children:
            if child.move == m:
                found = child
                break
        if found is None or not found.is_expanded or found.visit_count == 0:
            return None
        node = found
    return node


BATCH_SIZE_MCTS = 8  # Number of leaves to collect before batched eval

# Try to import C++ MCTS and batched self-play
try:
    from _ext_chess import MCTS as CppMCTS, move_to_index as cpp_move_to_index
    HAS_CPP_MCTS = True
except ImportError:
    HAS_CPP_MCTS = False

try:
    from _ext_chess import SelfPlayManager as CppSelfPlayManager
    HAS_CPP_SELFPLAY = True
except ImportError:
    HAS_CPP_SELFPLAY = False


def mcts_search_cpp(game, evaluator: AlphaZeroEvaluator,
                    num_simulations: int = 800, c_puct: float = 2.5,
                    dirichlet_alpha: float = 0.3, noise_weight: float = 0.25,
                    tactical_shortcuts: bool = True,
                    batch_size: int = 8):
    """
    MCTS using C++ tree search with Python neural network evaluation.
    Returns list of (move, visit_count) pairs.
    """
    legal = game.get_legal_moves()
    if not legal:
        return []

    # Build index→move lookup for mapping results back
    index_to_move = {}
    for m in legal:
        idx = move_to_index(m)
        index_to_move[idx] = m

    # Create C++ MCTS object
    mcts = CppMCTS(game, num_simulations, c_puct,
                   dirichlet_alpha, noise_weight,
                   tactical_shortcuts, batch_size)

    # First: expand root with a single NN call
    policy_logits, _ = evaluator.evaluate_with_policy(game)
    mcts.expand_root(policy_logits)

    # Main search loop (skipped if tactical shortcut fired)
    while not mcts.is_done():
        # Phase 1: C++ selects leaves and encodes their boards
        boards = mcts.select_leaves(batch_size)

        if len(boards) == 0:
            continue

        # Phase 2: batch NN evaluation on GPU
        with torch.no_grad():
            t = torch.tensor(np.asarray(boards), dtype=torch.float32,
                             device=evaluator.device)
            policies, values = evaluator.model(t)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        # Phase 3: C++ expands nodes and backpropagates
        mcts.process_results(policies, values)

    # Map results back to Move objects using policy indices
    results = mcts.get_results()  # list of (policy_index, visit_count)
    out = []
    for idx, visits in results:
        if idx in index_to_move:
            out.append((index_to_move[idx], visits))
    return out


def mcts_search(game, evaluator: AlphaZeroEvaluator,
                num_simulations: int = 800, c_puct: float = 2.5,
                dirichlet_alpha: float = 0.3, noise_weight: float = 0.25,
                tactical_shortcuts: bool = True,
                progress_callback=None,
                checkpoint_sims=None,
                checkpoint_callback=None,
                state_callback=None,
                should_stop=None,
                prev_root=None,
                return_root=False):
    """
    AlphaZero-style MCTS with batched neural network evaluation.
    Collects multiple leaves via virtual loss, evaluates in one GPU call.
    Returns (move_visits, root_value) where move_visits is list of (move, visit_count)
    and root_value is the SEARCH-REFINED Q-value of the root position
    (from current player's perspective; falls back to raw NN value if no sims).

    progress_callback: optional callable (sims_done, sims_total) invoked
    approximately once per MCTS batch (every ~BATCH_SIZE_MCTS=8 sims).

    checkpoint_sims / checkpoint_callback: optional. checkpoint_sims is a
    list/set of sim counts at which we want to snapshot the current search
    state; checkpoint_callback is called as
        checkpoint_callback(sim_count_reached, move_visits, refined_value)
    the first time sims_done crosses (>=) each value in checkpoint_sims.
    Move visits are current visit counts; refined_value is the current
    root Q-value in the current player's perspective. Used by the positional
    evaluation tool to get all sim-count snapshots from ONE MCTS run
    instead of running MCTS from scratch for each sim setting.

    Kept out of the hot path when None (checked once per batch).

    state_callback: optional callable, invoked every batch AFTER checkpoint
    processing with signature
        state_callback(sims_done, move_visits, refined_value)
    Same payload shape as checkpoint_callback. Enables live UI updates of
    the current tree state without waiting for a checkpoint threshold.
    Kept out of the hot path when None.

    should_stop: optional callable returning bool. Consulted at the END of
    each batch. If it returns True, MCTS returns immediately with the
    current state. Useful for interactive stop from a UI without waiting
    for num_simulations to complete. The returned move_visits reflect
    whatever the tree looked like at stop time.

    prev_root: optional MCTSNode from an earlier mcts_search(return_root=True)
    call, already promoted by the caller (see descend_root). If provided and
    valid (expanded, has children), it's used as the root and its existing
    visits are preserved — only (num_simulations - prev_root.visit_count)
    additional sims are run. Tactical shortcut is skipped (we've already
    committed to this subtree). Falls back to fresh MCTS if prev_root
    isn't usable.

    return_root: if True, returns (move_visits, root_value, root) so caller
    can pass root back as prev_root next call. Default False for backward
    compatibility with existing callers.
    """
    current = game.current_player

    # Check if we can reuse the previous root's subtree
    can_reuse = (prev_root is not None
                 and prev_root.is_expanded
                 and len(prev_root.children) > 0
                 and prev_root.visit_count > 0)

    if can_reuse:
        # ── REUSE PATH ──
        root = prev_root
        root.parent = None
        sims_done = root.visit_count

        # Root value from existing search state
        white_q = root.value_sum / root.visit_count
        root_value = white_q if current == Color.WHITE else -white_q

        # Add fresh Dirichlet noise to root's children priors (skipped for
        # deterministic paths like benchmarks where noise_weight=0)
        if dirichlet_alpha > 0 and noise_weight > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for child, n in zip(root.children, noise):
                child.prior = (1 - noise_weight) * child.prior + noise_weight * n

        # Skip the sim loop entirely if we already have enough visits
        if sims_done >= num_simulations:
            move_visits = [(ch.move, ch.visit_count) for ch in root.children]
            if return_root:
                return move_visits, root_value, root
            return move_visits, root_value

        # Skip the fresh-init block below and jump straight into the sim loop
        # (structured via a marker; see below)
        _skip_fresh_init = True
    else:
        _skip_fresh_init = False
        root = MCTSNode(game)
        sims_done = 0

    if not _skip_fresh_init:
        # ── FRESH PATH ──
        # Expand root with policy priors (single eval)
        legal = game.get_legal_moves()
        if not legal:
            if return_root:
                return [], 0.0, root
            return [], 0.0

        # Tactical shortcut: force instant wins (but allow blunders)
        if tactical_shortcuts:
            winning_moves = []
            for m in legal:
                gc = _copy_game(game)
                if not gc.make_move(m):
                    continue
                if gc.game_over and gc.winner == current:
                    winning_moves.append(m)
            if winning_moves:
                num_winners = len(winning_moves)
                per_move = num_simulations // num_winners
                remainder = num_simulations % num_winners
                result = []
                win_idx = 0
                for m in legal:
                    if m in winning_moves:
                        result.append((m, per_move + (1 if win_idx < remainder else 0)))
                        win_idx += 1
                    else:
                        result.append((m, 0))
                if return_root:
                    return result, 1.0, root
                return result, 1.0

        policy_logits, root_value = evaluator.evaluate_with_policy(game)

        move_indices = [move_to_index(m) for m in legal]
        move_logits = np.array([policy_logits[i] for i in move_indices])
        move_logits -= move_logits.max()
        probs = np.exp(move_logits)
        probs /= probs.sum() + 1e-8

        if dirichlet_alpha > 0 and noise_weight > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
            probs = (1 - noise_weight) * probs + noise_weight * noise

        for m, p in zip(legal, probs):
            gc = _copy_game(game)
            if gc.make_move(m):
                root.children.append(MCTSNode(gc, parent=root, move=m, prior=p))
        root.is_expanded = True

        if not root.children:
            if return_root:
                return [], root_value, root
            return [], root_value

        # Account for the initial NN eval as a sim. This matches reuse
        # semantics where the promoted node's visit_count already includes
        # the initial leaf visit from the old search. Without this, fresh's
        # first-sim UCB would use sqrt(0) (breaking ties by iteration order)
        # while reuse would use sqrt(N_prev) (breaking by prior). See
        # docstring for prev_root.
        white_root_value = root_value if current == Color.WHITE else -root_value
        root.visit_count = 1
        root.value_sum = white_root_value
        sims_done = 1

    # Precompute sorted checkpoint list (mutated in the loop as they fire)
    if checkpoint_callback is not None and checkpoint_sims:
        _remaining_checkpoints = sorted(
            {int(s) for s in checkpoint_sims if 0 < int(s) <= num_simulations})
    else:
        _remaining_checkpoints = []

    # Run simulations in batches
    # (sims_done is preserved from either fresh init (0) or reuse init
    # (existing visit_count), so the loop runs only the remaining sims)
    while sims_done < num_simulations:
        batch_size = min(BATCH_SIZE_MCTS, num_simulations - sims_done)
        leaves = []       # nodes needing NN eval
        terminal = []     # (node, value) for terminal/already-expanded nodes

        # Select batch_size leaves using virtual loss for diversity
        for _ in range(batch_size):
            node = root
            while node.is_expanded and node.children:
                node = node.best_child(c_puct)

            if node.game.game_over:
                tv = _terminal_value(node.game)
                terminal.append((node, tv))
                _add_virtual_loss(node)
            elif node.is_expanded:
                tv = _terminal_value(node.game)
                terminal.append((node, tv))
                _add_virtual_loss(node)
            else:
                leaves.append(node)
                _add_virtual_loss(node)

        # Batch evaluate all non-terminal leaves
        if leaves:
            games_batch = [n.game for n in leaves]
            all_policies, all_values = evaluator.batch_evaluate_with_policy(games_batch)

            for i, node in enumerate(leaves):
                _expand_node(node, all_policies[i])
                v = all_values[i]
                white_value = v if node.game.current_player == Color.WHITE else -v
                _remove_virtual_loss(node)
                _backpropagate(node, white_value)
                sims_done += 1

        # Handle terminal nodes
        for node, white_value in terminal:
            _remove_virtual_loss(node)
            _backpropagate(node, white_value)
            sims_done += 1

        # Progress hook: fire once per batch. Cheap when None.
        if progress_callback is not None:
            progress_callback(sims_done, num_simulations)

        # Checkpoint hook: fire once per crossed threshold.
        if checkpoint_callback is not None and _remaining_checkpoints:
            while _remaining_checkpoints and sims_done >= _remaining_checkpoints[0]:
                cp = _remaining_checkpoints.pop(0)
                if root.visit_count > 0:
                    _cp_white_q = root.value_sum / root.visit_count
                    _cp_val = _cp_white_q if current == Color.WHITE else -_cp_white_q
                else:
                    _cp_val = root_value
                _cp_visits = [(ch.move, ch.visit_count) for ch in root.children]
                checkpoint_callback(cp, _cp_visits, _cp_val)

        # State hook: fire every batch with current tree snapshot for live UI.
        if state_callback is not None:
            if root.visit_count > 0:
                _st_white_q = root.value_sum / root.visit_count
                _st_val = _st_white_q if current == Color.WHITE else -_st_white_q
            else:
                _st_val = root_value
            _st_visits = [(ch.move, ch.visit_count) for ch in root.children]
            state_callback(sims_done, _st_visits, _st_val)

        # Stop hook: allow interactive early termination.
        if should_stop is not None and should_stop():
            break

    # MCTS-refined root value (from current player's perspective).
    # root.value_sum tracks backprop totals in white's perspective.
    if root.visit_count > 0:
        white_q = root.value_sum / root.visit_count
        refined_root = white_q if current == Color.WHITE else -white_q
    else:
        refined_root = root_value

    move_visits = [(ch.move, ch.visit_count) for ch in root.children]
    if return_root:
        return move_visits, refined_root, root
    return move_visits, refined_root


# ═════════════════════════════════════════════════════════════════════════════
# Self-play
# ═════════════════════════════════════════════════════════════════════════════

def self_play_game(evaluator: AlphaZeroEvaluator,
                   num_simulations: int = 800,
                   c_puct: float = 2.5,
                   dirichlet_alpha: float = 0.3,
                   noise_weight: float = 0.25,
                   temp_threshold: int = 30,
                   max_moves: int = 200):
    """
    Play one self-play game using MCTS.

    Returns:
        boards:   list of encoded board tensors (115×8×8 numpy arrays)
        policies: list of policy target vectors (length 4864)
        outcome:  1.0 (white wins), -1.0 (black wins), 0.0 (draw)
    """
    game = ExtinctionChess()
    encoder = StateEncoder()

    boards = []
    policies = []
    players = []    # track which side is to move for each position

    move_count = 0
    while not game.game_over and move_count < max_moves:
        # Encode board
        boards.append(encoder.encode_board(game))
        players.append(game.current_player)

        # MCTS search (prefer C++ if available)
        if HAS_CPP_MCTS:
            move_visits = mcts_search_cpp(game, evaluator, num_simulations,
                                          c_puct, dirichlet_alpha, noise_weight)
        else:
            move_visits, _ = mcts_search(game, evaluator, num_simulations,
                                          c_puct, dirichlet_alpha, noise_weight)
        if not move_visits:
            break

        moves, counts = zip(*move_visits)
        counts = np.array(counts, dtype=np.float64)

        # Build policy target: normalized visit counts mapped to policy indices
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        total_visits = counts.sum()
        for m, c in zip(moves, counts):
            idx = move_to_index(m)
            policy_target[idx] = c / total_visits
        policies.append(policy_target)

        # Select move
        if move_count < temp_threshold:
            # Temperature 1: sample proportional to visit counts
            probs = counts / total_visits
            idx = np.random.choice(len(moves), p=probs)
        else:
            # Temperature → 0: pick most-visited
            idx = int(np.argmax(counts))

        game.make_move(moves[idx])
        move_count += 1

    # Game outcome
    if game.winner == Color.WHITE:
        outcome = 1.0
    elif game.winner == Color.BLACK:
        outcome = -1.0
    else:
        outcome = 0.0

    return boards, policies, players, outcome


# ═════════════════════════════════════════════════════════════════════════════
# Parallel self-play
# ═════════════════════════════════════════════════════════════════════════════

def _self_play_worker(model_path: str, device: str, num_games: int,
                      num_simulations: int, temp_threshold: int,
                      result_queue: Queue):
    """Worker process: loads model, plays num_games, puts results on queue."""
    model, _ = AlphaZeroNet.load_checkpoint(model_path, migrate=True)
    model = model.to(device)
    model.eval()
    evaluator = AlphaZeroEvaluator(model, device=device)

    results = []
    for _ in range(num_games):
        boards, policies, players, outcome = self_play_game(
            evaluator, num_simulations=num_simulations,
            temp_threshold=temp_threshold,
        )
        results.append((boards, policies, players, outcome))
    result_queue.put(results)


def parallel_self_play(model_path: str, games_per_iteration: int,
                       num_simulations: int, num_workers: int = 4,
                       temp_threshold: int = 30, device: str = "cuda"):
    """Run self-play across multiple processes, each with its own model copy.

    If multiple GPUs are available, workers are distributed across them.
    """
    num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0

    # Split games across workers
    base = games_per_iteration // num_workers
    remainder = games_per_iteration % num_workers
    games_per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    result_queue = Queue()
    processes = []
    for i in range(num_workers):
        if games_per_worker[i] == 0:
            continue
        # Distribute workers across GPUs
        if num_gpus > 1:
            worker_device = f"cuda:{i % num_gpus}"
        else:
            worker_device = device
        p = Process(target=_self_play_worker,
                    args=(model_path, worker_device, games_per_worker[i],
                          num_simulations, temp_threshold, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    all_results = []
    for _ in processes:
        all_results.extend(result_queue.get())

    for p in processes:
        p.join()

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Batched self-play (C++ manages all games, Python only does GPU inference)
# ═════════════════════════════════════════════════════════════════════════════

def batched_self_play(model, device, games_per_iteration: int,
                      num_simulations: int = 800, c_puct: float = 2.5,
                      dirichlet_alpha: float = 0.3, noise_weight: float = 0.25,
                      temp_threshold: int = 30, max_moves: int = 200,
                      num_parallel: int = 50, max_batch: int = 512,
                      mcts_batch_size: int = 8, num_threads: int = 1,
                      use_tree_reuse: bool = False):
    """
    Batched self-play using C++ SelfPlayManager.

    All game logic and MCTS run in C++. Python only handles batched GPU
    inference. This maximizes GPU utilization by collecting leaf positions
    from many simultaneous games into one large batch.

    use_tree_reuse: pass through to SelfPlayManager. When True, C++ side
    calls MCTS::promote(chosen_move) after each move instead of destroying
    the tree, reusing the subtree rooted at the played child. Default False
    (fresh MCTS every move — matches pre-2026-08 behavior).

    Returns list of (boards, policies, players, outcome) tuples.
    """
    manager = CppSelfPlayManager(
        num_parallel_games=num_parallel,
        total_games=games_per_iteration,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        noise_weight=noise_weight,
        tactical_shortcuts=True,
        temp_threshold=temp_threshold,
        max_moves=max_moves,
        mcts_batch_size=mcts_batch_size,
        num_threads=num_threads,
        use_tree_reuse=use_tree_reuse,
    )

    total_evals = 0
    # Pre-allocate buffer for collect_leaves (reused every cycle)
    leaf_buf = np.zeros((max_batch, NUM_INPUT_CHANNELS, 8, 8), dtype=np.float32)

    while not manager.is_done():
        # C++ writes positions into pre-allocated buffer
        num_leaves = manager.collect_leaves(leaf_buf, max_batch)

        if num_leaves == 0:
            continue

        # Slice the filled portion (view, no copy)
        boards = leaf_buf[:num_leaves]

        # Batch GPU inference
        with torch.no_grad():
            t = torch.tensor(boards, dtype=torch.float32, device=device)
            policies, values = model(t)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        # Feed results back to C++
        manager.process_results(policies, values)
        total_evals += num_leaves

    print(f"         batched self-play: {total_evals} NN evals, "
          f"{manager.games_completed()} games", flush=True)

    # Convert C++ results to the same format as self_play_game
    raw_records = manager.get_results()
    results = []
    for rec in raw_records:
        boards = [b.reshape(NUM_INPUT_CHANNELS, 8, 8) for b in rec["boards"]]
        policies = list(rec["policies"])
        players = list(rec["players"])
        outcome = rec["outcome"]
        results.append((boards, policies, players, outcome))

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Testing utility
# ═════════════════════════════════════════════════════════════════════════════

def test_vs_random(evaluator: AlphaZeroEvaluator, num_games: int = 100,
                   num_simulations: int = 100) -> float:
    """Win rate using MCTS against a random opponent."""
    wins, losses, draws = 0, 0, 0
    for i in range(num_games):
        game = ExtinctionChess()
        model_is_white = (i % 2 == 0)
        moves = 0
        while not game.game_over and moves < 200:
            legal = game.get_legal_moves()
            if not legal:
                break
            is_model = (game.current_player == Color.WHITE) == model_is_white
            if is_model:
                if HAS_CPP_MCTS:
                    mv = mcts_search_cpp(game, evaluator, num_simulations=num_simulations,
                                         dirichlet_alpha=0, noise_weight=0,
                                         tactical_shortcuts=False)
                else:
                    mv, _ = mcts_search(game, evaluator, num_simulations=num_simulations,
                                        dirichlet_alpha=0, noise_weight=0,
                                        tactical_shortcuts=False)
                if mv:
                    best = max(mv, key=lambda x: x[1])
                    move = best[0]
                else:
                    move = random.choice(legal)
            else:
                move = random.choice(legal)
            game.make_move(move)
            moves += 1
        if game.winner:
            if (game.winner == Color.WHITE) == model_is_white:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1
    print(f"         W={wins} L={losses} D={draws}")
    return (wins + 0.5 * draws) / num_games


# ═════════════════════════════════════════════════════════════════════════════
# Instant-win position generators (supplementary training data)
# ═════════════════════════════════════════════════════════════════════════════


def generate_extra_hard_win_positions(num_positions: int, max_random_moves: int = 200):
    """Generate positions where the current player's ONLY winning moves are
    long-range (distance >= 5). Mirrors the win-taking test filter exactly:
    every winning move in the position must be dist >= 5.

    Policy target is uniform over ALL winning moves (which are all extra-hard).

    Returns (boards, policies, values) in the same format as self-play data.
    """
    boards = []
    policies = []
    values = []

    while len(boards) < num_positions:
        game = ExtinctionChess()

        for _ in range(max_random_moves):
            if game.game_over:
                break

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            current = game.current_player
            winning_moves = []
            for m in legal_moves:
                gc = _copy_game(game)
                if gc.make_move(m) and gc.game_over and gc.winner == current:
                    winning_moves.append(m)

            if winning_moves:
                # Require ALL winning moves to be dist >= 5
                all_extra_hard = True
                for m in winning_moves:
                    df = abs(m.to_pos.file - m.from_pos.file)
                    dist = max(abs(m.to_pos.rank - m.from_pos.rank), df)
                    if dist < 5:
                        all_extra_hard = False
                        break

                if all_extra_hard:
                    board = np.asarray(game.encode_board(), dtype=np.float32)
                    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                    for wm in winning_moves:
                        policy[move_to_index(wm)] = 1.0
                    policy /= policy.sum()
                    value = 1.0 if current == Color.WHITE else -1.0

                    boards.append(board)
                    policies.append(policy)
                    values.append(value)
                    break

            move = random.choice(legal_moves)
            game.make_move(move)

    return boards[:num_positions], policies[:num_positions], values[:num_positions]


def generate_hard_win_positions(num_positions: int, max_random_moves: int = 200):
    """Generate positions where the current player has an instant win involving
    a 'hard' capture: backward, sideways, or long-range (distance >= 4).

    Plays random moves until a position with at least one qualifying winning
    capture is found. Policy target is uniform over ALL winning moves in the
    position (not just qualifying ones).

    Returns (boards, policies, values) in the same format as self-play data.
    """
    boards = []
    policies = []
    values = []

    while len(boards) < num_positions:
        game = ExtinctionChess()

        for _ in range(max_random_moves):
            if game.game_over:
                break

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # Check if current player has any instant wins
            current = game.current_player
            winning_moves = []
            for m in legal_moves:
                gc = _copy_game(game)
                if gc.make_move(m) and gc.game_over and gc.winner == current:
                    winning_moves.append(m)

            if winning_moves:
                # Check if any winning move qualifies as "hard"
                has_hard = False
                for m in winning_moves:
                    dr = m.to_pos.rank - m.from_pos.rank
                    if current == Color.BLACK:
                        dr = -dr
                    df = abs(m.to_pos.file - m.from_pos.file)
                    dist = max(abs(m.to_pos.rank - m.from_pos.rank), df)

                    if dist >= 4 or dr <= 0:  # long-range, sideways, or backward
                        has_hard = True
                        break

                if has_hard:
                    board = np.asarray(game.encode_board(), dtype=np.float32)

                    # Policy: uniform over ALL winning moves
                    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                    for wm in winning_moves:
                        policy[move_to_index(wm)] = 1.0
                    policy /= policy.sum()

                    # Value: +1 from current player's perspective
                    value = 1.0 if current == Color.WHITE else -1.0

                    boards.append(board)
                    policies.append(policy)
                    values.append(value)
                    break  # Start a new game

            # No qualifying win — make a random move
            move = random.choice(legal_moves)
            game.make_move(move)

    return boards[:num_positions], policies[:num_positions], values[:num_positions]

def _copy_game(game):
    """Copy a C++ game object (can't use deepcopy)."""
    gc = ExtinctionChess()
    gc.board = game.board.copy()
    gc.current_player = game.current_player
    gc.game_over = game.game_over
    gc.winner = game.winner
    return gc


def generate_instant_win_positions(num_positions: int, max_random_moves: int = 200):
    """Generate positions where the current player has an instant win.

    Makes random moves until a position with a winning capture appears.
    Returns (boards, policies, values) in the same format as self-play data.
    """
    boards = []
    policies = []
    values = []

    while len(boards) < num_positions:
        game = ExtinctionChess()

        for _ in range(max_random_moves):
            if game.game_over:
                break

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # Check if current player has any instant wins
            winning_moves = []
            for m in legal_moves:
                gc = _copy_game(game)
                if gc.make_move(m) and gc.game_over and gc.winner == game.current_player:
                    winning_moves.append(m)

            if winning_moves:
                # Found a position with instant win(s) — record it
                board = np.asarray(game.encode_board(), dtype=np.float32)

                # Policy: uniform over all winning moves
                policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                for wm in winning_moves:
                    policy[move_to_index(wm)] = 1.0
                policy /= policy.sum()

                # Value: +1 from WHITE's perspective
                value = 1.0 if game.current_player == Color.WHITE else -1.0

                boards.append(board)
                policies.append(policy)
                values.append(value)
                break  # Start a new game

            # No instant win — make a random move
            move = random.choice(legal_moves)
            game.make_move(move)

    return boards[:num_positions], policies[:num_positions], values[:num_positions]


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def train(
    iterations: int = 100,
    games_per_iteration: int = 100,
    num_simulations: int = 800,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    num_filters: int = 256,
    num_blocks: int = 20,
    models_dir: str = "models",
    resume: bool = True,
    eval_simulations: int = 100,
    num_workers: int = 1,
    # ── MCTS subtree reuse (C++ SelfPlayManager flag) ──
    # When True, C++ side calls MCTS::promote() after each move to reuse the
    # subtree rooted at the played child instead of destroying the tree and
    # starting fresh. Measured ~1.5x wall-time speedup on iter 930 smoke test
    # (job 586087, Aug 3 2026). See commands.txt "MCTS SUBTREE REUSE" section
    # for design + rollback procedure.
    use_tree_reuse: bool = False,
    # ── Self-play exploration (dirichlet noise at root of each self-play game) ──
    # Default values match the historic training config (dirichlet_alpha=0.3,
    # noise_weight=0.25, mirroring mcts_search's own defaults and the
    # AlphaZero paper). Raise both to force more exploration of low-prior
    # moves — needed to combat policy-head prior collapse on rare-move planes
    # (e.g. underpromotions to R/B/K on the winning promotion square).
    # Diagnostic (Aug 8 2026 tool session): with dirichlet_alpha=0.3, noise_weight=0.25
    # the R/B/K promotions never get MCTS visits at 800 sims on a forced-win
    # position because raw priors have collapsed below the UCB exploration
    # floor (see commands.txt for the mechanism + expected floor calculation).
    dirichlet_alpha: float = 0.3,
    noise_weight: float = 0.25,
    instant_win_positions: int = 0,
    hard_win_positions: int = 0,
    extra_hard_win_positions: int = 0,
    max_wall_time: float = 0,
    num_epochs: int = 5,
    drilling_epochs: int = 5,
    drilling_lr_factor: float = 0.5,
    extra_hard_epochs: int = 5,
    extra_hard_lr_factor: float = 0.1,
    replay_buffer_dir: str = None,
    replay_buffer_size: int = 1,
    # ── Decoupled helper jobs (recency injection; not stored in K-buffer) ──
    helpers_enabled: bool = False,
    helpers_per_iter: int = 2,
    # SLURM-specific config (used when helper_launcher is None)
    helper_script_path: str = None,
    helper_max_wait_seconds: int = 25,
    helper_primary_gres: str = "gpu:rtx_2080_ti:1",
    helper_primary_node: str = "delta-slurm1",
    helper_fallback_gres: str = "gpu:rtx_3090:1",
    helper_fallback_node: str = "trpro-slurm1",
    # Backend-agnostic launcher: callable (iter_num, helper_id, helper_dir) ->
    # (handle, output_path). If provided, used instead of SLURM. Lets us
    # plug in Modal, k8s, or any other concurrency backend.
    helper_launcher=None,
    # ── Benchmark battery (fire-and-forget after milestone checkpoints) ──
    # When True, after each multiple-of-10 checkpoint save, main sbatches a
    # 13-test benchmark battery on trpro-slurm2 (RTX 4090). Non-blocking —
    # main continues to next iter immediately. Aggregator writes a summary
    # to benchmark_results_dir/iter_<N>.txt when all tests complete.
    # Only meaningful on the cluster; Modal runs should pass False.
    benchmark_enabled: bool = False,
    benchmark_dir: str = "~/benchmark_battery",
    benchmark_results_dir: str = "~/extinction-chess/benchmark_results",
    benchmark_aggregator_script: str = "~/extinction-chess/src/aggregate_benchmark.py",
    benchmark_src_cwd: str = "~/extinction-chess/src",
):
    job_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"AlphaZero config: {num_blocks} blocks, {num_filters} filters, "
          f"{num_simulations} MCTS sims/move")

    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "az_latest.pt")

    # Load or create model
    if resume and os.path.exists(checkpoint_path):
        model, meta = AlphaZeroNet.load_checkpoint(checkpoint_path, migrate=True)
        start_iter = meta.get("iteration", 0)
        print(f"Resumed from iteration {start_iter}")
    else:
        model = AlphaZeroNet(in_channels=NUM_INPUT_CHANNELS, num_filters=num_filters,
                             num_blocks=num_blocks)
        start_iter = 0
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Fresh AlphaZeroNet: {total_params:,} parameters")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    evaluator = AlphaZeroEvaluator(model, device=str(device))

    best_win_rate = 0.0
    training_log = []

    for iteration in range(start_iter, start_iter + iterations):
        t0 = time.time()
        iter_num = iteration + 1

        # ── Wall-time check: exit cleanly before starting expensive self-play ──
        if max_wall_time > 0 and iter_num > start_iter + 1:
            elapsed = t0 - job_start_time
            remaining = max_wall_time - elapsed
            iters_done = iter_num - start_iter - 1
            avg_gen_time = (t0 - job_start_time) / iters_done
            needed = avg_gen_time + 1000
            if remaining < needed:
                print(f"         Wall time check: {remaining:.0f}s remaining, "
                      f"need ~{needed:.0f}s (avg {avg_gen_time:.0f}s + 1000s buffer), "
                      f"stopping before iter {iter_num}")
                break

        # ── Launch decoupled helper jobs (if enabled) ───────────────────────
        helper_specs = []
        if helpers_enabled and replay_buffer_dir:
            cleanup_stale_helper_files(iter_num, replay_buffer_dir)
            print(f"         [helper] launching {helpers_per_iter} helper job(s) "
                  f"for iter {iter_num}")
            if helper_launcher is not None:
                # Custom backend (e.g. Modal). Callable returns (handle, output_path).
                for hid in range(helpers_per_iter):
                    helper_specs.append(helper_launcher(iter_num, hid, replay_buffer_dir))
            elif helper_script_path:
                # SLURM backend (default for cluster)
                for hid in range(helpers_per_iter):
                    helper_specs.append(launch_helper_with_fallback(
                        iter_num, hid, helper_script_path, replay_buffer_dir,
                        max_wait_seconds=helper_max_wait_seconds,
                        primary_gres=helper_primary_gres,
                        primary_node=helper_primary_node,
                        fallback_gres=helper_fallback_gres,
                        fallback_node=helper_fallback_node,
                    ))
            else:
                print(f"         [helper] helpers_enabled but no launcher configured; skipping")
        helper_games_consumed = 0  # populated after self-play, used in training_log

        model.eval()

        # ── Self-play ───────────────────────────────────────────────────────
        all_boards = []
        all_policies = []
        all_values = []
        # Pre-terminal positions: last position of each decisive game
        terminal_boards = []
        terminal_policies = []
        terminal_values = []
        wins_w, wins_b, draws = 0, 0, 0

        if HAS_CPP_SELFPLAY:
            # Batched self-play: C++ manages all games, Python does GPU inference
            game_results = batched_self_play(
                model, device, games_per_iteration,
                num_simulations=num_simulations,
                dirichlet_alpha=dirichlet_alpha,
                noise_weight=noise_weight,
                temp_threshold=30,
                num_parallel=min(50, games_per_iteration),
                max_batch=512,
                num_threads=4,
                use_tree_reuse=use_tree_reuse,
            )
            for boards, policies, players, outcome in game_results:
                for b, pi, player in zip(boards, policies, players):
                    value = outcome if player == 0 else -outcome
                    all_boards.append(b)
                    all_policies.append(pi)
                    all_values.append(value)
                if outcome != 0 and len(boards) > 0:
                    terminal_boards.append(boards[-1])
                    terminal_policies.append(policies[-1])
                    tv = outcome if players[-1] == 0 else -outcome
                    terminal_values.append(tv)
                if outcome > 0: wins_w += 1
                elif outcome < 0: wins_b += 1
                else: draws += 1
        elif num_workers > 1:
            # Save current model for workers to load
            tmp_path = os.path.join(models_dir, "az_tmp_selfplay.pt")
            model.save_checkpoint(tmp_path, iteration=iter_num)
            game_results = parallel_self_play(
                tmp_path, games_per_iteration,
                num_simulations=num_simulations,
                num_workers=num_workers,
                device=str(device),
            )
            for boards, policies, players, outcome in game_results:
                for b, pi, player in zip(boards, policies, players):
                    value = outcome if player == Color.WHITE else -outcome
                    all_boards.append(b)
                    all_policies.append(pi)
                    all_values.append(value)
                if outcome != 0 and len(boards) > 0:
                    terminal_boards.append(boards[-1])
                    terminal_policies.append(policies[-1])
                    tv = outcome if players[-1] == Color.WHITE else -outcome
                    terminal_values.append(tv)
                if outcome > 0: wins_w += 1
                elif outcome < 0: wins_b += 1
                else: draws += 1
        else:
            for g in range(games_per_iteration):
                boards, policies, players, outcome = self_play_game(
                    evaluator,
                    num_simulations=num_simulations,
                    temp_threshold=30,
                )
                for b, pi, player in zip(boards, policies, players):
                    value = outcome if player == Color.WHITE else -outcome
                    all_boards.append(b)
                    all_policies.append(pi)
                    all_values.append(value)
                if outcome != 0 and len(boards) > 0:
                    terminal_boards.append(boards[-1])
                    terminal_policies.append(policies[-1])
                    tv = outcome if players[-1] == Color.WHITE else -outcome
                    terminal_values.append(tv)

                if outcome > 0: wins_w += 1
                elif outcome < 0: wins_b += 1
                else: draws += 1

        gen_time = time.time() - t0
        elapsed = time.time() - job_start_time
        eh, em, es = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
        print(f"[iter {iter_num}] W={wins_w} B={wins_b} D={draws} "
              f"| {len(all_boards)} positions | gen={gen_time:.1f}s "
              f"(total={eh}:{em:02d}:{es:02d})")

        # ── Persist this iter's games to replay buffer dir ──
        if replay_buffer_dir is not None and all_boards:
            t_rb = time.time()
            os.makedirs(replay_buffer_dir, exist_ok=True)
            iter_path = os.path.join(replay_buffer_dir, f"iter_{iter_num}.npz")
            atomic_savez_compressed(
                iter_path,
                boards=np.array(all_boards, dtype=np.uint8),
                policies=np.array(all_policies, dtype=np.float32),
                values=np.array(all_values, dtype=np.float32),
            )
            # Cleanup files older than replay_buffer_size iterations. Uses
            # the shared _replay_iter_num() helper so Modal-tagged files
            # (iter_<N>_modalA1.npz etc.) get deleted alongside iter_<N>.npz.
            cutoff = iter_num - replay_buffer_size + 1
            for fname in os.listdir(replay_buffer_dir):
                file_iter = _replay_iter_num(fname)
                if file_iter is not None and file_iter < cutoff:
                    try:
                        os.remove(os.path.join(replay_buffer_dir, fname))
                    except OSError:
                        pass
            rb_time = time.time() - t_rb
            print(f"         replay buffer: wrote iter_{iter_num}.npz "
                  f"(K={replay_buffer_size}) | {rb_time:.1f}s")

            # Phase 2: load buffer contents for training (replaces current iter's data)
            if replay_buffer_size > 1:
                t_rb_load = time.time()
                # Accept cluster's own 'iter_<N>.npz' and Modal helper
                # variants like 'iter_<N>_modalA1.npz'. See _REPLAY_ITER_RE.
                # Files sorted by (iter_num, filename) for stable, reproducible
                # concatenation order across runs.
                files = sorted(
                    [f for f in os.listdir(replay_buffer_dir)
                     if _replay_iter_num(f) is not None],
                    key=lambda f: (_replay_iter_num(f), f),
                )
                buf_boards, buf_policies, buf_values = [], [], []
                skipped = 0
                # Track positions by source tag so we can log where the
                # training data came from (cluster-own, modal helpers,
                # elite manual runs).
                positions_by_source = {"cluster": 0, "elite": 0, "modal": 0,
                                       "other": 0}
                for fname in files:
                    # Use context manager so the NpzFile zip handle closes
                    # immediately — needed for Modal Volume reload to work
                    # since it refuses to refresh while files are still open.
                    # try/except keeps training alive if a single file is
                    # corrupt (partial write, filesystem hiccup, bad Modal
                    # delivery). Warning is logged loudly so real corruption
                    # isn't masked; run just uses whatever loaded OK.
                    try:
                        with np.load(os.path.join(replay_buffer_dir, fname)) as data:
                            b = data["boards"].astype(np.float32)
                            buf_boards.append(b)
                            buf_policies.append(np.asarray(data["policies"]))
                            buf_values.append(np.asarray(data["values"]))
                            # Classify source by filename tag for logging.
                            tag = _replay_file_tag(fname)
                            if tag is None:
                                positions_by_source["cluster"] += len(b)
                            elif tag.startswith("elite"):
                                positions_by_source["elite"] += len(b)
                            elif tag.startswith("modal"):
                                positions_by_source["modal"] += len(b)
                            else:
                                positions_by_source["other"] += len(b)
                    except Exception as e:
                        print(f"         [replay buffer] WARNING: skipped "
                              f"{fname}: {type(e).__name__}: {e}", flush=True)
                        skipped += 1
                if buf_boards:
                    all_boards = np.concatenate(buf_boards)
                    all_policies = np.concatenate(buf_policies)
                    all_values = np.concatenate(buf_values)
                else:
                    # All files failed — extremely rare. Fall back to the
                    # in-memory current-iter data (already in all_boards/etc.)
                    # rather than crash.
                    print(f"         [replay buffer] WARNING: all files "
                          f"skipped, falling back to current iter's data only",
                          flush=True)
                rb_load_time = time.time() - t_rb_load
                loaded_files = len(files) - skipped
                unique_iters = len({_replay_iter_num(f) for f in files})
                if loaded_files == unique_iters:
                    # Standard case (one file per iter) — preserve original
                    # log format for backward compat with history parsing.
                    summary = f"loaded {unique_iters} iters"
                else:
                    summary = f"loaded {loaded_files} files ({unique_iters} iters)"
                if skipped:
                    summary += f", SKIPPED {skipped}"
                # Append per-source breakdown only when non-cluster contributions
                # exist. Preserves original single-number log format for the
                # default case (no elite / no modal helper data).
                external_sources = {k: v for k, v in positions_by_source.items()
                                    if k != "cluster" and v > 0}
                if external_sources:
                    parts = [f"{positions_by_source['cluster']} cluster"]
                    for k in ("elite", "modal", "other"):
                        if positions_by_source[k] > 0:
                            parts.append(f"{positions_by_source[k]} {k}")
                    breakdown = " + ".join(parts)
                    positions_str = (f"{len(all_boards)} positions total "
                                     f"({breakdown})")
                else:
                    positions_str = f"{len(all_boards)} positions total"
                print(f"         replay buffer: {summary}, "
                      f"{positions_str} | {rb_load_time:.1f}s")

        # ── Consume helper games (recency injection, NOT in K-buffer) ───────
        if helpers_enabled and helper_specs:
            h_b, h_p, h_v, hgc = consume_helper_files(helper_specs)
            if h_b is not None:
                all_boards = np.concatenate([all_boards, h_b])
                all_policies = np.concatenate([all_policies, h_p])
                all_values = np.concatenate([all_values, h_v])
                helper_games_consumed = hgc
                print(f"         [helper] +{hgc} games merged, "
                      f"{len(all_boards)} positions total for training")
            else:
                print(f"         [helper] no helper data this iter")

        # ── Supplementary instant-win positions (iters 270-280) ────────────
        if instant_win_positions > 0 and 270 <= iter_num <= 280:
            iw_boards, iw_policies, iw_values = generate_instant_win_positions(
                instant_win_positions
            )
            all_boards.extend(iw_boards)
            all_policies.extend(iw_policies)
            all_values.extend(iw_values)
            print(f"         +{len(iw_boards)} instant-win positions "
                  f"({len(all_boards)} total)")

        # ── Training ────────────────────────────────────────────────────────
        t1 = time.time()
        model.train()

        X = torch.tensor(np.array(all_boards), dtype=torch.float32, device=device)
        pi_target = torch.tensor(np.array(all_policies), dtype=torch.float32, device=device)
        v_target = torch.tensor(np.array(all_values), dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(X, pi_target, v_target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss, total_ploss, total_vloss, n_batches = 0, 0, 0, 0
        for _epoch in range(num_epochs):
            for bx, bpi, bv in loader:
                optimizer.zero_grad()
                pred_p, pred_v = model(bx)

                # Policy loss: cross-entropy with MCTS visit distribution
                # -sum(pi * log(softmax(pred_p)))
                log_probs = F.log_softmax(pred_p, dim=1)
                policy_loss = -torch.sum(bpi * log_probs, dim=1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(pred_v, bv)

                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                n_batches += 1

        train_time = time.time() - t1
        avg_loss = total_loss / max(n_batches, 1)
        avg_pl = total_ploss / max(n_batches, 1)
        avg_vl = total_vloss / max(n_batches, 1)
        print(f"         loss={avg_loss:.4f} (policy={avg_pl:.4f} value={avg_vl:.4f}) "
              f"| train={train_time:.1f}s")

        # ── Terminal position drilling ──────────────────────────────────
        # Add synthetic hard-capture positions to the drilling pool
        if hard_win_positions > 0:
            t_hw = time.time()
            hw_boards, hw_policies, hw_values = generate_hard_win_positions(
                hard_win_positions
            )
            terminal_boards.extend(hw_boards)
            terminal_policies.extend(hw_policies)
            terminal_values.extend(hw_values)
            hw_time = time.time() - t_hw
            print(f"         +{len(hw_boards)} hard-capture positions "
                  f"({len(terminal_boards)} total drilling) | {hw_time:.1f}s")

        # Generate extra-hard positions separately (not added to main drilling pool)
        eh_boards, eh_policies, eh_values = [], [], []
        if extra_hard_win_positions > 0:
            t_eh = time.time()
            eh_boards, eh_policies, eh_values = generate_extra_hard_win_positions(
                extra_hard_win_positions
            )
            eh_time = time.time() - t_eh
            print(f"         +{len(eh_boards)} extra-hard positions "
                  f"(separate phase, {extra_hard_lr_factor}x LR) | {eh_time:.1f}s")

        if terminal_boards:
            t2 = time.time()
            tX = torch.tensor(np.array(terminal_boards),
                              dtype=torch.float32, device=device)
            tpi = torch.tensor(np.array(terminal_policies),
                               dtype=torch.float32, device=device)
            tv = torch.tensor(np.array(terminal_values),
                              dtype=torch.float32, device=device)

            t_dataset = torch.utils.data.TensorDataset(tX, tpi, tv)
            t_loader = torch.utils.data.DataLoader(
                t_dataset, batch_size=batch_size, shuffle=True)

            # Reduce learning rate for drilling
            orig_lr = optimizer.param_groups[0]['lr']
            for pg in optimizer.param_groups:
                pg['lr'] = orig_lr * drilling_lr_factor

            t_loss, t_ploss, t_vloss, t_nb = 0, 0, 0, 0
            for _epoch in range(drilling_epochs):
                for bx, bpi, bv in t_loader:
                    optimizer.zero_grad()
                    pred_p, pred_v = model(bx)
                    log_probs = F.log_softmax(pred_p, dim=1)
                    policy_loss = -torch.sum(bpi * log_probs, dim=1).mean()
                    value_loss = F.mse_loss(pred_v, bv)
                    loss = policy_loss + value_loss
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()
                    t_ploss += policy_loss.item()
                    t_vloss += value_loss.item()
                    t_nb += 1

            # Restore learning rate
            for pg in optimizer.param_groups:
                pg['lr'] = orig_lr

            t_time = time.time() - t2
            t_avg = t_loss / max(t_nb, 1)
            t_avgp = t_ploss / max(t_nb, 1)
            t_avgv = t_vloss / max(t_nb, 1)
            print(f"         terminal drilling: {len(terminal_boards)} positions, "
                  f"loss={t_avg:.4f} (p={t_avgp:.4f} v={t_avgv:.4f}) "
                  f"| {t_time:.1f}s")

        # ── Extra-hard drilling (separate phase, gentler LR) ────────────────
        if eh_boards:
            t3 = time.time()
            ehX = torch.tensor(np.array(eh_boards),
                               dtype=torch.float32, device=device)
            ehpi = torch.tensor(np.array(eh_policies),
                                dtype=torch.float32, device=device)
            ehv = torch.tensor(np.array(eh_values),
                               dtype=torch.float32, device=device)

            eh_dataset = torch.utils.data.TensorDataset(ehX, ehpi, ehv)
            eh_loader = torch.utils.data.DataLoader(
                eh_dataset, batch_size=batch_size, shuffle=True)

            orig_lr = optimizer.param_groups[0]['lr']
            for pg in optimizer.param_groups:
                pg['lr'] = orig_lr * extra_hard_lr_factor

            eh_loss, eh_ploss, eh_vloss, eh_nb = 0, 0, 0, 0
            for _epoch in range(extra_hard_epochs):
                for bx, bpi, bv in eh_loader:
                    optimizer.zero_grad()
                    pred_p, pred_v = model(bx)
                    log_probs = F.log_softmax(pred_p, dim=1)
                    policy_loss = -torch.sum(bpi * log_probs, dim=1).mean()
                    value_loss = F.mse_loss(pred_v, bv)
                    loss = policy_loss + value_loss
                    loss.backward()
                    optimizer.step()
                    eh_loss += loss.item()
                    eh_ploss += policy_loss.item()
                    eh_vloss += value_loss.item()
                    eh_nb += 1

            for pg in optimizer.param_groups:
                pg['lr'] = orig_lr

            eh_time = time.time() - t3
            eh_avg = eh_loss / max(eh_nb, 1)
            eh_avgp = eh_ploss / max(eh_nb, 1)
            eh_avgv = eh_vloss / max(eh_nb, 1)
            print(f"         extra-hard drilling: {len(eh_boards)} positions, "
                  f"loss={eh_avg:.4f} (p={eh_avgp:.4f} v={eh_avgv:.4f}) "
                  f"| {eh_time:.1f}s")

        # Save checkpoint
        model.save_checkpoint(checkpoint_path, iteration=iter_num)

        training_log.append({
            "iteration": iter_num,
            "loss": avg_loss,
            "policy_loss": avg_pl,
            "value_loss": avg_vl,
            "wins_white": wins_w,
            "wins_black": wins_b,
            "draws": draws,
            "gen_time": gen_time,
            "train_time": train_time,
            "helper_games": helper_games_consumed,
        })

        # ── Evaluate every 10 iterations ────────────────────────────────────
        if iter_num % 10 == 0:
            model.eval()
            eval_evaluator = AlphaZeroEvaluator(model, device=str(device))
            wr = test_vs_random(eval_evaluator, num_games=50,
                                num_simulations=eval_simulations)
            print(f"         win rate vs random: {wr:.1%}")

            versioned = os.path.join(models_dir,
                f"az_iter_{iter_num}_{int(wr*100)}pct.pt")
            model.save_checkpoint(versioned, iteration=iter_num, win_rate=wr)

            if wr > best_win_rate:
                best_win_rate = wr
                model.save_checkpoint(
                    os.path.join(models_dir, "az_best.pt"),
                    iteration=iter_num, win_rate=wr)
                print(f"         ★ new best: {wr:.1%}")

            if benchmark_enabled:
                try:
                    launch_benchmark_battery(
                        iter_num, models_dir,
                        benchmark_dir=benchmark_dir,
                        results_dir=benchmark_results_dir,
                        aggregator_script=benchmark_aggregator_script,
                        src_cwd=benchmark_src_cwd,
                    )
                except Exception as e:
                    print(f"         [benchmark] battery launch failed: {e}")

    # Final save
    model.eval()
    final_evaluator = AlphaZeroEvaluator(model, device=str(device))
    final_wr = test_vs_random(final_evaluator, num_games=100,
                              num_simulations=eval_simulations)
    print(f"\nFinal win rate vs random: {final_wr:.1%}")

    model.save_checkpoint(
        os.path.join(models_dir, "az_final.pt"),
        iteration=start_iter + iterations, win_rate=final_wr)

    log_path = os.path.join(models_dir, "az_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Done — {start_iter + iterations} iterations, best={best_win_rate:.1%}")
