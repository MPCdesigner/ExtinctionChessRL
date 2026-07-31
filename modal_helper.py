"""
Modal helper for Extinction Chess self-play generation.

Runs one batch of self-play games against a given checkpoint on Modal's GPU.
Output .npz is written to the "extinction-chess-helper-outputs" Modal
Volume, from where the cluster-side cron daemon downloads it and drops it
into ~/extinction-chess/replay_buffer/ with the name given as output_filename.

Deploy once per Modal account (both accounts share the same app name and
volume names, but their data is isolated per account):
    modal --profile henry-account-a deploy modal_helper.py
    modal --profile henry-account-b deploy modal_helper.py

Invoke (from cluster daemon or manual):
    modal --profile henry-account-a run modal_helper.py::run_helper \\
        --checkpoint-filename az_iter_920_100pct.pt \\
        --output-filename iter_920_modalA1.npz \\
        --num-games 200

Output format contract (matches src/alphazero.py replay-buffer loader):
    boards:   (N, 115, 8, 8) uint8
    policies: (N, 4864)      float32
    values:   (N,)           float32
    num_games: int32         (informational)
    iter_num:  int32         (informational — from checkpoint meta)
"""

import modal
import os
import sys
import subprocess
import time

app = modal.App("extinction-chess-helper")

# Modal image mirrors what cluster provides at helper runtime: Python 3.10,
# build tools for the C++ extension, torch/numpy/pybind11. Full src/ tree
# is uploaded so alphazero.py + state_encoder.py + cpp/ compile the same
# way as on cluster.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "g++")
    .pip_install("torch", "numpy", "pybind11")
    .add_local_dir(
        "src", "/root/src",
        ignore=lambda p: (
            ".venv" in str(p)
            or "__pycache__" in str(p)
            or str(p).endswith(".pyc")
            or "/build/" in str(p)
        ),
    )
)

ckpt_vol = modal.Volume.from_name("extinction-chess-ckpts",
                                  create_if_missing=True)
out_vol  = modal.Volume.from_name("extinction-chess-helper-outputs",
                                  create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",           # cheapest reasonable GPU for our model size
    timeout=7200,       # 2h hard limit — helper should finish in ~1-1.5h
    volumes={"/ckpts": ckpt_vol, "/out": out_vol},
)
def run_helper(
    checkpoint_filename: str,
    output_filename: str,
    num_games: int = 200,
    num_simulations: int = 800,
    num_parallel: int = 50,
    num_threads: int = 4,
):
    """Generate one batch of self-play games and write the .npz to /out.

    Mirrors src/run_helper.py's logic exactly so output is
    byte-format-compatible with cluster's replay-buffer loader.
    """
    sys.path.insert(0, "/root/src")
    os.chdir("/root/src")

    # Build C++ extension fresh each call — Modal caches the image layer but
    # not per-function build artifacts. Adds ~30s to startup; unavoidable.
    print("[modal-helper] building C++ extension...", flush=True)
    subprocess.check_call(
        [sys.executable, "setup.py", "build_ext", "--inplace"])

    # Refresh volume view in case files changed since function cold-start.
    ckpt_vol.reload()

    # Import after build so the compiled .so is available.
    import numpy as np
    import torch
    from alphazero import (
        AlphaZeroNet, HAS_CPP_SELFPLAY,
        atomic_savez_compressed, batched_self_play,
    )
    assert HAS_CPP_SELFPLAY, "C++ self-play extension failed to build"

    ckpt_path = f"/ckpts/{checkpoint_filename}"
    out_path  = f"/out/{output_filename}"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint {ckpt_path} not found on Modal Volume. "
            f"Upload it first with: modal volume put "
            f"extinction-chess-ckpts <local_path> /{checkpoint_filename}")

    print(f"[modal-helper] loading {ckpt_path}", flush=True)
    model, meta = AlphaZeroNet.load_checkpoint(ckpt_path, migrate=True)
    iter_num = meta.get("iteration", -1)
    print(f"[modal-helper] checkpoint iteration: {iter_num}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[modal-helper] device: {device}", flush=True)
    model = model.to(device)
    model.eval()

    t0 = time.time()
    print(f"[modal-helper] generating {num_games} games "
          f"(sims={num_simulations}, threads={num_threads}, "
          f"parallel={num_parallel})", flush=True)

    game_results = batched_self_play(
        model, device, num_games,
        num_simulations=num_simulations,
        temp_threshold=30,
        num_parallel=min(num_parallel, num_games),
        max_batch=512,
        num_threads=num_threads,
    )

    # Flatten to position-level arrays. Value target flips sign based on
    # which player was to move at that position — exactly matches
    # run_helper.py so cluster loader consumes it identically.
    all_boards, all_policies, all_values = [], [], []
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
    print(f"[modal-helper] {len(game_results)} games | "
          f"{len(all_boards)} positions | W={wins_w} B={wins_b} D={draws} "
          f"| gen={gen_time:.1f}s", flush=True)

    print(f"[modal-helper] writing {out_path}", flush=True)
    atomic_savez_compressed(
        out_path,
        boards=np.array(all_boards, dtype=np.uint8),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
        num_games=np.int32(len(game_results)),
        iter_num=np.int32(iter_num),
    )
    # Commit so subsequent 'modal volume get' sees the file. Without this
    # the daemon might download a stale view.
    out_vol.commit()
    print(f"[modal-helper] done.", flush=True)
