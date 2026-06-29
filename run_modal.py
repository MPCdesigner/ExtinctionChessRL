"""
Run AlphaZero Extinction Chess training on Modal.

Layout:
- Main training (A100 80GB) runs the full train() loop with the
  Modal-native helper_launcher passed in.
- helper_function (A10G) is spawned by main once per helper per iter,
  generates ~200 self-play games against az_latest.pt, writes an
  atomic .npz to the shared Volume, and exits.
- Volume mount: /app/state, with subdirs:
    /app/state/models/         (checkpoints)
    /app/state/replay_buffer/  (iter_N.npz + helper_v{N}_id{M}.npz)

Usage:
    pip install modal
    modal setup                   # one-time auth

    # one-time: push the cluster's current state into the Modal volume
    modal run run_modal.py::upload

    # run training (uses the latest model + buffer in the volume)
    modal run run_modal.py
"""

import modal

# ── Modal image: install deps + build C++ engine ───────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++")
    .pip_install("torch", "numpy", "pybind11")
    .add_local_dir("src", remote_path="/app/src", copy=True,
                   ignore=[".venv", "venv", "__pycache__", "build", "models",
                           "*.pyc", "*.so", "*.pyd"])
    .run_commands("cd /app/src && python setup.py build_ext --inplace")
)

app = modal.App("extinction-chess-alphazero", image=image)

# Single volume holds models + replay buffer + helper files.
vol = modal.Volume.from_name("extinction-chess-state", create_if_missing=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helper function (A10G) — one helper invocation generates ~200 games
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A10G",
    timeout=2 * 3600,
    volumes={"/app/state": vol},
    retries=0,  # Don't auto-retry — a failed helper just gets skipped by main
)
def helper_function(iter_num: int, helper_id: int,
                    num_games: int = 200,
                    num_simulations: int = 800,
                    num_threads: int = 4,
                    model_path: str = "/app/state/models/az_latest.pt",
                    output_dir: str = "/app/state/replay_buffer"):
    import sys
    sys.path.insert(0, "/app/src")
    import os
    import time
    import numpy as np
    import torch

    from alphazero import (
        AlphaZeroNet, batched_self_play, HAS_CPP_SELFPLAY,
        atomic_savez_compressed,
    )

    if not HAS_CPP_SELFPLAY:
        raise RuntimeError("Helper requires C++ self-play extension (_ext_chess).")

    # Refresh volume to see latest model written by main
    vol.reload()

    output_path = f"{output_dir}/helper_v{iter_num}_id{helper_id}.npz"

    print(f"[helper {helper_id}] Loading model from {model_path}", flush=True)
    model, meta = AlphaZeroNet.load_checkpoint(model_path, migrate=True)
    model_iter = meta.get("iteration", -1)
    print(f"[helper {helper_id}] Model iteration: {model_iter}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    t0 = time.time()
    print(f"[helper {helper_id}] Generating {num_games} games "
          f"(sims={num_simulations}, threads={num_threads})", flush=True)

    game_results = batched_self_play(
        model, device, num_games,
        num_simulations=num_simulations,
        temp_threshold=30,
        num_parallel=min(50, num_games),
        max_batch=512,
        num_threads=num_threads,
    )

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
    print(f"[helper {helper_id}] {len(game_results)} games | "
          f"{len(all_boards)} positions | W={wins_w} B={wins_b} D={draws} "
          f"| gen={gen_time:.1f}s", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    atomic_savez_compressed(
        output_path,
        boards=np.array(all_boards, dtype=np.uint8),
        policies=np.array(all_policies, dtype=np.float32),
        values=np.array(all_values, dtype=np.float32),
        num_games=np.int32(len(game_results)),
        iter_num=np.int32(model_iter),
    )
    vol.commit()  # ensure main can see the file
    print(f"[helper {helper_id}] Wrote {output_path}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# Main training function (A100 80GB)
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A10G",
    timeout=23 * 3600,
    volumes={"/app/state": vol},
    retries=0,  # Don't auto-retry — a 3h training is too expensive to silently re-run
)
def train_alphazero():
    import sys
    sys.path.insert(0, "/app/src")
    import multiprocessing
    import os

    multiprocessing.set_start_method("spawn", force=True)
    from alphazero import train

    # ── Modal helper handle: wraps FunctionCall + volume reload on wait ──
    class ModalHelperHandle:
        def __init__(self, function_call):
            self.fc = function_call

        def get(self, timeout=None):
            try:
                self.fc.get(timeout=timeout)
            except Exception as e:
                print(f"         [helper] FunctionCall.get() error: {e}",
                      flush=True)
            vol.reload()  # see fresh files written by helper

        def __repr__(self):
            return f"ModalHelperHandle({self.fc.object_id})"

    # ── Helper launcher passed to train() ──
    def modal_helper_launcher(iter_num, helper_id, helper_dir):
        output_path = f"{helper_dir}/helper_v{iter_num}_id{helper_id}.npz"
        # Make sure helpers see the latest model we just wrote
        vol.commit()
        fc = helper_function.spawn(iter_num, helper_id)
        print(f"         [helper {helper_id}] spawned Modal call "
              f"(id={fc.object_id})", flush=True)
        return (ModalHelperHandle(fc), output_path)

    os.makedirs("/app/state/models", exist_ok=True)
    os.makedirs("/app/state/replay_buffer", exist_ok=True)
    vol.reload()

    train(
        iterations=5,  # Sized to fit ~$30 budget on A10G (~$6/iter)
        games_per_iteration=400,
        num_simulations=800,
        learning_rate=0.00002,
        models_dir="/app/state/models",
        resume=True,
        num_workers=4,
        hard_win_positions=300,
        extra_hard_win_positions=0,
        max_wall_time=22 * 3600,  # exit cleanly before Modal 23h timeout
        num_epochs=3,
        drilling_epochs=5,
        drilling_lr_factor=0.5,
        extra_hard_epochs=5,
        extra_hard_lr_factor=0.025,
        replay_buffer_dir="/app/state/replay_buffer",
        replay_buffer_size=5,
        # Helpers via Modal
        helpers_enabled=True,
        helpers_per_iter=2,
        helper_launcher=modal_helper_launcher,
    )

    vol.commit()  # final flush of new checkpoint + buffer state


# ═════════════════════════════════════════════════════════════════════════════
# State upload: push local checkpoint + replay buffer into the Modal Volume
# ═════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def upload():
    """Push local models/az_latest.pt and replay_buffer/iter_*.npz into the
    Modal Volume. Run this once before starting training to seed the volume
    with the state we want to resume from."""
    import os
    import glob

    files_to_upload = []

    model_path = "models/az_latest.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found locally. SCP from cluster first.")
        return
    files_to_upload.append((model_path, "/models/az_latest.pt"))

    for path in sorted(glob.glob("replay_buffer/iter_*.npz")):
        files_to_upload.append((path, "/replay_buffer/" + os.path.basename(path)))

    if not files_to_upload:
        print("Nothing to upload.")
        return

    print(f"Uploading {len(files_to_upload)} file(s) to Modal Volume:")
    with vol.batch_upload() as batch:
        for local, remote in files_to_upload:
            size_mb = os.path.getsize(local) / 1e6
            print(f"  {local} → {remote}  ({size_mb:.1f} MB)")
            batch.put_file(local, remote)
    print("Upload complete.")


@app.local_entrypoint()
def main():
    """Run training on Modal. Assumes the volume has been seeded via upload."""
    train_alphazero.remote()


@app.function(volumes={"/app/state": vol})
def list_state():
    """List what's currently in the Modal Volume."""
    import os
    for root in ["/app/state/models", "/app/state/replay_buffer"]:
        if not os.path.isdir(root):
            print(f"{root}: (does not exist)")
            continue
        print(f"{root}:")
        for f in sorted(os.listdir(root)):
            p = os.path.join(root, f)
            sz = os.path.getsize(p) / 1e6
            print(f"  {f}  ({sz:.1f} MB)")


@app.local_entrypoint()
def list_volume():
    """Inspect the Modal Volume contents."""
    list_state.remote()


# ═════════════════════════════════════════════════════════════════════════════
# Smoke test — runs the full pipeline (main + helpers) with tiny config
# Isolated in /app/state/smoke/ so it doesn't touch production state.
# Expected cost: ~$0.30-0.50, ~10-15 minutes wall time.
# ═════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A10G",
    timeout=30 * 60,
    volumes={"/app/state": vol},
)
def smoke_test_train():
    import sys
    sys.path.insert(0, "/app/src")
    import multiprocessing
    import os
    import shutil

    multiprocessing.set_start_method("spawn", force=True)
    from alphazero import train

    smoke_models = "/app/state/smoke/models"
    smoke_buffer = "/app/state/smoke/replay_buffer"

    vol.reload()
    os.makedirs(smoke_models, exist_ok=True)
    os.makedirs(smoke_buffer, exist_ok=True)

    # Cleanup orphans from any previous failed run (e.g. iter_*.npz.tmp.npz
    # left over from a pre-fix atomic_savez bug, or *.tmp partial writes).
    for f in os.listdir(smoke_buffer):
        if (f.endswith(".tmp")
                or f.endswith(".tmp.npz")
                or (f.startswith("iter_") and f.endswith(".npz")
                    and not f[5:-4].isdigit())):
            path = f"{smoke_buffer}/{f}"
            print(f"[smoke] cleanup orphan: {f}", flush=True)
            os.remove(path)

    # Seed smoke dirs with current production state
    prod_model = "/app/state/models/az_latest.pt"
    smoke_model = f"{smoke_models}/az_latest.pt"
    if not os.path.exists(smoke_model):
        print(f"[smoke] copying {prod_model} -> {smoke_model}", flush=True)
        shutil.copy(prod_model, smoke_model)
    prod_buffer = "/app/state/replay_buffer"
    if os.path.isdir(prod_buffer):
        for f in sorted(os.listdir(prod_buffer))[-5:]:
            if f.startswith("iter_") and f.endswith(".npz"):
                dst = f"{smoke_buffer}/{f}"
                if not os.path.exists(dst):
                    print(f"[smoke] copying {f}", flush=True)
                    shutil.copy(f"{prod_buffer}/{f}", dst)
    vol.commit()

    class ModalHelperHandle:
        def __init__(self, fc):
            self.fc = fc

        def get(self, timeout=None):
            try:
                self.fc.get(timeout=timeout)
            except Exception as e:
                print(f"         [helper] FunctionCall.get() error: {e}", flush=True)
            vol.reload()

        def __repr__(self):
            return f"ModalHelperHandle({self.fc.object_id})"

    def smoke_helper_launcher(iter_num, helper_id, helper_dir):
        output_path = f"{helper_dir}/helper_v{iter_num}_id{helper_id}.npz"
        vol.commit()
        fc = helper_function.spawn(
            iter_num, helper_id,
            num_games=10,
            num_simulations=50,
            num_threads=4,
            model_path=smoke_model,
            output_dir=smoke_buffer,
        )
        print(f"         [helper {helper_id}] spawned smoke call "
              f"(id={fc.object_id})", flush=True)
        return (ModalHelperHandle(fc), output_path)

    train(
        iterations=1,
        games_per_iteration=20,
        num_simulations=50,
        learning_rate=0.00002,
        models_dir=smoke_models,
        resume=True,
        num_workers=4,
        hard_win_positions=20,
        extra_hard_win_positions=0,
        max_wall_time=25 * 60,
        num_epochs=1,
        drilling_epochs=1,
        drilling_lr_factor=0.5,
        extra_hard_epochs=1,
        extra_hard_lr_factor=0.025,
        replay_buffer_dir=smoke_buffer,
        replay_buffer_size=5,
        helpers_enabled=True,
        helpers_per_iter=2,
        helper_launcher=smoke_helper_launcher,
    )

    vol.commit()
    print("[smoke] complete", flush=True)


@app.local_entrypoint()
def smoke():
    """Run a tiny end-to-end smoke test on Modal.

    Verifies: image build, model load from Volume, main self-play,
    helper spawning + consumption, training step, checkpoint save.
    Uses isolated /app/state/smoke/ — production state untouched.
    """
    smoke_test_train.remote()
