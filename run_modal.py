"""
[ARCHIVED] Run AlphaZero Extinction Chess training on Modal.
Not currently in use — training moved to WATonomous SLURM cluster.
Kept as a backup in case cluster access is lost.

Usage:
    pip install modal
    modal setup          # authenticate (one-time)
    modal run run_modal.py
"""

import modal

# ── Modal image: install deps + build C++ engine ───────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("g++")
    .pip_install("torch", "numpy", "pybind11")
    .add_local_dir("src", remote_path="/app/src", copy=True,
                   ignore=[".venv", "venv", "__pycache__", "build", "models", "*.pyc", "*.so", "*.pyd"])
    .run_commands("cd /app/src && python setup.py build_ext --inplace")
)

app = modal.App("extinction-chess-alphazero", image=image)

# Persistent volume to save checkpoints across runs
vol = modal.Volume.from_name("extinction-chess-models", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=24 * 3600,         # 24 hours max
    volumes={"/app/models": vol},
)
def train_alphazero(
    iterations: int = 100,
    games_per_iteration: int = 200,
    num_simulations: int = 400,
):
    import sys
    sys.path.insert(0, "/app/src")
    from alphazero import train
    train(
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        num_simulations=num_simulations,
        models_dir="/app/models",
        resume=True,
    )

    # Commit volume so checkpoints persist
    vol.commit()


@app.function(
    volumes={"/app/models": vol},
)
def download_model():
    """Download the best model checkpoint."""
    import os
    best = "/app/models/az_best.pt"
    latest = "/app/models/az_latest.pt"
    path = best if os.path.exists(best) else latest
    if not os.path.exists(path):
        print("No model found yet.")
        return None
    with open(path, "rb") as f:
        data = f.read()
    print(f"Downloaded {path} ({len(data)/1024:.0f} KB)")
    return data


@app.local_entrypoint()
def main():
    train_alphazero.remote(
        iterations=100,
        games_per_iteration=200,
        num_simulations=400,
    )


@app.local_entrypoint()
def download():
    import os
    data = download_model.remote()
    if data:
        os.makedirs("models", exist_ok=True)
        with open("models/az_best.pt", "wb") as f:
            f.write(data)
        print(f"Saved to models/az_best.pt ({len(data)/1024:.0f} KB)")
