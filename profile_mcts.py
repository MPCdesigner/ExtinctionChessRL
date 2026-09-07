"""
Where does MCTS wall time actually go?

Measured on the same GPU the web demo uses, so the numbers are directly
comparable to the ~66 sims/sec observed over the WebSocket.

    modal run profile_mcts.py
"""

import modal

app = modal.App("extinction-chess-profile")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir("src", remote_path="/root/src",
                   ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p))
    .add_local_dir("tools", remote_path="/root/tools",
                   ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p))
)

models_volume = modal.Volume.from_name("extinction-chess-models")
MODELS_DIR = "/models"


@app.function(image=image, gpu="L4", volumes={MODELS_DIR: models_volume}, timeout=1800)
def profile(checkpoint: str = "az_iter1040.pt"):
    import sys, os, time
    sys.path.insert(0, "/root/src")

    import torch
    import alphazero as az
    from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search
    from extinction_chess import ExtinctionChess

    dev = "cuda"
    net, meta = AlphaZeroNet.load_checkpoint(os.path.join(MODELS_DIR, checkpoint),
                                             migrate=True)
    net = net.to(dev).eval()
    ev = AlphaZeroEvaluator(net, device=dev)
    print(f"iter {meta.get('iteration')} | BATCH_SIZE_MCTS = {az.BATCH_SIZE_MCTS}\n")

    def timeit(fn, n, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    # ── 1. Raw network forward, isolated from everything else ───────────────
    print("RAW NETWORK FORWARD (pre-encoded tensor, no game logic)")
    for bs in (1, 8, 16, 32, 64, 128, 256):
        x = torch.zeros(bs, 115, 8, 8, device=dev)
        with torch.no_grad():
            dt = timeit(lambda: net(x), 20)
        print(f"  batch {bs:>3}: {dt*1000:7.2f} ms  ->  {bs/dt:8.0f} evals/sec")

    # ── 2. The evaluator path MCTS actually calls (encoding + forward) ──────
    print("\nEVALUATOR PATH (encode game states + forward) — what MCTS calls")
    g = ExtinctionChess()
    for bs in (1, 8, 32, 64):
        games = [g] * bs
        dt = timeit(lambda: ev.batch_evaluate_with_policy(games), 20)
        print(f"  batch {bs:>3}: {dt*1000:7.2f} ms  ->  {bs/dt:8.0f} evals/sec")

    # ── 3. Pure-Python game logic ───────────────────────────────────────────
    print("\nGAME LOGIC (pure Python, per call)")
    dt = timeit(lambda: g.get_legal_moves(), 200, warmup=10)
    print(f"  get_legal_moves():  {dt*1000:6.3f} ms  ({len(g.get_legal_moves())} moves)")

    # ── 4. End-to-end MCTS, with NN time isolated by wrapping the evaluator ─
    print("\nEND-TO-END MCTS (NN time separated from tree work)")
    orig = ev.batch_evaluate_with_policy
    stats = {"t": 0.0, "calls": 0, "leaves": 0}

    def wrapped(games_batch):
        t0 = time.perf_counter()
        r = orig(games_batch)
        torch.cuda.synchronize()
        stats["t"] += time.perf_counter() - t0
        stats["calls"] += 1
        stats["leaves"] += len(games_batch)
        return r

    ev.batch_evaluate_with_policy = wrapped

    for sims in (200, 800):
        stats.update(t=0.0, calls=0, leaves=0)
        fresh = ExtinctionChess()
        t0 = time.perf_counter()
        mcts_search(fresh, ev, num_simulations=sims, c_puct=2.5,
                    dirichlet_alpha=0.0, noise_weight=0.0,
                    tactical_shortcuts=True)
        total = time.perf_counter() - t0
        nn, tree = stats["t"], total - stats["t"]
        print(f"\n  {sims} sims in {total:.2f}s  ->  {sims/total:.0f} sims/sec")
        print(f"    neural net : {nn:6.2f}s ({nn/total*100:4.1f}%)  "
              f"{stats['calls']} calls, avg leaves/call {stats['leaves']/max(1,stats['calls']):.1f}")
        print(f"    tree/Python: {tree:6.2f}s ({tree/total*100:4.1f}%)")

    ev.batch_evaluate_with_policy = orig

    # ── 5. Does a bigger leaf batch help? ───────────────────────────────────
    print("\nEFFECT OF BATCH_SIZE_MCTS")
    for bs in (8, 16, 32, 64):
        az.BATCH_SIZE_MCTS = bs
        fresh = ExtinctionChess()
        t0 = time.perf_counter()
        mcts_search(fresh, ev, num_simulations=400, c_puct=2.5,
                    dirichlet_alpha=0.0, noise_weight=0.0, tactical_shortcuts=True)
        dt = time.perf_counter() - t0
        print(f"  BATCH_SIZE_MCTS={bs:>3}: 400 sims in {dt:5.2f}s -> {400/dt:5.0f} sims/sec")


@app.function(image=image, gpu="L4", volumes={MODELS_DIR: models_volume}, timeout=1800)
def hotspots(checkpoint: str = "az_iter1040.pt", sims: int = 400):
    """Function-level breakdown of the 90% that isn't the neural net."""
    import sys, os, cProfile, pstats, io
    sys.path.insert(0, "/root/src")

    import torch
    from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search
    from extinction_chess import ExtinctionChess

    net, _ = AlphaZeroNet.load_checkpoint(os.path.join(MODELS_DIR, checkpoint),
                                          migrate=True)
    net = net.to("cuda").eval()
    ev = AlphaZeroEvaluator(net, device="cuda")

    g = ExtinctionChess()
    mcts_search(g, ev, num_simulations=32, c_puct=2.5, dirichlet_alpha=0.0,
                noise_weight=0.0, tactical_shortcuts=True)   # warm up

    pr = cProfile.Profile()
    fresh = ExtinctionChess()
    pr.enable()
    mcts_search(fresh, ev, num_simulations=sims, c_puct=2.5,
                dirichlet_alpha=0.0, noise_weight=0.0, tactical_shortcuts=True)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22)
    print(s.getvalue())

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(18)
    print("\n=== BY SELF TIME (where CPU cycles actually go) ===")
    print(s2.getvalue())


@app.local_entrypoint()
def main(mode: str = "profile"):
    if mode == "hotspots":
        hotspots.remote()
    else:
        profile.remote()
