"""
Head-to-head: the C5-crippled model vs the fixed model.

Does the zeroed-plane encoder actually make the engine WEAKER, or does the
net largely ignore those planes? §15 measured input divergence (25% top-1
policy agreement); this measures whether that divergence costs games.

METHOD — the encoder is the ONLY variable
  * one process, one checkpoint, one C++ game engine, one MCTS
  * both sides search the same number of sims
  * "crippled" wraps the ENCODER (not the evaluator), so every evaluate
    path is covered, and reproduces exactly what the Python fallback
    produced: history 12-107 and castling 110-113 zeroed, and plane 114
    set to halfmove_clock/100.0 (the Python encoder wrote it; C++ doesn't)
  * PAIRED games: each random opening is played twice with colours
    swapped, so opening luck and any first-move advantage cancel
  * engines play deterministically (no root noise); diversity comes from
    the random opening plies, not from randomising the engines

    modal run match_c5.py                      # 20 games @ 800 sims
    modal run match_c5.py --games 8 --sims 400 # quicker smoke test
"""

import modal

app = modal.App("extinction-chess-c5-match")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("g++", "build-essential")
    .pip_install("torch", "numpy", "pybind11")
    .add_local_dir("src", remote_path="/root/src", copy=True,
                   ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p)
                   or str(p).endswith(".pyd") or str(p).endswith(".so"))
    .add_local_dir("tools", remote_path="/root/tools", copy=True,
                   ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p))
    .run_commands("cd /root/src && python setup.py build_ext --inplace")
)

models_volume = modal.Volume.from_name("extinction-chess-models")
MODELS_DIR = "/models"


@app.function(image=image, gpu="L4", volumes={MODELS_DIR: models_volume},
              timeout=10800)
def run_match(games: int = 20, sims: int = 800, opening_plies: int = 6,
              max_plies: int = 240, seed: int = 0,
              checkpoint: str = "az_iter1040.pt"):
    import sys, os, time, random
    sys.path.insert(0, "/root/src")
    import numpy as np
    import torch

    from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search
    from extinction_chess import ExtinctionChess, Color

    try:
        import _ext_chess  # noqa: F401
        print("game engine: C++ (_ext_chess)")
    except ImportError:
        raise SystemExit("C++ extension missing — build it or the test is invalid")

    net, meta = AlphaZeroNet.load_checkpoint(os.path.join(MODELS_DIR, checkpoint),
                                             migrate=True)
    net = net.to("cuda").eval()
    print(f"checkpoint: {checkpoint} (iter {meta.get('iteration')})\n")

    class _HideCppEncode:
        """Proxy that hides encode_board so StateEncoder falls through to its
        pure-Python branch (state_encoder.py:45 gates on hasattr). Everything
        else forwards to the real C++ game object."""
        __slots__ = ("_g",)

        def __init__(self, g):
            object.__setattr__(self, "_g", g)

        def __getattr__(self, k):
            if k == "encode_board":
                raise AttributeError(k)
            return getattr(object.__getattribute__(self, "_g"), k)

    class CrippledEncoder:
        """Runs the ACTUAL pure-Python encoder — the real C5 code path, not a
        tensor-level imitation of it. Same rules engine, same everything else."""
        def __init__(self, inner):
            self._inner = inner

        def encode_board(self, g):
            return self._inner.encode_board(_HideCppEncode(g))

        def __getattr__(self, k):
            return getattr(self._inner, k)

    ev_full = AlphaZeroEvaluator(net, device="cuda")
    ev_crip = AlphaZeroEvaluator(net, device="cuda")
    ev_crip.encoder = CrippledEncoder(ev_crip.encoder)

    # ── Sanity: does our reproduction match §15's measured divergence? ──────
    rng = random.Random(1234)
    probe = []
    # Each probe entry must be its OWN game object — appending a mutated
    # object repeatedly would sample one position 200 times.
    while len(probe) < 200:
        g = ExtinctionChess()
        for _ in range(rng.randint(8, 60)):
            lm = g.get_legal_moves()
            if not lm or g.game_over:
                break
            g.make_move(rng.choice(lm))
        if not g.game_over:
            probe.append(g)   # never mutated again
    pf, vf = ev_full.batch_evaluate_with_policy(probe)
    pc, vc = ev_crip.batch_evaluate_with_policy(probe)
    top1 = float((pf.argmax(1) == pc.argmax(1)).mean())
    vdel = float(np.abs(vf - vc).mean())
    print(f"reproduction check over {len(probe)} positions:")
    print(f"  top-1 policy agreement : {top1*100:.1f}%   (§15 measured 25.0%)")
    print(f"  value |delta| mean     : {vdel:.4f}       (§15 measured 0.4176)")
    if top1 > 0.60:
        print("  ⚠ divergence much smaller than §15 — reproduction may be wrong")
    print()

    def best_move(game, ev):
        out = mcts_search(game, ev, num_simulations=sims, c_puct=2.5,
                          dirichlet_alpha=0.0, noise_weight=0.0,
                          tactical_shortcuts=True)
        mv = out[0] if isinstance(out, tuple) else out
        if not mv:
            return None
        return max(mv, key=lambda x: x[1])[0]

    def play(opening, full_is_white):
        g = ExtinctionChess()
        for m in opening:
            g.make_move(m)
        plies = len(opening)
        while not g.game_over and plies < max_plies:
            white_to_move = (g.current_player == Color.WHITE)
            ev = ev_full if (white_to_move == full_is_white) else ev_crip
            mv = best_move(g, ev)
            if mv is None:
                break
            g.make_move(mv)
            plies += 1
        if not g.game_over:
            return "draw", plies, "ply-cap"
        w = g.winner
        if w is None:
            return "draw", plies, (getattr(g, "draw_reason", None) or "draw")
        winner_is_white = (w == Color.WHITE)
        who = "full" if (winner_is_white == full_is_white) else "crippled"
        return who, plies, (getattr(g, "draw_reason", None) or "extinction")

    orng = random.Random(seed)
    pairs = (games + 1) // 2
    score = {"full": 0, "crippled": 0, "draw": 0}
    t0 = time.time()

    print(f"{games} games @ {sims} sims, {opening_plies} random opening plies, "
          f"paired colours\n")
    n = 0
    for p in range(pairs):
        # one random opening, replayed for both colour assignments
        og = ExtinctionChess()
        opening = []
        for _ in range(opening_plies):
            lm = og.get_legal_moves()
            if not lm or og.game_over:
                break
            mv = orng.choice(lm)
            opening.append(mv)
            og.make_move(mv)

        for full_is_white in (True, False):
            if n >= games:
                break
            who, plies, why = play(opening, full_is_white)
            score[who] += 1
            n += 1
            side = "W" if full_is_white else "B"
            print(f"  game {n:>2}/{games}  full={side}  -> {who:<8} "
                  f"({plies} plies, {why})   running: "
                  f"full {score['full']} / crip {score['crippled']} / draw {score['draw']}")

    el = time.time() - t0
    pts = score["full"] + 0.5 * score["draw"]
    print("\n" + "=" * 60)
    print(f"FULL (fixed)    : {score['full']}")
    print(f"CRIPPLED (C5)   : {score['crippled']}")
    print(f"DRAW            : {score['draw']}")
    print(f"score for FULL  : {pts}/{n}  =  {pts/n*100:.1f}%")
    print(f"elapsed {el/60:.1f} min  ({el/max(1,n):.0f}s per game)")
    print("=" * 60)
    # crude 2-sigma band on a binomial with n decisive games
    dec = score["full"] + score["crippled"]
    if dec:
        import math
        se = math.sqrt(0.25 / dec)
        print(f"decisive games: {dec}, 2-sigma on a coin flip = "
              f"±{2*se*100:.1f}% — treat anything inside that as noise")
    return score


@app.local_entrypoint()
def main(games: int = 20, sims: int = 800, seed: int = 0):
    run_match.remote(games=games, sims=sims, seed=seed)
