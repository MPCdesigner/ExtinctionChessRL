"""
Is the web integration weakening the engine?

Side A = EXACTLY what web_play.py does: the threaded Engine with pondering
         and tree reuse, played out with the same sim-ceiling wait loop and
         the same legality guard.
Side B = plain mcts_search(num_simulations=N) fresh every move — no ponder,
         no tree reuse. The "reference" way to consume the engine.

Same checkpoint, same C++ rules engine, same sim budget. Paired games with
colours swapped. If A loses badly to B, the fault is in how web_play drives
the Engine, not in the engine itself.

    modal run match_webpath.py --games 10 --sims 800
"""

import modal

app = modal.App("extinction-chess-webpath")

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
def run(games: int = 10, sims: int = 800, opening_plies: int = 6,
        max_plies: int = 240, seed: int = 0, checkpoint: str = "az_iter1040.pt"):
    import sys, os, time, random
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/tools")
    import torch

    from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search
    from extinction_chess import ExtinctionChess, Color
    from play_timed.engine import Engine

    ckpt = os.path.join(MODELS_DIR, checkpoint)
    net, meta = AlphaZeroNet.load_checkpoint(ckpt, migrate=True)
    net = net.to("cuda").eval()
    ev = AlphaZeroEvaluator(net, device="cuda")
    print(f"{checkpoint} (iter {meta.get('iteration')}) | sims={sims}\n")

    engine = Engine(model_path=ckpt, device="cuda", tactical_level="basic",
                    c_puct=2.5, dirichlet_alpha=0.0, noise_weight=0.0)

    def key(m):
        return (m.from_pos.rank, m.from_pos.file, m.to_pos.rank, m.to_pos.file,
                m.promotion.name if m.promotion else None)

    # ── Side A: web_play.py's exact drive loop ───────────────────────────
    def web_move(game, budget_s=30.0):
        legal_now = {key(m) for m in game.get_legal_moves()}
        started = time.monotonic()
        deadline = started + budget_s
        result = None
        while True:
            cand = engine.get_current_result()
            if cand is not None and key(cand["move"]) in legal_now:
                result = cand
                # atomic count from the result itself (see web_play.py)
                if (cand["search_snapshot"]["sim_count"] >= sims
                        or time.monotonic() >= deadline):
                    break
            if time.monotonic() >= started + budget_s + 30:
                break
            time.sleep(0.02)
        if result is None:
            return None, 0
        return result["move"], result["search_snapshot"]["sim_count"]

    # ── Side B: plain fresh search ───────────────────────────────────────
    def plain_move(game):
        out = mcts_search(game, ev, num_simulations=sims, c_puct=2.5,
                          dirichlet_alpha=0.0, noise_weight=0.0,
                          tactical_shortcuts=True)
        mv = out[0] if isinstance(out, tuple) else out
        if not mv:
            return None, 0
        return max(mv, key=lambda x: x[1])[0], sims

    def play(opening, web_is_white):
        g = ExtinctionChess()
        engine.stop()
        for m in opening:
            g.make_move(m)
        engine.start_from(g)
        plies = len(opening)
        web_sims = []
        while not g.game_over and plies < max_plies:
            web_turn = ((g.current_player == Color.WHITE) == web_is_white)
            if web_turn:
                mv, n = web_move(g)
                if mv is None:
                    break
                web_sims.append(n)
                g.make_move(mv)
                engine.descend(mv)
            else:
                mv, _ = plain_move(g)
                if mv is None:
                    break
                g.make_move(mv)
                engine.descend(mv)   # keep the ponder tree aligned
            plies += 1
        avg = sum(web_sims) / len(web_sims) if web_sims else 0
        if not g.game_over:
            return "draw", plies, avg
        w = g.winner
        if w is None:
            return "draw", plies, avg
        who = "web" if ((w == Color.WHITE) == web_is_white) else "plain"
        return who, plies, avg

    orng = random.Random(seed)
    score = {"web": 0, "plain": 0, "draw": 0}
    n = 0
    t0 = time.time()
    for p in range((games + 1) // 2):
        og = ExtinctionChess()
        opening = []
        for _ in range(opening_plies):
            lm = og.get_legal_moves()
            if not lm or og.game_over:
                break
            mv = orng.choice(lm)
            opening.append(mv)
            og.make_move(mv)
        for web_is_white in (True, False):
            if n >= games:
                break
            who, plies, avg = play(opening, web_is_white)
            score[who] += 1
            n += 1
            print(f"  game {n:>2}/{games} web={'W' if web_is_white else 'B'} -> "
                  f"{who:<6} ({plies} plies, web avg sims={avg:.0f})  "
                  f"running web {score['web']} / plain {score['plain']} / draw {score['draw']}")

    engine.shutdown()
    pts = score["web"] + 0.5 * score["draw"]
    dec = score["web"] + score["plain"]
    print("\n" + "=" * 58)
    print(f"WEB PATH (Engine+ponder) : {score['web']}")
    print(f"PLAIN mcts_search        : {score['plain']}")
    print(f"DRAW                     : {score['draw']}")
    print(f"score for WEB PATH       : {pts}/{n} = {pts/n*100:.1f}%")
    if dec:
        import math
        print(f"2-sigma on a coin flip   : ±{2*math.sqrt(0.25/dec)*100:.1f}%")
    print(f"elapsed {(time.time()-t0)/60:.1f} min")
    print("=" * 58)
    return score


@app.function(image=image, gpu="L4", volumes={MODELS_DIR: models_volume}, timeout=5400)
def wintake(n: int = 120, sims: int = 800, checkpoint: str = "az_iter1040.pt"):
    """What does losing the root tactical shortcut actually cost?

    The reuse path skips it, so the web engine effectively runs with
    tactical_shortcuts=False. Measure win-taking (positions where at least one
    legal move ends the game immediately) with the flag ON vs OFF.
    """
    import sys, os, random, time
    sys.path.insert(0, "/root/src")
    import torch
    from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search
    from extinction_chess import ExtinctionChess

    net, _ = AlphaZeroNet.load_checkpoint(os.path.join(MODELS_DIR, checkpoint),
                                          migrate=True)
    net = net.to("cuda").eval()
    ev = AlphaZeroEvaluator(net, device="cuda")

    def replay(hist):
        g = ExtinctionChess()
        for m in hist:
            g.make_move(m)
        return g

    def winning_moves(g):
        out = []
        for m in g.get_legal_moves():
            gc = replay_from(g)
            if gc is None:
                return []
            if gc.make_move(m) and gc.game_over and gc.winner == g.current_player:
                out.append(m)
        return out

    # cheap game copy via history replay (C++ object has no guaranteed .copy)
    hist_of = {}
    def replay_from(g):
        return replay(hist_of.get(id(g), []))

    rng = random.Random(7)
    positions = []
    tries = 0
    while len(positions) < n and tries < 8000:
        tries += 1
        g = ExtinctionChess()
        hist = []
        hist_of[id(g)] = hist
        depth = rng.randint(6, 70)
        for _ in range(depth):
            lm = g.get_legal_moves()
            if not lm or g.game_over:
                break
            wm = winning_moves(g)
            if wm:
                positions.append((list(hist), [(m.from_pos.rank, m.from_pos.file,
                                               m.to_pos.rank, m.to_pos.file)
                                              for m in wm]))
                break
            mv = rng.choice(lm)
            g.make_move(mv)
            hist.append(mv)
            hist_of[id(g)] = hist
    print(f"collected {len(positions)} positions with an immediate win available\n")

    def took_win(mv, wins):
        return (mv.from_pos.rank, mv.from_pos.file, mv.to_pos.rank, mv.to_pos.file) in wins

    res = {}
    for label, flag in (("shortcuts ON  (fresh path)", True),
                        ("shortcuts OFF (reuse path)", False)):
        hit = 0
        t0 = time.time()
        for hist, wins in positions:
            g = replay(hist)
            out = mcts_search(g, ev, num_simulations=sims, c_puct=2.5,
                              dirichlet_alpha=0.0, noise_weight=0.0,
                              tactical_shortcuts=flag)
            mv = out[0] if isinstance(out, tuple) else out
            if mv and took_win(max(mv, key=lambda x: x[1])[0], wins):
                hit += 1
        res[label] = hit
        print(f"{label}: {hit}/{len(positions)} = {hit/len(positions)*100:.1f}% "
              f"({time.time()-t0:.0f}s)")

    a, b = list(res.values())
    print(f"\ncost of losing the shortcut: {(a-b)/len(positions)*100:.1f} "
          f"percentage points of win-taking")
    return res


@app.function(image=image, gpu="L4", volumes={MODELS_DIR: models_volume}, timeout=900)
def signcheck(trials: int = 12):
    """Does current_player still flip on the move that ENDS the game?

    web_play.py infers which colour just moved by reading current_player
    AFTER make_move. If the game-ending move leaves current_player on the
    mover, that inference inverts exactly on the final move — which is the
    move whose evaluation a player actually looks at.
    """
    import sys, random
    sys.path.insert(0, "/root/src")
    from extinction_chess import ExtinctionChess, Color

    rng = random.Random(3)
    seen = 0
    flipped = 0
    for t in range(400):
        g = ExtinctionChess()
        prev = None
        while not g.game_over:
            lm = g.get_legal_moves()
            if not lm:
                break
            mover = g.current_player
            g.make_move(rng.choice(lm))
            prev = mover
            if g.game_over:
                after = g.current_player
                w = g.winner
                did_flip = (after != mover)
                flipped += 1 if did_flip else 0
                seen += 1
                if seen <= 6:
                    print(f"  mover={mover.name:<5} after_move={after.name:<5} "
                          f"winner={w.name if w else None:<5} flipped={did_flip}")
                break
        if seen >= trials:
            break
    print(f"\ngame-ending moves sampled: {seen}")
    print(f"current_player flipped on: {flipped}/{seen}")
    print("VERDICT:", "flips normally — after-the-fact inference is safe"
          if flipped == seen else
          "DOES NOT FLIP on the final move — reading current_player after "
          "make_move inverts the evaluation sign exactly on game-ending moves")


@app.function(image=image, gpu="L4", volumes={MODELS_DIR: models_volume}, timeout=1800)
def noisecheck(checkpoint: str = "az_iter1040.pt", alpha: float = 1.0,
               weight: float = 0.3, chunk: int = 30, chunks: int = 67):
    """Does the ponder loop's per-chunk noise compound?

    Engine ponders in PONDER_CHUNK_SIMS=30 increments, each a fresh
    mcts_search(prev_root=...) call, and the reuse path re-applies Dirichlet
    noise every time. Measure what that does to the root priors versus a
    single application.
    """
    import sys, os, math
    sys.path.insert(0, "/root/src")
    import numpy as np, torch
    from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search
    from extinction_chess import ExtinctionChess

    net, _ = AlphaZeroNet.load_checkpoint(os.path.join(MODELS_DIR, checkpoint),
                                          migrate=True)
    net = net.to("cuda").eval()
    ev = AlphaZeroEvaluator(net, device="cuda")

    def describe(root, label):
        p = np.array([c.prior for c in root.children], dtype=np.float64)
        p = p / max(p.sum(), 1e-9)
        ent = -(p * np.log(p + 1e-12)).sum()
        print(f"  {label:34s} n={len(p):3d}  max_prior={p.max():.4f}  "
              f"top1_share={p.max():.3f}  entropy={ent:.3f}  "
              f"(uniform entropy={math.log(len(p)):.3f})")
        return p

    g = ExtinctionChess()
    for mv in list(g.get_legal_moves())[:1]:
        pass

    print(f"alpha={alpha} weight={weight}, chunk={chunk}, chunks={chunks}\n")

    # Baseline: one fresh search, noise applied exactly once
    _, _, root1 = mcts_search(g, ev, num_simulations=chunk, c_puct=3.5,
                              dirichlet_alpha=alpha, noise_weight=weight,
                              tactical_shortcuts=False, return_root=True)
    p1 = describe(root1, "after 1 chunk (noise x1)")

    # Now drive it exactly like the ponder loop does
    root = root1
    for i in range(chunks - 1):
        target = root.visit_count + chunk
        _, _, root = mcts_search(g, ev, num_simulations=target, c_puct=3.5,
                                 dirichlet_alpha=alpha, noise_weight=weight,
                                 tactical_shortcuts=False, prev_root=root,
                                 return_root=True)
    pN = describe(root, f"after {chunks} chunks (noise x{chunks})")

    # Reference: no noise at all
    _, _, root0 = mcts_search(g, ev, num_simulations=chunk, c_puct=3.5,
                              dirichlet_alpha=0.0, noise_weight=0.0,
                              tactical_shortcuts=False, return_root=True)
    p0 = describe(root0, "no noise (raw policy priors)")

    retained = (1 - weight) ** chunks
    print(f"\n  original prior weight retained after {chunks} applications: "
          f"(1-{weight})^{chunks} = {retained:.3e}")
    print(f"  correlation with raw policy: 1-chunk={np.corrcoef(p1,p0)[0,1]:.3f}  "
          f"{chunks}-chunk={np.corrcoef(pN,p0)[0,1]:.3f}")


@app.local_entrypoint()
def main(games: int = 10, sims: int = 800, mode: str = "match", n: int = 120):
    if mode == "signcheck":
        signcheck.remote()
        return
    if mode == "noisecheck":
        noisecheck.remote()
        return
    if mode == "wintake":
        wintake.remote(n=n, sims=sims)
    else:
        run.remote(games=games, sims=sims)
