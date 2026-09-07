"""
Modal backend for the browser demo: play the extinction chess engine.

DESIGN
------
One WebSocket connection == one game == one container == one dedicated GPU.

Modal keeps a single function call alive for the lifetime of a WebSocket
connection, and without @modal.concurrent a container serves one input at a
time. So the connection *is* the session: the MCTS tree and the ponder thread
live in container memory for the whole game, and the chess clock bounds how
long that container can exist. No session store, no sticky routing, no
external web server.

The server is authoritative for rules — it ships the legal move list to the
client each turn — so the browser only renders the board and forwards clicks.

Deploy:
    modal deploy web_play.py

Dev (hot-reload, prints a temporary URL):
    modal serve web_play.py

The checkpoint must already be on the models volume:
    modal volume put extinction-chess-models models/az_iter1040.pt /az_iter1040.pt

COST
----
Container lifetime is the billing window, and pondering keeps the GPU busy
the whole time — that's the point, not waste: the engine searches during your
opponent's think time, so its own moves come out near-instantly off an already
deep tree (see briefing section 12).

A 5+3 game caps out around 14 minutes of wall clock. On L4 ($0.000222/s) that
is roughly $0.20 per game plus CPU/memory, so ~$30 of credit covers well over
a hundred full games. MAX_CONCURRENT_GAMES below is the hard ceiling on
simultaneous containers, which is also your cost ceiling.
"""

import modal

app = modal.App("extinction-chess-web")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # C++ toolchain for _ext_chess (see build step at end of image).
    .apt_install("g++", "build-essential")
    .pip_install("torch", "numpy", "fastapi[standard]", "pybind11")
    # Mirror the repo layout: engine.py resolves src/ as ../../src relative
    # to itself, so tools/ and src/ must sit side by side.
    .add_local_dir("src", remote_path="/root/src", copy=True,
                   ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p)
                   or str(p).endswith(".pyd") or str(p).endswith(".so"))
    .add_local_dir("tools", remote_path="/root/tools", copy=True,
                   ignore=lambda p: ".venv" in str(p) or "__pycache__" in str(p))
    # Build _ext_chess so state_encoder.encode_board uses the C++ path.
    # Without this, the Python fallback zeroes 100 of 115 planes and the
    # net plays essentially a different model (measured: 25% top-1 policy
    # agreement with the real trained model). See briefing §14.C5.
    # copy=True on add_local_dir above is required for run_commands to
    # see the sources at image-build time.
    .run_commands("cd /root/src && python setup.py build_ext --inplace")
)

models_volume = modal.Volume.from_name("extinction-chess-models", create_if_missing=True)
MODELS_DIR = "/models"

# ── Tunables ────────────────────────────────────────────────────────────────
CHECKPOINT = "az_iter1040.pt"   # briefing §6: latest+strongest. az_iter810.pt
                                # is the validated pre-regression peak if you
                                # want stability over peak strength.
GPU_TIER = "L4"                 # T4 is cheaper but ~2x slower; A10 has headroom
SIM_CEILING = 800               # briefing §13: training default
TACTICAL_LEVEL = "basic"        # matches training-time behaviour
C_PUCT = 2.5

MAX_CONCURRENT_GAMES = 6        # hard cost ceiling
GAME_TIMEOUT_S = 3600           # a container can never outlive this

# Origins allowed to call this endpoint. Add your domain before going public;
# "*" is fine while developing.
ALLOWED_ORIGINS = ["*"]


@app.function(
    image=image,
    gpu=GPU_TIER,
    volumes={MODELS_DIR: models_volume},
    timeout=GAME_TIMEOUT_S,
    # NOTE: these are the current parameter names. On older modal clients they
    # were container_idle_timeout and concurrency_limit respectively.
    scaledown_window=120,           # stay warm briefly between games
    max_containers=MAX_CONCURRENT_GAMES,
)
# Deliberately NO @modal.concurrent — one connection per container is what
# gives each game its own GPU and its own MCTS tree.
@modal.asgi_app()
def web():
    import asyncio
    import os
    import sys
    import time

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/src")
    sys.path.insert(0, "/root/tools")

    import torch
    from extinction_chess import Color, ExtinctionChess, PieceType, Position
    from play_timed.engine import Engine

    api = FastAPI()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(MODELS_DIR, CHECKPOINT)
    if not os.path.exists(ckpt):
        raise RuntimeError(
            f"checkpoint missing at {ckpt} — upload it with:\n"
            f"  modal volume put extinction-chess-models "
            f"models/{CHECKPOINT} /{CHECKPOINT}"
        )

    # Loaded once per container, before any request is served.
    engine = Engine(
        model_path=ckpt,
        device=device,
        tactical_level=TACTICAL_LEVEL,
        c_puct=C_PUCT,
        dirichlet_alpha=0.0,   # deterministic play
        noise_weight=0.0,
    )
    print(f"engine ready: {CHECKPOINT} on {device}")

    # ── Serialization ───────────────────────────────────────────────────────
    # Position(rank=0, file=0) is a1. Board goes out as board[rank][file] with
    # rank 0 = white's back rank; the client flips for the black-side view.

    def sq(pos) -> str:
        return f"{'abcdefgh'[pos.file]}{pos.rank + 1}"

    def parse_sq(s: str):
        return Position(rank=int(s[1]) - 1, file="abcdefgh".index(s[0]))

    def piece_str(p) -> str:
        return ("w" if p.color == Color.WHITE else "b") + p.piece_type.value

    def board_json(game):
        rows = []
        for r in range(8):
            row = []
            for f in range(8):
                p = game.board.get_piece(Position(rank=r, file=f))
                row.append(piece_str(p) if p else None)
            rows.append(row)
        return rows

    def legal_json(game):
        out = []
        for m in game.get_legal_moves():
            out.append({
                "from": sq(m.from_pos),
                "to": sq(m.to_pos),
                "promotion": m.promotion.value if m.promotion else None,
            })
        return out

    def counts_json(game):
        """Piece counts per side — this is the win condition, so surface it."""
        res = {}
        for color, key in ((Color.WHITE, "white"), (Color.BLACK, "black")):
            counts = game.board.get_piece_count(color)
            res[key] = {pt.value: counts.get(pt, 0) for pt in PieceType}
        return res

    def over_json(game):
        if not getattr(game, "game_over", False):
            return None
        winner = getattr(game, "winner", None)
        return {
            "winner": None if winner is None
                      else ("white" if winner == Color.WHITE else "black"),
            "reason": getattr(game, "draw_reason", None) or "extinction",
        }

    def state_msg(game, clocks, human_color):
        return {
            "type": "state",
            "board": board_json(game),
            "to_move": "white" if game.current_player == Color.WHITE else "black",
            "human_color": human_color,
            "legal": legal_json(game),
            "counts": counts_json(game),
            "clocks": {"white": round(clocks["white"], 1),
                       "black": round(clocks["black"], 1)},
            "game_over": over_json(game),
        }

    # ── Time budget (ported from tools/play_timed/state.py:359) ─────────────
    def thinking_budget(remaining, base_seconds, increment, delay=0.0):
        base = remaining / 30 + increment          # assume ~30 moves left
        hard_cap = max(30, base_seconds / 4)
        budget = min(base, hard_cap)
        if remaining < 10:                          # time-trouble safety
            budget = min(budget, remaining * 0.3)
        return budget + delay

    @api.get("/health")
    async def health():
        """Cheap GET that wakes a container so the first move isn't cold."""
        return {"ok": True, "checkpoint": CHECKPOINT, "device": device}

    @api.websocket("/play")
    async def play(ws: WebSocket):
        await ws.accept()
        game = None
        clocks = {"white": 0.0, "black": 0.0}
        human_color = "white"
        base_s = 300.0
        inc_s = 3.0
        sim_ceiling = SIM_CEILING

        def move_key(m):
            return (sq(m.from_pos), sq(m.to_pos),
                    m.promotion.value if m.promotion else None)

        async def engine_turn():
            """Wait for sim ceiling or deadline, then play the engine's move."""
            nonlocal game

            eng_key = "black" if human_color == "white" else "white"
            budget = thinking_budget(clocks[eng_key], base_s, inc_s)
            started = time.monotonic()
            deadline = started + budget
            hard_deadline = started + budget + 30   # safety net for a stuck worker

            # Engine.descend() is ASYNCHRONOUS — it queues a request that the
            # worker thread consumes between MCTS chunks. So immediately after
            # a descend, get_current_result() may still describe the PREVIOUS
            # root, whose best move belongs to the other player. Pondering makes
            # this far more likely, not less: a deep ponder tree satisfies the
            # sim ceiling instantly, so without this guard we'd read the stale
            # root every time the engine was thinking well.
            #
            # A stale move is always by the opposite colour and therefore never
            # legal in the current position, so legality is a reliable freshness
            # test. Computed here, after the human's move has been applied.
            legal_now = {move_key(m) for m in game.get_legal_moves()}

            result = None
            last_sent = -1
            while True:
                snap = engine.get_status_snapshot()
                cand = engine.get_current_result()

                if cand is not None and move_key(cand["move"]) in legal_now:
                    result = cand
                    if (snap["sim_count"] >= sim_ceiling
                            or time.monotonic() >= deadline):
                        break

                if time.monotonic() >= hard_deadline:
                    break

                # Only emit on change — the worker publishes per 30-sim chunk,
                # so polling faster than that just spams the socket.
                if snap["sim_count"] != last_sent:
                    await ws.send_json({"type": "thinking",
                                        "sims": snap["sim_count"],
                                        "ceiling": sim_ceiling})
                    last_sent = snap["sim_count"]
                await asyncio.sleep(0.05)

            if result is None:
                await ws.send_json({"type": "error",
                                    "message": "engine produced no legal move"})
                return

            spent = time.monotonic() - started
            clocks[eng_key] = max(0.0, clocks[eng_key] - spent) + inc_s

            mv = result["move"]
            snapshot = result["search_snapshot"]

            if not game.make_move(mv):
                # Should be unreachable given the legality guard above, but a
                # silent False here is what desynced the client before.
                await ws.send_json({"type": "error",
                                    "message": "engine move rejected by rules"})
                return
            engine.descend(mv)   # keep pondering into the human's turn

            await ws.send_json({
                "type": "engine_move",
                "from": sq(mv.from_pos),
                "to": sq(mv.to_pos),
                "promotion": mv.promotion.value if mv.promotion else None,
                "analysis": {
                    "sims": snapshot["sim_count"],
                    # root_value is from the mover's perspective; flip so the
                    # client can always display it as "white is winning".
                    "value": (snapshot["root_value"]
                              if game.current_player == Color.BLACK
                              else -snapshot["root_value"]),
                    "seconds": round(spent, 2),
                },
            })
            await ws.send_json(state_msg(game, clocks, human_color))

        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("type")

                if kind == "new_game":
                    human_color = msg.get("human_color", "white")
                    base_s = float(msg.get("base", 300))
                    inc_s = float(msg.get("increment", 3))
                    sim_ceiling = int(msg.get("sim_ceiling", SIM_CEILING))

                    game = ExtinctionChess()
                    clocks = {"white": base_s, "black": base_s}

                    engine.stop()
                    engine.start_from(game)

                    await ws.send_json(state_msg(game, clocks, human_color))
                    if human_color == "black":
                        await engine_turn()      # engine opens as white
                    turn_started = time.monotonic()

                elif kind == "move":
                    if game is None or getattr(game, "game_over", False):
                        await ws.send_json({"type": "error",
                                            "message": "no active game"})
                        continue

                    # Match against the rules engine's own move list rather
                    # than constructing a Move — that's what carries the
                    # castling / en-passant flags correctly.
                    want = (msg.get("from"), msg.get("to"), msg.get("promotion"))
                    chosen = None
                    for m in game.get_legal_moves():
                        got = (sq(m.from_pos), sq(m.to_pos),
                               m.promotion.value if m.promotion else None)
                        if got == want:
                            chosen = m
                            break
                    if chosen is None:
                        await ws.send_json({"type": "error",
                                            "message": "illegal move"})
                        continue

                    hk = human_color
                    spent = time.monotonic() - turn_started
                    clocks[hk] = clocks[hk] - spent
                    if clocks[hk] <= 0:
                        clocks[hk] = 0.0
                        game.game_over = True
                        engine.stop()
                        msg_out = state_msg(game, clocks, human_color)
                        msg_out["game_over"] = {
                            "winner": "black" if hk == "white" else "white",
                            "reason": "timeout",
                        }
                        await ws.send_json(msg_out)
                        continue
                    clocks[hk] += inc_s

                    game.make_move(chosen)
                    engine.descend(chosen)
                    await ws.send_json(state_msg(game, clocks, human_color))

                    if not getattr(game, "game_over", False):
                        await engine_turn()
                    turn_started = time.monotonic()

                elif kind == "resign":
                    # state_msg dereferences game.board, so a resign arriving
                    # before new_game would crash the handler.
                    if game is None:
                        await ws.send_json({"type": "error",
                                            "message": "no active game"})
                        continue
                    engine.stop()
                    game.game_over = True
                    msg_out = state_msg(game, clocks, human_color)
                    msg_out["game_over"] = {
                        "winner": "black" if human_color == "white" else "white",
                        "reason": "resignation",
                    }
                    await ws.send_json(msg_out)

                elif kind == "ping":
                    await ws.send_json({"type": "pong"})

        except WebSocketDisconnect:
            pass
        finally:
            # Release the ponder thread so the container can scale down.
            try:
                engine.stop()
            except Exception:
                pass

    return api
