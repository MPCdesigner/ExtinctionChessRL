"""
[ARCHIVED] Modal training script for Extinction Chess RL.
Not currently in use — training moved to WATonomous SLURM cluster.
Kept as a backup in case cluster access is lost.

Usage:
    modal run modal_train.py                          # default: CNN, 50 iters, 200 games each
    modal run modal_train.py --iterations 100 --games 500
    modal run modal_train.py --mlp                    # use old MLP instead of CNN
    modal run modal_train.py::test                    # test latest checkpoint vs random
    modal run modal_train.py::convert_numpy_model     # import existing numpy model
"""

import modal
import os
import sys

# ── Modal resources ──────────────────────────────────────────────────────────

app = modal.App("extinction-chess-rl")

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(
        "src",
        remote_path="/root/src",
        ignore=lambda path: ".venv" in str(path) or "__pycache__" in str(path),
    )
)

model_volume = modal.Volume.from_name("extinction-chess-models", create_if_missing=True)
MODELS_DIR = "/models"

# ── Training ─────────────────────────────────────────────────────────────────

@app.function(
    image=training_image,
    gpu="L4",
    timeout=28800,  # 8 hours max
    volumes={MODELS_DIR: model_volume},
)
def train(
    iterations: int = 50,
    games_per_iteration: int = 200,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    resume: bool = True,
    use_cnn: bool = True,
    # CNN params
    num_filters: int = 128,
    num_res_blocks: int = 6,
    # MLP params
    hidden_sizes: list = [256, 128, 64],
    # MCTS params
    use_mcts: bool = False,
    mcts_simulations: int = 100,
):
    sys.path.insert(0, "/root/src")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import json
    import time
    import random

    from torch_model import (
        ChessNet, TorchEvaluator,
        ChessCNN, CNNEvaluator,
        play_one_game, play_vs_greedy, play_vs_checkpoint, test_vs_random,
        play_one_game_mcts, play_vs_greedy_mcts,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {'CNN' if use_cnn else 'MLP'}")

    # ── checkpoint paths ──────────────────────────────────────────────────
    prefix = "cnn" if use_cnn else "mlp"
    checkpoint_path = os.path.join(MODELS_DIR, f"{prefix}_latest.pt")

    # ── load or create model ──────────────────────────────────────────────
    if use_cnn:
        if resume and os.path.exists(checkpoint_path):
            model, meta = ChessCNN.load_checkpoint(checkpoint_path)
            start_iter = meta.get("iteration", 0)
            print(f"Resumed CNN from iteration {start_iter}")
        else:
            model = ChessCNN(in_channels=14, num_filters=num_filters, num_res_blocks=num_res_blocks)
            start_iter = 0
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Fresh CNN: 14×8×8 → {num_res_blocks} res blocks ({num_filters} filters) → 1")
            print(f"Total parameters: {total_params:,}")
    else:
        if resume and os.path.exists(checkpoint_path):
            model, meta = ChessNet.load_checkpoint(
                checkpoint_path, input_size=39, hidden_sizes=hidden_sizes
            )
            start_iter = meta.get("iteration", 0)
            print(f"Resumed MLP from iteration {start_iter}")
        else:
            model = ChessNet(input_size=39, hidden_sizes=hidden_sizes)
            start_iter = 0
            print(f"Fresh MLP: 39 → {hidden_sizes} → 1")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if use_cnn:
        evaluator = CNNEvaluator(model, device=str(device))
    else:
        evaluator = TorchEvaluator(model, device=str(device))

    training_log = []
    best_win_rate = 0.0

    # ── checkpoint pool for past opponents ────────────────────────────────
    checkpoint_pool = []  # list of file paths
    pool_dir = os.path.join(MODELS_DIR, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    if not resume:
        # Fresh start — clear stale pool from previous (possibly bugged) runs
        for f in os.listdir(pool_dir):
            os.remove(os.path.join(pool_dir, f))
        print("Cleared old checkpoint pool")
    else:
        # Load existing pool checkpoints
        for f in sorted(os.listdir(pool_dir)):
            if f.endswith(".pt"):
                checkpoint_pool.append(os.path.join(pool_dir, f))
    print(f"Checkpoint pool: {len(checkpoint_pool)} past opponents")

    noise_weight = 0.25
    dirichlet_alpha = 0.3

    if use_mcts:
        print(f"MCTS enabled: {mcts_simulations} simulations/move")

    # ── main loop ─────────────────────────────────────────────────────────
    for iteration in range(start_iter, start_iter + iterations):
        t0 = time.time()
        iter_num = iteration + 1

        # ── generate games: 1/3 self-play, 1/3 vs checkpoint, 1/3 vs greedy
        model.eval()
        all_positions, all_outcomes = [], []
        wins_w, wins_b, draws = 0, 0, 0

        chunk = games_per_iteration // 3
        # Boundaries: [0, chunk) = self-play, [chunk, 2*chunk) = vs checkpoint, rest = vs greedy

        # Load a random opponent from the pool (if any)
        opponent_eval = None
        if checkpoint_pool:
            opp_path = random.choice(checkpoint_pool)
            if use_cnn:
                opp_model, _ = ChessCNN.load_checkpoint(opp_path)
                opp_model = opp_model.to(device)
                opponent_eval = CNNEvaluator(opp_model, device=str(device))
            else:
                opp_model, _ = ChessNet.load_checkpoint(opp_path, input_size=39, hidden_sizes=hidden_sizes)
                opp_model = opp_model.to(device)
                opponent_eval = TorchEvaluator(opp_model, device=str(device))

        for g in range(games_per_iteration):
            if g < chunk:
                # Self-play
                if use_mcts:
                    positions, outcome = play_one_game_mcts(
                        evaluator,
                        num_simulations=mcts_simulations,
                        dirichlet_alpha=dirichlet_alpha,
                        noise_weight=noise_weight,
                        use_cnn=use_cnn,
                    )
                else:
                    positions, outcome = play_one_game(
                        evaluator,
                        max_moves=200,
                        dirichlet_alpha=dirichlet_alpha,
                        noise_weight=noise_weight,
                        use_cnn=use_cnn,
                    )
            elif g < 2 * chunk and opponent_eval is not None:
                # Play vs past checkpoint
                positions, outcome = play_vs_checkpoint(
                    evaluator,
                    opponent_eval,
                    max_moves=200,
                    dirichlet_alpha=dirichlet_alpha,
                    noise_weight=noise_weight,
                    use_cnn=use_cnn,
                    model_is_white=(g % 2 == 0),
                )
            else:
                # Play vs greedy-random
                if use_mcts:
                    positions, outcome = play_vs_greedy_mcts(
                        evaluator,
                        num_simulations=mcts_simulations,
                        dirichlet_alpha=dirichlet_alpha,
                        noise_weight=noise_weight,
                        use_cnn=use_cnn,
                        model_is_white=(g % 2 == 0),
                    )
                else:
                    positions, outcome = play_vs_greedy(
                        evaluator,
                        max_moves=200,
                        dirichlet_alpha=dirichlet_alpha,
                        noise_weight=noise_weight,
                        use_cnn=use_cnn,
                        model_is_white=(g % 2 == 0),
                    )

            for i, pos in enumerate(positions):
                perspective = 1.0 if i % 2 == 0 else -1.0
                all_positions.append(pos)
                all_outcomes.append(outcome * perspective)

            if outcome > 0:
                wins_w += 1
            elif outcome < 0:
                wins_b += 1
            else:
                draws += 1

        print(
            f"[iter {iter_num}] games W={wins_w} B={wins_b} D={draws} "
            f"| positions={len(all_positions)} | pool={len(checkpoint_pool)}"
        )

        # ── training ──────────────────────────────────────────────────────
        model.train()
        X = torch.tensor(np.array(all_positions), dtype=torch.float32, device=device)
        y = torch.tensor(np.array(all_outcomes), dtype=torch.float32, device=device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss, n_batches = 0.0, 0
        for _epoch in range(5):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = nn.MSELoss()(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"         loss={avg_loss:.4f} | time={elapsed:.1f}s | noise={noise_weight:.2f}")

        training_log.append(
            {
                "iteration": iter_num,
                "loss": avg_loss,
                "wins_white": wins_w,
                "wins_black": wins_b,
                "draws": draws,
                "time": elapsed,
                "noise_weight": noise_weight,
            }
        )

        # ── save checkpoint every iteration ──────────────────────────────
        model.save_checkpoint(checkpoint_path, iteration=iter_num)
        model_volume.commit()

        # ── evaluate every 10 iterations ──────────────────────────────────
        if iter_num % 10 == 0:
            model.eval()
            if use_cnn:
                evaluator_tmp = CNNEvaluator(model, device=str(device))
            else:
                evaluator_tmp = TorchEvaluator(model, device=str(device))
            win_rate = test_vs_random(evaluator_tmp, num_games=100)
            print(f"         win rate vs random: {win_rate:.1%}")

            versioned = os.path.join(
                MODELS_DIR, f"{prefix}_iter_{iter_num}_{int(win_rate*100)}pct.pt"
            )
            model.save_checkpoint(versioned, iteration=iter_num, win_rate=win_rate)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = os.path.join(MODELS_DIR, f"{prefix}_best.pt")
                model.save_checkpoint(best_path, iteration=iter_num, win_rate=win_rate)
                print(f"         ★ new best model: {win_rate:.1%}")

                # Add to checkpoint pool (keep max 10)
                pool_path = os.path.join(pool_dir, f"pool_iter_{iter_num}.pt")
                model.save_checkpoint(pool_path, iteration=iter_num, win_rate=win_rate)
                checkpoint_pool.append(pool_path)
                if len(checkpoint_pool) > 10:
                    old = checkpoint_pool.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                print(f"         added to pool (size={len(checkpoint_pool)})")

            model_volume.commit()

    # ── final save ────────────────────────────────────────────────────────
    model.eval()
    if use_cnn:
        evaluator_final = CNNEvaluator(model, device=str(device))
    else:
        evaluator_final = TorchEvaluator(model, device=str(device))
    final_wr = test_vs_random(evaluator_final, num_games=200)
    print(f"\nFinal win rate vs random: {final_wr:.1%}")

    model.save_checkpoint(checkpoint_path, iteration=start_iter + iterations, win_rate=final_wr)
    final_path = os.path.join(MODELS_DIR, f"{prefix}_final.pt")
    model.save_checkpoint(final_path, iteration=start_iter + iterations, win_rate=final_wr)

    log_path = os.path.join(MODELS_DIR, f"{prefix}_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    model_volume.commit()
    print(f"Done — {start_iter + iterations} total iterations, best={best_win_rate:.1%}")
    return training_log


# ── Test latest checkpoint ────────────────────────────────────────────────────

@app.function(
    image=training_image,
    gpu="T4",
    timeout=600,
    volumes={MODELS_DIR: model_volume},
)
def test(num_games: int = 200, use_cnn: bool = True):
    sys.path.insert(0, "/root/src")

    from torch_model import ChessNet, TorchEvaluator, ChessCNN, CNNEvaluator, test_vs_random
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix = "cnn" if use_cnn else "mlp"
    checkpoint_path = os.path.join(MODELS_DIR, f"{prefix}_latest.pt")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Run training first.")
        return

    if use_cnn:
        model, meta = ChessCNN.load_checkpoint(checkpoint_path)
        model = model.to(device)
        evaluator = CNNEvaluator(model, device=str(device))
    else:
        model, meta = ChessNet.load_checkpoint(checkpoint_path, input_size=39, hidden_sizes=[256, 128, 64])
        model = model.to(device)
        evaluator = TorchEvaluator(model, device=str(device))

    print(f"Loaded {prefix.upper()} checkpoint — iteration {meta.get('iteration', '?')}")
    win_rate = test_vs_random(evaluator, num_games=num_games)
    print(f"Win rate vs random ({num_games} games): {win_rate:.1%}")


# ── Convert existing numpy model to PyTorch checkpoint ────────────────────────

@app.function(
    image=training_image,
    timeout=120,
    volumes={MODELS_DIR: model_volume},
)
def convert_numpy_model(numpy_model_path: str = "models/versions/model_v4_extinction_focused_96pct.json"):
    sys.path.insert(0, "/root/src")

    from torch_model import ChessNet

    remote_path = os.path.join("/root/src", numpy_model_path)
    if not os.path.exists(remote_path):
        print(f"Not found: {remote_path}")
        print("Available files in /root/src/models/:")
        for root, dirs, files in os.walk("/root/src/models"):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        return

    model = ChessNet.from_numpy_model(remote_path)
    out_path = os.path.join(MODELS_DIR, "mlp_latest.pt")
    model.save_checkpoint(out_path, source=numpy_model_path, converted=True)
    model_volume.commit()
    print(f"Converted {numpy_model_path} → {out_path}")


# ── CNN vs MLP head-to-head ────────────────────────────────────────────────────

@app.function(
    image=training_image,
    gpu="T4",
    timeout=3600,
    volumes={MODELS_DIR: model_volume},
)
def cnn_vs_mlp(num_games: int = 100):
    sys.path.insert(0, "/root/src")

    import torch
    import random
    from extinction_chess import ExtinctionChess, Color
    from torch_model import (
        ChessNet, TorchEvaluator,
        ChessCNN, CNNEvaluator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CNN — prefer early checkpoint (before collapse)
    cnn_path = os.path.join(MODELS_DIR, "cnn_iter_10_95pct.pt")
    if not os.path.exists(cnn_path):
        cnn_path = os.path.join(MODELS_DIR, "cnn_best.pt")
    if not os.path.exists(cnn_path):
        cnn_path = os.path.join(MODELS_DIR, "cnn_latest.pt")
    cnn_model, cnn_meta = ChessCNN.load_checkpoint(cnn_path)
    cnn_model = cnn_model.to(device)
    cnn_eval = CNNEvaluator(cnn_model, device=str(device))
    print(f"CNN: iteration {cnn_meta.get('iteration', '?')}")

    # Load MLP
    mlp_path = os.path.join(MODELS_DIR, "latest.pt")
    if not os.path.exists(mlp_path):
        mlp_path = os.path.join(MODELS_DIR, "mlp_latest.pt")
    mlp_model, mlp_meta = ChessNet.load_checkpoint(mlp_path, input_size=39, hidden_sizes=[256, 128, 64])
    mlp_model = mlp_model.to(device)
    mlp_eval = TorchEvaluator(mlp_model, device=str(device))
    print(f"MLP: iteration {mlp_meta.get('iteration', '?')}")

    def select_move(game, evaluator):
        legal = game.get_legal_moves()
        if not legal:
            return None
        best_score, best_move = -float("inf"), legal[0]
        for m in legal:
            gc = ExtinctionChess()
            gc.board = game.board.copy()
            gc.current_player = game.current_player
            gc.game_over = game.game_over
            if gc.make_move(m):
                if gc.game_over:
                    # Terminal: engine doesn't switch current_player
                    if gc.winner == game.current_player:
                        s = 1.0
                    elif gc.winner is not None:
                        s = -1.0
                    else:
                        s = 0.0
                else:
                    # Non-terminal: current_player switched, negate
                    s = -evaluator.evaluate(gc)
                if s > best_score:
                    best_score, best_move = s, m
        return best_move

    cnn_wins, mlp_wins, draws = 0, 0, 0

    print(f"\nPlaying {num_games} games (CNN vs MLP)...")
    for game_num in range(num_games):
        game = ExtinctionChess()
        cnn_is_white = game_num % 2 == 0
        moves = 0

        while not game.game_over and moves < 200:
            is_cnn_turn = (game.current_player == Color.WHITE) == cnn_is_white
            if is_cnn_turn:
                move = select_move(game, cnn_eval)
            else:
                move = select_move(game, mlp_eval)

            if move:
                game.make_move(move)
                moves += 1
            else:
                break

        cnn_color = "W" if cnn_is_white else "B"
        if game.winner:
            cnn_won = (game.winner == Color.WHITE) == cnn_is_white
            winner_str = "CNN" if cnn_won else "MLP"
            if cnn_won:
                cnn_wins += 1
            else:
                mlp_wins += 1
        else:
            winner_str = "Draw"
            if game.draw_reason:
                winner_str += f" ({game.draw_reason})"
            draws += 1

        # Show extinct piece type if there was a winner
        extinct_info = ""
        if game.winner:
            loser_color = Color.BLACK if game.winner == Color.WHITE else Color.WHITE
            extinct = game.get_extinct_pieces(loser_color)
            if extinct:
                extinct_info = f" [{','.join(p.value for p in extinct)} extinct]"

        print(f"  Game {game_num+1}: CNN={cnn_color} | {moves} moves | {winner_str}{extinct_info}")

        if (game_num + 1) % 10 == 0:
            print(f"  --- Score: CNN={cnn_wins} MLP={mlp_wins} Draws={draws} ---")

    print(f"\nFinal results ({num_games} games):")
    print(f"  CNN wins: {cnn_wins} ({100*cnn_wins/num_games:.1f}%)")
    print(f"  MLP wins: {mlp_wins} ({100*mlp_wins/num_games:.1f}%)")
    print(f"  Draws:    {draws} ({100*draws/num_games:.1f}%)")


# ── CLI entry point ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    iterations: int = 50,
    games: int = 200,
    lr: float = 0.001,
    no_resume: bool = False,
    mlp: bool = False,
    mcts: bool = False,
    simulations: int = 100,
):
    result = train.remote(
        iterations=iterations,
        games_per_iteration=games,
        learning_rate=lr,
        resume=not no_resume,
        use_cnn=not mlp,
        use_mcts=mcts,
        mcts_simulations=simulations,
    )
    print(f"\nTraining finished — {len(result)} iterations logged.")
