"""
[ARCHIVED] Standalone training script — runs on any machine with Python + PyTorch.
No Modal dependency. Not currently in use — training moved to WATonomous SLURM cluster.
Kept as a backup in case cluster access is lost.

Usage:
    python train.py                                    # default: CNN, 50 iters
    python train.py --iterations 100 --games 200
    python train.py --mcts --simulations 100           # MCTS self-play
    python train.py --no-resume                        # fresh start
    python train.py --mlp                              # old MLP architecture
"""

import os
import sys
import argparse
import time
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from torch_model import (
    ChessNet, TorchEvaluator,
    ChessCNN, CNNEvaluator,
    play_one_game, play_vs_greedy, play_vs_checkpoint, test_vs_random,
    play_one_game_mcts, play_vs_greedy_mcts,
)


def train(
    iterations=50,
    games_per_iteration=200,
    learning_rate=0.001,
    batch_size=64,
    resume=True,
    use_cnn=True,
    num_filters=128,
    num_res_blocks=6,
    hidden_sizes=None,
    use_mcts=False,
    mcts_simulations=100,
    models_dir="models",
):
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: {'CNN' if use_cnn else 'MLP'}")

    os.makedirs(models_dir, exist_ok=True)

    # ── checkpoint paths
    prefix = "cnn" if use_cnn else "mlp"
    checkpoint_path = os.path.join(models_dir, f"{prefix}_latest.pt")

    # ── load or create model
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

    # ── checkpoint pool
    checkpoint_pool = []
    pool_dir = os.path.join(models_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    if not resume:
        for f in os.listdir(pool_dir):
            os.remove(os.path.join(pool_dir, f))
        print("Cleared old checkpoint pool")
    else:
        for f in sorted(os.listdir(pool_dir)):
            if f.endswith(".pt"):
                checkpoint_pool.append(os.path.join(pool_dir, f))
    print(f"Checkpoint pool: {len(checkpoint_pool)} past opponents")

    noise_weight = 0.25
    dirichlet_alpha = 0.3

    if use_mcts:
        print(f"MCTS enabled: {mcts_simulations} simulations/move")

    # ── main loop
    for iteration in range(start_iter, start_iter + iterations):
        t0 = time.time()
        iter_num = iteration + 1

        model.eval()
        all_positions, all_outcomes = [], []
        wins_w, wins_b, draws = 0, 0, 0

        chunk = games_per_iteration // 3

        # Load a random opponent from the pool
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

        # ── training
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
        print(f"         loss={avg_loss:.4f} | time={elapsed:.1f}s")

        training_log.append({
            "iteration": iter_num,
            "loss": avg_loss,
            "wins_white": wins_w,
            "wins_black": wins_b,
            "draws": draws,
            "time": elapsed,
        })

        # ── save checkpoint every iteration
        model.save_checkpoint(checkpoint_path, iteration=iter_num)

        # ── evaluate every 10 iterations
        if iter_num % 10 == 0:
            model.eval()
            if use_cnn:
                evaluator_tmp = CNNEvaluator(model, device=str(device))
            else:
                evaluator_tmp = TorchEvaluator(model, device=str(device))
            win_rate = test_vs_random(evaluator_tmp, num_games=100)
            print(f"         win rate vs random: {win_rate:.1%}")

            versioned = os.path.join(
                models_dir, f"{prefix}_iter_{iter_num}_{int(win_rate*100)}pct.pt"
            )
            model.save_checkpoint(versioned, iteration=iter_num, win_rate=win_rate)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_path = os.path.join(models_dir, f"{prefix}_best.pt")
                model.save_checkpoint(best_path, iteration=iter_num, win_rate=win_rate)
                print(f"         ★ new best model: {win_rate:.1%}")

                pool_path = os.path.join(pool_dir, f"pool_iter_{iter_num}.pt")
                model.save_checkpoint(pool_path, iteration=iter_num, win_rate=win_rate)
                checkpoint_pool.append(pool_path)
                if len(checkpoint_pool) > 10:
                    old = checkpoint_pool.pop(0)
                    if os.path.exists(old):
                        os.remove(old)
                print(f"         added to pool (size={len(checkpoint_pool)})")

    # ── final save
    model.eval()
    if use_cnn:
        evaluator_final = CNNEvaluator(model, device=str(device))
    else:
        evaluator_final = TorchEvaluator(model, device=str(device))
    final_wr = test_vs_random(evaluator_final, num_games=200)
    print(f"\nFinal win rate vs random: {final_wr:.1%}")

    model.save_checkpoint(checkpoint_path, iteration=start_iter + iterations, win_rate=final_wr)
    final_path = os.path.join(models_dir, f"{prefix}_final.pt")
    model.save_checkpoint(final_path, iteration=start_iter + iterations, win_rate=final_wr)

    log_path = os.path.join(models_dir, f"{prefix}_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"Done — {start_iter + iterations} total iterations, best={best_win_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Extinction Chess RL")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--mcts", action="store_true")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    train(
        iterations=args.iterations,
        games_per_iteration=args.games,
        learning_rate=args.lr,
        resume=not args.no_resume,
        use_cnn=not args.mlp,
        use_mcts=args.mcts,
        mcts_simulations=args.simulations,
        models_dir=args.models_dir,
    )
