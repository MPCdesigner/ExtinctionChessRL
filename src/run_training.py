import multiprocessing
import os
from alphazero import train

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train(
        iterations=100,
        games_per_iteration=400,
        num_simulations=800,
        learning_rate=0.00005,  # bumped Jul 7 from 0.00002 to break iter 700-740 H2H plateau
        models_dir="../models",
        resume=True,
        num_workers=4,
        # instant_win_positions=1000,  # Disabled for now; revisit at iter 270
        hard_win_positions=300,
        extra_hard_win_positions=0,  # disabled to stop further damage
        max_wall_time=23 * 3600,  # Exit cleanly before 24h SLURM limit
        num_epochs=3,  # reduced from 10; with K=5 buffer, each position seen 15x over its lifetime
        drilling_epochs=5,
        drilling_lr_factor=0.2,  # adjusted from 0.5 so drilling LR stays at 0.00001 despite main LR bump
        extra_hard_epochs=5,
        extra_hard_lr_factor=0.025,
        replay_buffer_dir="../replay_buffer",
        replay_buffer_size=5,  # rolling window of last 5 iterations
        # ── Decoupled helper jobs (recency injection) ──
        # Each iter, main sbatches helpers_per_iter helpers that each generate
        # 200 games against az_latest.pt. Main consumes the resulting .npz files
        # at training time and concatenates them with its own self-play data.
        # Helper data is NOT stored in the K-buffer.
        helpers_enabled=True,
        helpers_per_iter=2,
        helper_script_path=os.path.expanduser("~/extinction-chess/helper.sh"),
        helper_max_wait_seconds=25,
        helper_primary_gres="gpu:rtx_2080_ti:1",
        helper_primary_node="delta-slurm1",
        helper_fallback_gres="gpu:rtx_3090:1",
        helper_fallback_node="trpro-slurm1",
        # ── Benchmark battery ──
        # Fires 13 tests + aggregator on trpro-slurm2 after each ×10 iter save.
        # Non-blocking: main continues immediately, summary appears ~5-6h later
        # in ~/extinction-chess/benchmark_results/iter_<N>.txt.
        benchmark_enabled=True,
    )
