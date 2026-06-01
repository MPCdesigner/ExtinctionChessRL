import multiprocessing
from alphazero import train

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train(
        iterations=100,
        games_per_iteration=400,
        num_simulations=800,
        learning_rate=0.0001,
        models_dir="../models",
        resume=True,
        num_workers=4,
        # instant_win_positions=1000,  # Disabled for now; revisit at iter 270
        hard_win_positions=300,
        extra_hard_win_positions=300,
        max_wall_time=23 * 3600,  # Exit cleanly before 24h SLURM limit
        num_epochs=10,
        drilling_epochs=5,
        drilling_lr_factor=0.5,
        extra_hard_epochs=5,
        extra_hard_lr_factor=0.025,
    )
