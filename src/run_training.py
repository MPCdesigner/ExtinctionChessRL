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
    )
