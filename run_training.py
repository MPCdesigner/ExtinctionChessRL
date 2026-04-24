import multiprocessing
from alphazero import train

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train(
        iterations=100,
        games_per_iteration=400,
        num_simulations=400,
        models_dir="../models",
        resume=True,
        num_workers=4,
    )
