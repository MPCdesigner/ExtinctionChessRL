#!/bin/bash
#SBATCH --job-name=win-taking
#SBATCH --output=/home/h74liang/bench_win_taking_%j.log
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd ~/extinction-chess/src

python3 bench_win_taking.py \
    --models az_iter_360_98pct.pt az_iter_350_100pct.pt az_iter_340_100pct.pt az_iter_100_100pct.pt \
    --sims 20 50 100 200 \
    --positions 200 \
    --directions backward sideways \
    --min-distance 3
