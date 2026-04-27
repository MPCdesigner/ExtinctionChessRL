#!/bin/bash
#SBATCH --job-name=ext-compare
#SBATCH --partition=compute
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=compare_%j.log

export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd ~/extinction-chess/src
python3 setup.py build_ext --inplace
python3 compare_extensive.py --m1 az_iter_190_100pct.pt --m2 az_iter_100_100pct.pt --device cuda
