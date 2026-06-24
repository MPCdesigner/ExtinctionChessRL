#!/bin/bash
#SBATCH --job-name=ext-helper
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h74liang/helper_%j.log

# Helper SLURM script — spawned by main training at the start of each iter.
#
# Each invocation generates ~200 self-play games against the current
# az_latest.pt model and writes them to OUTPUT_PATH. Main consumes these
# files at its training time and deletes them after.
#
# Usage (from main's launcher logic):
#   sbatch --gres=gpu:rtx_2080_ti:1 --nodelist=delta-slurm1 \
#          ~/extinction-chess/helper.sh /path/to/helper_v681_id0.npz
#
# The --gres and --nodelist are passed at submission time so main can
# fall back from 2080 Ti to 3090 if the preferred node is busy.
#
# This script does NOT auto-resubmit — helpers are one-shot. Main
# launches fresh helpers each iter.

export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
cd ~/extinction-chess/src

OUTPUT_PATH="${1:?Usage: sbatch helper.sh <output_path>}"
MODEL_PATH="${MODEL_PATH:-$HOME/extinction-chess/models/az_latest.pt}"

python3 setup.py build_ext --inplace

python3 run_helper.py \
    --model-path "$MODEL_PATH" \
    --output-path "$OUTPUT_PATH" \
    --num-threads 4
