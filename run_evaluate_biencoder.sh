#!/bin/bash
#SBATCH --job-name=eval_biencoder
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"

CHECKPOINT=${1:?"Uso: sbatch run_evaluate_biencoder.sh <checkpoint_name> [split]"}
SPLIT=${2:-"test"}

echo "Checkpoint: $CHECKPOINT"
echo "Split:      $SPLIT"
echo "========================================================"

~/.conda/envs/tesis/bin/python -u scripts/evaluate_finetuned.py \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT"

echo "========================================================"
echo "Fin: $(date)"
