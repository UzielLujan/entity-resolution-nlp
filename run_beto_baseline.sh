#!/bin/bash
#SBATCH --job-name=beto_mnrl_hpc_run_e
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"

TEMPERATURE=${1:-"0.05"}
RUN_NAME=${2:-"beto_mnrl_hpc_run_e"}
PARQUET=${3:-"~/Data/INER/processed/tesis1/dataset_split.parquet"}

~/.conda/envs/tesis/bin/python -u scripts/run_train_biencoder.py \
    --model BETO \
    --parquet $PARQUET \
    --output ~/Data/INER/models/checkpoints/$RUN_NAME \
    --epochs 20 \
    --batch-size 64 \
    --n-aug 0 \
    --max-seq-length 512 \
    --base-lr 2e-5 \
    --temperature $TEMPERATURE \
    --patience 3

echo "Fin: $(date)"
