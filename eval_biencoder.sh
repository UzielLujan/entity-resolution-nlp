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

# Evalúa un checkpoint fine-tuneado del Bi-Encoder sobre el split indicado.
# Parámetros (en orden):
#   $1 CHECKPOINT  obligatorio — nombre del run canónico del Bi-Encoder
#   $2 SPLIT       default "test"
#   $3 DATASET     opcional — ruta al split.parquet (default: variante canónica)
#
# Uso típico (con dataset v2):
#   sbatch eval_biencoder.sh beto_mnrl_hpc_v2_tok_skipnull test \
#       ~/Data/INER/modeling/data/tok_skipnull/split.parquet

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"

CHECKPOINT=${1:?"Uso: sbatch eval_biencoder.sh <checkpoint_name> [split] [dataset]"}
SPLIT=${2:-"test"}
DATASET=${3:-""}

echo "Checkpoint: $CHECKPOINT"
echo "Split:      $SPLIT"
echo "Dataset:    ${DATASET:-default de la variante canónica}"
echo "========================================================"

if [ -n "$DATASET" ]; then
    ~/.conda/envs/tesis/bin/python -u scripts/evaluate_finetuned.py \
        --checkpoint "$CHECKPOINT" \
        --split "$SPLIT" \
        --dataset "$DATASET"
else
    ~/.conda/envs/tesis/bin/python -u scripts/evaluate_finetuned.py \
        --checkpoint "$CHECKPOINT" \
        --split "$SPLIT"
fi

echo "========================================================"
echo "Fin: $(date)"
