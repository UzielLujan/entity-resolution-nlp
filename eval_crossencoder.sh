#!/bin/bash
#SBATCH --job-name=eval_crossencoder
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

# Evalúa el Cross-Encoder sobre un parquet de pares etiquetados.
# Parámetros posicionales (en orden):
#   $1 MODE         "test" (default) o "find-threshold"
#   $2 CHECKPOINT   Ruta al checkpoint del CE (con best/ adentro)
#   $3 DATASET      Parquet con cols record_id + text (para lookup)
#   $4 PAIRS        Parquet de pares (record_id_a, record_id_b, label)
#   $5 THRESHOLD    Umbral binarización (default 0.5; ignorado en find-threshold)
#
# Uso típico:
#   sbatch eval_crossencoder.sh test ~/Data/INER/models/checkpoints/beto_bce_hpc_v2_tok_skipnull/best \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/dataset.parquet \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_test.parquet

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE=${1:-"test"}
CHECKPOINT=${2:?"Falta CHECKPOINT (ruta al best/ del CE)"}
DATASET=${3:?"Falta DATASET (parquet de registros)"}
PAIRS=${4:?"Falta PAIRS (parquet de pares)"}
THRESHOLD=${5:-"0.5"}

echo "Mode:       $MODE"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "Pairs:      $PAIRS"
echo "Threshold:  $THRESHOLD"
echo "========================================================"

if [ "$MODE" = "find-threshold" ]; then
    ~/.conda/envs/tesis/bin/python -u scripts/evaluate_crossencoder.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "$DATASET" \
        --pairs "$PAIRS" \
        --find-threshold
else
    ~/.conda/envs/tesis/bin/python -u scripts/evaluate_crossencoder.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "$DATASET" \
        --pairs "$PAIRS" \
        --threshold "$THRESHOLD"
fi

echo "========================================================"
echo "Fin: $(date)"
