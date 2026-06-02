#!/bin/bash
#SBATCH --job-name=calibrate_ce
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

# Calibración del Cross-Encoder (Vía A): temperature scaling + incertidumbre por vínculo.
# POST-HOC sobre el CE ya entrenado — NO re-entrena.
#
# Parámetros posicionales (en orden):
#   $1 CHECKPOINT   Ruta al best/ del CE
#   $2 DATASET      Parquet con cols record_id + text (lookup)
#   $3 VAL_PAIRS    Parquet de pares de validación (ajuste de T)
#   $4 TEST_PAIRS   Parquet de pares de test (incertidumbre por vínculo)
#
# Uso típico:
#   sbatch calibrate_crossencoder.sh \
#       ~/Data/INER/models/checkpoints/beto_bce_hpc_v2_tok_skipnull/best \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/dataset.parquet \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_val.parquet \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_test.parquet

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT=${1:?"Falta CHECKPOINT (ruta al best/ del CE)"}
DATASET=${2:?"Falta DATASET (parquet de registros)"}
VAL_PAIRS=${3:?"Falta VAL_PAIRS (parquet de pares de validación)"}
TEST_PAIRS=${4:?"Falta TEST_PAIRS (parquet de pares de test)"}

echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "Val pairs:  $VAL_PAIRS"
echo "Test pairs: $TEST_PAIRS"
echo "========================================================"

~/.conda/envs/tesis/bin/python -u scripts/calibrate_crossencoder.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --val-pairs "$VAL_PAIRS" \
    --test-pairs "$TEST_PAIRS"

echo "========================================================"
echo "Fin: $(date)"
