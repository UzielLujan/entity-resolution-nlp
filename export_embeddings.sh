#!/bin/bash
#SBATCH --job-name=export_embeddings
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

# Exporta los embeddings del Bi-Encoder a embeddings.parquet [record_id, embedding].
# Parámetros (en orden):
#   $1 CHECKPOINT  obligatorio — nombre del run canónico del Bi-Encoder
#   $2 DATASET     obligatorio — ruta al dataset.parquet COMPLETO (no el _split)
#
# Uso típico (variante canónica tok_skipnull):
#   sbatch export_embeddings.sh beto_mnrl_hpc_v2_tok_skipnull \
#       ~/Data/INER/processed/default/output/tok_skipnull/dataset.parquet

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"

CHECKPOINT=${1:?"Uso: sbatch export_embeddings.sh <checkpoint_name> <dataset.parquet>"}
DATASET=${2:?"Uso: sbatch export_embeddings.sh <checkpoint_name> <dataset.parquet>"}

echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "========================================================"

~/.conda/envs/tesis/bin/python -u scripts/export_embeddings.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --batch-size 64

echo "========================================================"
echo "Fin: $(date)"
