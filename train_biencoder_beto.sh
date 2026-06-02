#!/bin/bash
#SBATCH --job-name=train_biencoder_beto
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

# Entrena el Bi-Encoder con BETO + MNRL sobre un dataset_split.parquet dado.
# Hiperparámetros congelados a la config ganadora de Run E (val_loss=1.0224 sobre v1):
#   lr=2e-5, n-aug=0, max-seq=512, patience=3, epochs=20
# Lo que SÍ se parametriza desde sbatch (en orden):
#   $1 TEMPERATURE  default 0.07
#   $2 RUN_NAME     default beto_mnrl_v2
#   $3 PARQUET      default tesis1/dataset_split.parquet
#   $4 BATCH_SIZE   default 64
#   $5 EPOCHS       default 20
#   $6 MAX_SEQ      default 384  (medido: max real ≤ 368 tokens en todas las variantes v2)
#
# Uso típico (override con sbatch --job-name=...):
#   sbatch --job-name=beto_mnrl_v2_tok_skipnull train_biencoder_beto.sh \
#       0.07 beto_mnrl_v2_tok_skipnull \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/dataset_split.parquet \
#       64 20 384

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"

# Mitiga fragmentación de memoria CUDA — permite a PyTorch usar bloques más grandes
# en vez de fragmentar el pool. Crítico al borde del límite de memoria de la GPU.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEMPERATURE=${1:-"0.07"}
RUN_NAME=${2:-"beto_mnrl_v2"}
PARQUET=${3:-"~/Data/INER/processed/tesis1/dataset_split.parquet"}
BATCH_SIZE=${4:-"64"}
EPOCHS=${5:-"20"}
MAX_SEQ=${6:-"384"}

echo "Config: model=BETO temp=$TEMPERATURE batch=$BATCH_SIZE epochs=$EPOCHS max_seq=$MAX_SEQ run=$RUN_NAME"
echo "Dataset: $PARQUET"
echo "========================================================"

~/.conda/envs/tesis/bin/python -u scripts/run_train_biencoder.py \
    --model BETO \
    --dataset $PARQUET \
    --output ~/Data/INER/models/checkpoints/$RUN_NAME \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --n-aug 0 \
    --max-seq-length $MAX_SEQ \
    --base-lr 2e-5 \
    --temperature $TEMPERATURE \
    --patience 3 \
    --only-best

echo "Fin: $(date)"
