#!/bin/bash
#SBATCH --job-name=train_biencoder_roberta
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

# Entrena el Bi-Encoder con RoBERTa-biomedical + MNRL sobre un dataset_split.parquet.
# Hiperparámetros congelados a la config de RoBERTa Run B (val=1.0313 sobre v1).
# Parámetros desde sbatch (en orden):
#   $1 TEMPERATURE  default 0.07
#   $2 RUN_NAME     default roberta_bio_v2
#   $3 PARQUET      default tesis1/dataset_split.parquet
#   $4 BATCH_SIZE   default 64
#
# Uso típico:
#   sbatch --job-name=roberta_bio_v2_tok_skipnull train_biencoder_roberta.sh \
#       0.07 roberta_bio_v2_tok_skipnull \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/dataset_split.parquet \
#       64

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"

TEMPERATURE=${1:-"0.07"}
RUN_NAME=${2:-"roberta_bio_v2"}
PARQUET=${3:-"~/Data/INER/processed/tesis1/dataset_split.parquet"}
BATCH_SIZE=${4:-"64"}
EPOCHS=${5:-"20"}
MAX_SEQ=${6:-"384"}

echo "Config: model=RoBERTa-biomedical temp=$TEMPERATURE batch=$BATCH_SIZE epochs=$EPOCHS max_seq=$MAX_SEQ run=$RUN_NAME"
echo "Dataset: $PARQUET"
echo "========================================================"

~/.conda/envs/tesis/bin/python -u scripts/run_train_biencoder.py \
    --model RoBERTa-biomedical \
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
