#!/bin/bash
#SBATCH --job-name=train_crossencoder_beto
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --chdir=/home/est_posgrado_uziel.lujan/Projects/entity-resolution-nlp
#SBATCH --output=logs/%x-%j.log

# Entrena el Cross-Encoder con BETO + BCE sobre pairs minados (HNM del BE ganador).
# Hiperparámetros congelados:
#   model=BETO, lr=2e-5, epochs=3, batch=16, max_seq=512, pos_weight=8, patience=2
# Lo que SÍ se parametriza desde sbatch (en orden):
#   $1 RUN_NAME      default beto_bce_hpc_v2
#   $2 DATASET       parquet de registros (cols record_id, text)
#   $3 PAIRS_TRAIN   parquet de pares train
#   $4 PAIRS_VAL     parquet de pares val
#   $5 EPOCHS        default 3
#   $6 BATCH_SIZE    default 16
#
# Uso típico:
#   sbatch --job-name=beto_bce_hpc_v2_tok_skipnull train_crossencoder_beto.sh \
#       beto_bce_hpc_v2_tok_skipnull \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/dataset.parquet \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_train.parquet \
#       ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_val.parquet

set -e
mkdir -p logs

echo "========================================================"
echo "Job ID: $SLURM_JOB_ID | Host: $(hostname)"
echo "Inicio: $(date)"
echo "========================================================"

export PATH="/opt/anaconda_python311/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME=${1:-"beto_bce_hpc_v2"}
DATASET=${2:?"Falta DATASET (parquet de registros con cols record_id, text)"}
PAIRS_TRAIN=${3:?"Falta PAIRS_TRAIN (parquet con record_id_a, record_id_b, label)"}
PAIRS_VAL=${4:?"Falta PAIRS_VAL"}
EPOCHS=${5:-"3"}
BATCH_SIZE=${6:-"16"}

echo "Config: model=BETO epochs=$EPOCHS batch=$BATCH_SIZE run=$RUN_NAME"
echo "Dataset:     $DATASET"
echo "Pairs train: $PAIRS_TRAIN"
echo "Pairs val:   $PAIRS_VAL"
echo "========================================================"

~/.conda/envs/tesis/bin/python -u scripts/train_crossencoder.py \
    --model BETO \
    --dataset "$DATASET" \
    --pairs-train "$PAIRS_TRAIN" \
    --pairs-val "$PAIRS_VAL" \
    --output ~/Data/INER/models/checkpoints/$RUN_NAME \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 2e-5 \
    --max-seq-length 512 \
    --pos-weight 8 \
    --patience 2 \
    --only-best

echo "Fin: $(date)"
