# Comandos del pipeline neuronal

Guía operativa de `entity-resolution-nlp`. Ejecute los comandos desde la raíz del proyecto y use `--help` para consultar opciones adicionales.

## 1. Alcance del repositorio
Este repositorio consume un `dataset.parquet` preparado y ejecuta exclusivamente partición, entrenamiento, evaluación y exportación de artefactos neuronales. No procesa datos crudos ni construye el ground truth.

```text
dataset.parquet preparado
        ↓
run_splitting.py
        ↓
run_train_biencoder.py
        ↓
evaluate_finetuned.py
        ↓
mine_hard_pairs.py
        ↓
train_crossencoder.py
        ↓
evaluate_crossencoder.py
        ↓
calibrate_crossencoder.py
```

Ramas complementarias: `evaluate_zeroshot.py`, `measure_token_distribution.py`, `plot_training_curves.py`, `visualize_embeddings.py`, `export_embeddings.py` y `sanity_check_paraphrase.py`.

## 2. Configuración y rutas
`INER_DATA_ROOT` define la raíz externa. Si no existe, `config.py` usa `~/Data/INER`. Para reutilizar esta guía en una misma terminal:

```bash
export INER_DATA_ROOT="${INER_DATA_ROOT:-$HOME/Data/INER}"
VARIANT=tok_skipnull
DATASET="$INER_DATA_ROOT/processed/default/output/$VARIANT/dataset.parquet"
SPLIT="$INER_DATA_ROOT/tesis/splits/${VARIANT}_split.parquet"
PAIRS_DIR="$INER_DATA_ROOT/tesis/splits/$VARIANT"
BE_RUN=beto_mnrl_hpc_v2_tok_skipnull
CE_RUN=beto_bce_hpc_v2_tok_skipnull
```

La raíz también puede persistirse en `.env` como `INER_DATA_ROOT=/ruta/a/INER`.

| Artefacto | Ruta |
|---|---|
| Dataset de entrada | `processed/default/output/<variant>/dataset.parquet` |
| Split por entidad | `tesis/splits/<variant>_split.parquet` |
| Modelos | `models/{pretrained,checkpoints}/` |
| Pares minados | `tesis/splits/<variant>/pairs_{train,val,test}.parquet` |
| Evaluaciones y figuras | `outputs/` |
| Embeddings exportados | `embeddings/` |

## 3. Mapa de scripts activos
| Script | Función |
|---|---|
| `download_model.py` | Descarga y empaqueta modelos locales |
| `run_splitting.py` | Divide entidades en train, val y test |
| `evaluate_zeroshot.py` | Evalúa modelos sin fine-tuning |
| `run_train_biencoder.py` | Entrena el Bi-Encoder |
| `evaluate_finetuned.py` | Evalúa checkpoints del Bi-Encoder |
| `measure_token_distribution.py` | Mide longitudes tokenizadas |
| `plot_training_curves.py` | Grafica historiales de entrenamiento |
| `visualize_embeddings.py` | Proyecta embeddings con UMAP |
| `mine_hard_pairs.py` | Genera pares para el Cross-Encoder |
| `train_crossencoder.py` | Entrena el Cross-Encoder |
| `evaluate_crossencoder.py` | Busca umbral o evalúa el Cross-Encoder |
| `calibrate_crossencoder.py` | Calibra y exporta incertidumbre |
| `export_embeddings.py` | Exporta embeddings por `record_id` |
| `sanity_check_paraphrase.py` | Verifica el baseline multilingüe |

Los siete `.sh` de la raíz son wrappers SLURM. Pase rutas explícitas: varios comentarios y defaults internos aún conservan nomenclatura legacy.

## 4. Preparación del entorno y modelos
```bash
micromamba activate tesis
uv pip install -e .
python scripts/download_model.py --all
python scripts/measure_token_distribution.py --model BETO --dataset "$DATASET"
```

Los modelos quedan en `$INER_DATA_ROOT/models/pretrained/`; sincronice ese directorio antes de trabajar en nodos sin internet. Para una descarga individual use `download_model.py --model <HUGGINGFACE_ID> --name <NOMBRE_LOCAL>`.

## 5. Partición por entidad
```bash
python scripts/run_splitting.py --variant "$VARIANT"
```

Produce `$SPLIT` con columna `split`. Con `--dataset` puede indicarse otro parquet, pero la salida seguirá nombrándose a partir de `--variant`.

## 6. Evaluación zero-shot
La evaluación zero-shot usa normalmente la variante sin tokens y sin nulos:

```bash
python scripts/evaluate_zeroshot.py --all \
  --dataset "$INER_DATA_ROOT/processed/default/output/notok_skipnull/dataset.parquet"
python scripts/sanity_check_paraphrase.py
```

Los resultados se escriben en `outputs/evaluation/`.

## 7. Entrenamiento del Bi-Encoder
Smoke test local:

```bash
python scripts/run_train_biencoder.py --model BETO --dataset "$SPLIT" \
  --output "$INER_DATA_ROOT/models/checkpoints/${BE_RUN}_smoke" \
  --epochs 1 --batch-size 8 --n-aug 0 --max-seq-length 384
```

Job SLURM con BETO:

```bash
sbatch --job-name="$BE_RUN" train_biencoder_beto.sh \
  0.07 "$BE_RUN" "$SPLIT" 64 20 384
```

Para RoBERTa use `train_biencoder_roberta.sh` con los mismos argumentos. Los checkpoints quedan en `models/checkpoints/<run>/`; `best/` es el checkpoint usual.

## 8. Evaluación y visualización del Bi-Encoder
```bash
python scripts/evaluate_finetuned.py --checkpoint "$BE_RUN" --dataset "$SPLIT" --split test
python scripts/plot_training_curves.py --checkpoint "$BE_RUN"
python scripts/visualize_embeddings.py --checkpoint "$BE_RUN" \
  --dataset "$SPLIT" --split test --dims 3
sbatch eval_biencoder.sh "$BE_RUN" test "$SPLIT"
```

## 9. Hard Negative Mining
Use un directorio por variante para no sobreescribir pares de otros experimentos:

```bash
python scripts/mine_hard_pairs.py --checkpoint "$BE_RUN" --dataset "$SPLIT" \
  --top-k 20 --output-dir "$PAIRS_DIR"
```

Produce `pairs_train.parquet`, `pairs_val.parquet` y `pairs_test.parquet`.

## 10. Entrenamiento del Cross-Encoder
```bash
python scripts/train_crossencoder.py --model BETO --dataset "$SPLIT" \
  --pairs-train "$PAIRS_DIR/pairs_train.parquet" \
  --pairs-val "$PAIRS_DIR/pairs_val.parquet" \
  --output "$INER_DATA_ROOT/models/checkpoints/$CE_RUN" \
  --epochs 3 --batch-size 16 --lr 2e-5 --max-seq-length 512 --only-best

sbatch --job-name="$CE_RUN" train_crossencoder_beto.sh "$CE_RUN" "$SPLIT" \
  "$PAIRS_DIR/pairs_train.parquet" "$PAIRS_DIR/pairs_val.parquet" 3 16
```

## 11. Evaluación y calibración del Cross-Encoder
Busque el umbral sobre validación, asígnelo a `THRESHOLD` y después evalúe test:

```bash
CE_BEST="$INER_DATA_ROOT/models/checkpoints/$CE_RUN/best"
python scripts/evaluate_crossencoder.py --checkpoint "$CE_BEST" --dataset "$SPLIT" \
  --pairs "$PAIRS_DIR/pairs_val.parquet" --find-threshold
THRESHOLD=0.00  # Sustituir por el valor obtenido en validación
python scripts/evaluate_crossencoder.py --checkpoint "$CE_BEST" --dataset "$SPLIT" \
  --pairs "$PAIRS_DIR/pairs_test.parquet" --threshold "$THRESHOLD"
python scripts/calibrate_crossencoder.py --checkpoint "$CE_BEST" --dataset "$SPLIT" \
  --val-pairs "$PAIRS_DIR/pairs_val.parquet" \
  --test-pairs "$PAIRS_DIR/pairs_test.parquet"
```

Wrappers disponibles: `eval_crossencoder.sh` y `calibrate_crossencoder.sh`.

## 12. Exportación de embeddings
La exportación usa el dataset completo, no el split. Indique la salida para no escribir dentro del directorio de entrada:

```bash
python scripts/export_embeddings.py --checkpoint "$BE_RUN" --dataset "$DATASET" \
  --output "$INER_DATA_ROOT/embeddings/${VARIANT}_embeddings.parquet"
```

`export_embeddings.sh` no acepta una ruta de salida y usa el nombre predeterminado junto al dataset; prefiera la CLI anterior para mantener separados los artefactos.
