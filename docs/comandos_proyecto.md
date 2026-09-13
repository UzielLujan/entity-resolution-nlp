# Comandos del pipeline neuronal

Guía operativa de `entity-resolution-nlp`. Ejecute los comandos desde la raíz del proyecto y use `--help` para consultar opciones adicionales.

## 1. Alcance del repositorio
Este repositorio consume un `dataset.parquet` preparado y ejecuta exclusivamente partición, entrenamiento, evaluación y exportación de artefactos neuronales. No procesa datos crudos ni construye el ground truth. El esquema, los invariantes y la propiedad de cada ruta se definen en [`data_contract.md`](data_contract.md).

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
`INER_DATA_ROOT` define la raíz externa y es obligatoria. Después de clonar el repositorio, cree la configuración local:

```bash
cp .env.example .env
# Edite .env y asigne una ruta absoluta a INER_DATA_ROOT.
```

`config.py` carga siempre el `.env` de la raíz del repositorio, independientemente del directorio desde el que se invoque Python. Para reutilizar las rutas en una terminal también puede cargarlo en el shell:

```bash
set -a
source .env
set +a
VARIANT=tok_skipnull
DATASET="$INER_DATA_ROOT/processed/default/output/$VARIANT/dataset.parquet"
SPLIT="$INER_DATA_ROOT/modeling/data/$VARIANT/split.parquet"
PAIRS_DIR="$INER_DATA_ROOT/modeling/data/$VARIANT"
BE_RUN=beto_mnrl_hpc_v2_tok_skipnull
CE_RUN=beto_bce_hpc_v2_tok_skipnull
EMBEDDINGS="$INER_DATA_ROOT/modeling/embeddings/$VARIANT/embeddings.parquet"
```

`.env` no se versiona. `.env.example` documenta la única variable requerida y la ubicación mínima de `dataset.parquet`. Valide la configuración antes de ejecutar el pipeline:

```bash
source .venv/bin/activate
python -m record_linkage.config
```

| Artefacto | Ruta |
|---|---|
| Dataset de entrada | `processed/default/output/<variant>/dataset.parquet` |
| Split por entidad | `modeling/data/<variant>/split.parquet` |
| Modelos preentrenados | `modeling/models/pretrained/` |
| Checkpoints | `modeling/models/{biencoder,crossencoder}/<variant>/<run>/` |
| Pares minados | `modeling/data/<variant>/pairs_{train,val,test}.parquet` |
| Evaluaciones y figuras | `modeling/outputs/` |
| Embeddings exportados | `modeling/embeddings/<variant>/embeddings.parquet` |

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

Los siete `.sh` de la raíz son wrappers exclusivos del clúster: localizan el repositorio, cargan `.env` y usan el comando `python` del entorno Conda activo al enviar el job. Los datasets requeridos se pasan como argumentos explícitos.

## 4. Preparación del entorno y modelos
```bash
uv sync
source .venv/bin/activate
python scripts/download_model.py --all
python scripts/measure_token_distribution.py --model BETO --dataset "$DATASET"
```

Los modelos quedan en `$INER_DATA_ROOT/modeling/models/pretrained/`; sincronice ese directorio antes de trabajar en nodos sin internet. Para una descarga individual use `download_model.py --model <HUGGINGFACE_ID> --name <NOMBRE_LOCAL>`.

## 5. Partición por entidad
```bash
python scripts/run_splitting.py --variant "$VARIANT"
```

Produce `$SPLIT` con columna `split`. Con `--dataset` puede indicarse otro parquet, pero la salida seguirá nombrándose a partir de `--variant`.

## 6. Evaluación zero-shot
La evaluación zero-shot oficial usa la variante sin tokens y sin nulos sobre el
split de prueba:

```bash
.venv/bin/python scripts/evaluate_zeroshot.py --all \
  --dataset "$INER_DATA_ROOT/modeling/data/notok_skipnull/split.parquet" \
  --split test --batch-size 32
.venv/bin/python scripts/sanity_check_paraphrase.py
```

Los resultados se escriben en `modeling/outputs/evaluation/biencoder/zeroshot/`.

## 7. Entrenamiento del Bi-Encoder
Smoke test local:

```bash
python scripts/run_train_biencoder.py --model BETO --variant "$VARIANT" --dataset "$SPLIT" \
  --output "$INER_DATA_ROOT/modeling/models/biencoder/$VARIANT/${BE_RUN}_smoke" \
  --epochs 1 --batch-size 8 --n-aug 0 --max-seq-length 384
```

Job SLURM con BETO:

```bash
sbatch --job-name="$BE_RUN" train_biencoder_beto.sh \
  0.07 "$BE_RUN" "$SPLIT" 64 20 384
```

Para RoBERTa use `train_biencoder_roberta.sh` con los mismos argumentos. Los checkpoints quedan en `modeling/models/biencoder/<variant>/<run>/`; `best/` es el checkpoint usual.

## 8. Evaluación y visualización del Bi-Encoder
```bash
python scripts/evaluate_finetuned.py --checkpoint "$BE_RUN" --variant "$VARIANT" \
  --dataset "$SPLIT" --split test
python scripts/plot_training_curves.py --checkpoint "$BE_RUN" --variant "$VARIANT"
python scripts/visualize_embeddings.py --variant "$VARIANT" \
  --dataset "$SPLIT" --embeddings "$EMBEDDINGS" --split test --dims 3
sbatch eval_biencoder.sh "$BE_RUN" test "$SPLIT"
```

El visualizador escribe `embedding_space_<split>_<dims>d.html`. Los sufijos
opcionales son `_linkable`, `_dark` y `_highlight-<entity_id>`. Las opciones de
`--split` son `train`, `val`, `test` y `all`; el título del gráfico solo reporta
el número de registros mostrados y si se filtró a entidades vinculables.

## 9. Hard Negative Mining
Por defecto, los pares se escriben en el directorio canónico de la variante:

```bash
python scripts/mine_hard_pairs.py --checkpoint "$BE_RUN" --variant "$VARIANT" \
  --dataset "$SPLIT" \
  --top-k 20
```

Produce `pairs_train.parquet`, `pairs_val.parquet` y `pairs_test.parquet` en
`$INER_DATA_ROOT/modeling/data/$VARIANT`. `--output-dir` permite sobrescribir esta ruta
para ejecuciones experimentales.

## 10. Entrenamiento del Cross-Encoder
```bash
python scripts/train_crossencoder.py --model BETO --variant "$VARIANT" --dataset "$SPLIT" \
  --pairs-train "$PAIRS_DIR/pairs_train.parquet" \
  --pairs-val "$PAIRS_DIR/pairs_val.parquet" \
  --output "$INER_DATA_ROOT/modeling/models/crossencoder/$VARIANT/$CE_RUN" \
  --epochs 3 --batch-size 16 --lr 2e-5 --max-seq-length 512 --only-best

sbatch --job-name="$CE_RUN" train_crossencoder_beto.sh "$CE_RUN" "$SPLIT" \
  "$PAIRS_DIR/pairs_train.parquet" "$PAIRS_DIR/pairs_val.parquet" 3 16
```

## 11. Evaluación y calibración del Cross-Encoder
Busque el umbral sobre validación, asígnelo a `THRESHOLD` y después evalúe test:

```bash
CE_BEST="$INER_DATA_ROOT/modeling/models/crossencoder/$VARIANT/$CE_RUN/best"
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
La exportación usa el dataset completo, no el split. La variante se infiere del directorio que contiene el dataset y la salida queda fuera del directorio de entrada:

```bash
python scripts/export_embeddings.py --checkpoint "$BE_RUN" --variant "$VARIANT" \
  --dataset "$DATASET"
sbatch export_embeddings.sh "$BE_RUN" "$DATASET"
```

Ambos comandos producen `$EMBEDDINGS`. Use `--output` en la CLI solo cuando necesite una ruta explícita distinta del contrato canónico.
