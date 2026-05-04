# Comandos del Pipeline de Datos

Flujo completo desde los CSVs crudos hasta el consumo en entrenamiento.
Todas las rutas de salida son relativas a `~/Data/INER/`.

---

## Comportamiento de rutas por script

| Script | Entrada | Salida | Rutas explícitas necesarias |
|---|---|---|---|
| `run_preprocessing.py` | `RAW_FILES` en `config.py` (fija) | `processed/<perfil>/` | Ninguna — deriva todo de `--perfil` |
| `run_dataset.py` | `processed/<perfil>/` | `processed/<perfil>[_sin_tokens]/` | Ninguna — deriva todo de `--perfil` y flags |
| `run_splitting.py` | `processed/<perfil>/dataset.parquet` | mismo directorio, `dataset_split.parquet` | Ninguna — deriva todo de `--perfil` |
| `evaluate_zeroshot.py` | `processed/tesis0_sin_tokens/dataset.parquet` | `outputs/evaluation/` | Opcional: `--dataset-path` si se usa otro parquet |
| `run_train_biencoder.py` | `processed/tesis1/dataset_split.parquet` | `models/checkpoints/<model>_mnrl/` | **Recomendado** pasar `--output` para nombrar el run; `--parquet` si el perfil no es `tesis1` |

> Los únicos scripts donde vale la pena pasar rutas explícitas son `evaluate_zeroshot.py`
> (cuando se evalúa un dataset no estándar) y `run_train_biencoder.py` (para nombrar el run
> con sentido, ej. `beto_mnrl_run05_hpc`, y evitar sobreescribir checkpoints anteriores).

---

## Etapa 0 — Descarga de modelos preentrenados

Descargar antes de cualquier entrenamiento local o en HPC (los nodos de cómputo no tienen internet).

```bash
# Un modelo específico
python scripts/download_model.py --model dccuchile/bert-base-spanish-wwm-cased --name BETO
python scripts/download_model.py --model PlanTL-GOB-ES/roberta-base-biomedical-clinical-es --name RoBERTa-biomedical
python scripts/download_model.py --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --name paraphrase-multilingual

# Todos los modelos conocidos de una vez
python scripts/download_model.py --all
```

**Salida:** `models/pretrained/<name>/`

---

## Etapa 1 — Preprocesamiento (`run_preprocessing.py`)

Lee los CSVs crudos del INER y produce CSVs limpios por perfil.

```bash
# Perfil iner   — limpieza completa para entregables INER (consultoría)
python scripts/run_preprocessing.py --perfil iner

# Perfil tesis0 — intervención mínima; base para el modo Sin Tokens (evaluación zero-shot)
python scripts/run_preprocessing.py --perfil tesis0

# Perfil tesis1 — mínima intervención para fine-tuning (perfil principal de tesis)
python scripts/run_preprocessing.py --perfil tesis1

# Perfil tesis2 — limpieza + renombrado semántico de columnas
python scripts/run_preprocessing.py --perfil tesis2

# Verificar rutas antes de correr
python scripts/run_preprocessing.py --check-paths
```

**Salida:** `processed/<perfil>/{comorbilidad_clean.csv, econo_clean.csv, trabajo_social_clean.csv}`

---

## Etapa 2 — Construcción del dataset (`run_dataset.py`)

Serializa los CSVs limpios en registros con tokens especiales, asigna `entity_id` y guarda un `.parquet`.

### Fine-tuning (con tokens especiales `[BLK_*]`)

```bash
# Perfil estándar para entrenamiento del Bi-Encoder
python scripts/run_dataset.py --perfil tesis1

# Perfil tesis2 (limpieza más agresiva)
python scripts/run_dataset.py --perfil tesis2

# Validar rutas antes de construir
python scripts/run_dataset.py --perfil tesis1 --check-paths
```

**Salida:** `processed/tesis1/dataset.parquet` (o `tesis2/`)

### Sin tokens especiales — serialización `Clave: Valor`

```bash
# Dataset completo sin tokens (para evaluación sin fine-tuning)
python scripts/run_dataset.py --perfil tesis0 --no-special-tokens
```

**Salida:** `processed/tesis0_sin_tokens/dataset.parquet`

### Solo nombres — campo nombre únicamente

```bash
# Dataset reducido: text = solo el campo nombre de cada registro
# Requiere --perfil tesis0 y --no-special-tokens
python scripts/run_dataset.py --perfil tesis0 --no-special-tokens --solo-nombres
```

**Salida:** `processed/tesis0_sin_tokens_solo_nombres/dataset.parquet`

---

## Etapa 3 — Partición train/val/test (`run_splitting.py`)

Divide el dataset a nivel de entidad (sin data leakage). Produce un único parquet con columna `split`.

```bash
# Perfil principal de tesis
python scripts/run_splitting.py --perfil tesis1

# Con proporciones personalizadas
python scripts/run_splitting.py --perfil tesis1 --train 0.70 --val 0.15 --seed 42

# Para el dataset sin tokens (evaluación zero-shot)
python scripts/run_splitting.py --perfil tesis0_sin_tokens

# Otros perfiles disponibles
python scripts/run_splitting.py --perfil tesis0
python scripts/run_splitting.py --perfil tesis2
python scripts/run_splitting.py --perfil iner
```

**Salida:** `processed/<perfil>/dataset_split.parquet`

---

## Etapa 4A — Evaluación zero-shot (`evaluate_zeroshot.py`)

Evalúa modelos preentrenados **sin fine-tuning** sobre los pares residuales del INER.
Métricas: Recall@K (K=1,5,10,20,50) y MRR.

```bash
# Un modelo
python scripts/evaluate_zeroshot.py --model BETO
python scripts/evaluate_zeroshot.py --model RoBERTa-biomedical
python scripts/evaluate_zeroshot.py --model paraphrase-multilingual

# Varios modelos en una pasada
python scripts/evaluate_zeroshot.py --model BETO --model RoBERTa-biomedical

# Todos los modelos conocidos
python scripts/evaluate_zeroshot.py --all

# Dataset alternativo
python scripts/evaluate_zeroshot.py --model BETO --dataset-path /ruta/custom/dataset.parquet
```

**Dataset por defecto:** `processed/tesis0_sin_tokens/dataset.parquet`  
**Salida:** `outputs/evaluation/zeroshot_<model>.json`

---

## Etapa 4B — Sanity check paraphrase

Verifica que `paraphrase-multilingual` funciona correctamente antes de interpretar sus métricas.
Sin argumentos.

```bash
python scripts/sanity_check_paraphrase.py
```

---

## Etapa 5 — Entrenamiento Bi-Encoder (`run_train_biencoder.py`)

Entrena el Bi-Encoder con MNRL. Consume `dataset_split.parquet` (Etapa 3).

**Parámetros clave:**

| Flag | Default | Descripción |
|---|---|---|
| `--model` | BETO | Nombre del modelo en `models/pretrained/` |
| `--parquet` | `tesis1/dataset_split.parquet` | Ruta al parquet con columna `split` |
| `--output` | `models/checkpoints/<model>_mnrl` | Directorio de checkpoints |
| `--epochs` | 2 | Épocas de entrenamiento |
| `--batch-size` | 8 | Tamaño de batch (usar 64+ en HPC) |
| `--n-aug` | 0 | Pares sintéticos por registro (0 = solo naturales) |
| `--max-seq-length` | 384 | Tokens máximos (384 local, 512 HPC) |
| `--viz` | False | Guarda matrices MNRL de 3 batches de la época 1 |

**Salida:** `models/checkpoints/<output>/epoch_NN/` + `training_history.json`

### Smoke test local (1 época, sin augmentación)

```bash
python scripts/run_train_biencoder.py \
    --model BETO \
    --output ~/Data/INER/models/checkpoints/beto_mnrl_smoke \
    --epochs 1 --batch-size 8 --n-aug 0 --max-seq-length 384 --viz
```

### Local con dataset alternativo

```bash
# Ejemplo: dataset sin tokens especiales
python scripts/run_train_biencoder.py \
    --model BETO \
    --parquet ~/Data/INER/processed/tesis0_sin_tokens/dataset_split.parquet \
    --output ~/Data/INER/models/checkpoints/beto_mnrl_sin_tokens \
    --epochs 1 --batch-size 8 --n-aug 0 --max-seq-length 384
```

### HPC — baseline limpio (primer job real)

```bash
python scripts/run_train_biencoder.py \
    --model BETO \
    --output ~/Data/INER/models/checkpoints/beto_mnrl_hpc_baseline \
    --epochs 10 --batch-size 64 --n-aug 0 --max-seq-length 512
```

### HPC — con augmentación (una vez rediseñada)

```bash
python scripts/run_train_biencoder.py \
    --model BETO \
    --output ~/Data/INER/models/checkpoints/beto_mnrl_hpc_aug \
    --epochs 10 --batch-size 64 --n-aug 2 --max-seq-length 512
```

### HPC — RoBERTa-biomedical (tras confirmar BETO)

```bash
python scripts/run_train_biencoder.py \
    --model RoBERTa-biomedical \
    --output ~/Data/INER/models/checkpoints/roberta_bio_hpc_baseline \
    --epochs 10 --batch-size 64 --n-aug 0 --max-seq-length 512
```

> **Convención de nombres para `--output`:** `<modelo>_mnrl_<entorno>_<variante>`
> Ejemplos: `beto_mnrl_hpc_baseline`, `beto_mnrl_hpc_aug_v2`, `roberta_bio_hpc_baseline`
> Esto evita sobreescribir checkpoints anteriores y facilita comparar runs en el historial.

---

## Flujo completo (ejemplo tesis1)

```bash
# 1. Descargar modelos (una sola vez)
python scripts/download_model.py --all

# 2. Preprocesar
python scripts/run_preprocessing.py --perfil tesis1

# 3. Construir dataset
python scripts/run_dataset.py --perfil tesis1

# 4. Particionar
python scripts/run_splitting.py --perfil tesis1

# 5. Entrenar
python scripts/run_train_biencoder.py --model BETO --epochs 10 --batch-size 64 --n-aug 0
```

## Flujo completo (evaluación zero-shot)

```bash
# 1. Preprocesar con perfil base
python scripts/run_preprocessing.py --perfil tesis0

# 2. Construir dataset sin tokens
python scripts/run_dataset.py --perfil tesis0 --no-special-tokens

# 3. Evaluar modelos preentrenados
python scripts/evaluate_zeroshot.py --all
```
