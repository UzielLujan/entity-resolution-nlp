# Comandos del Pipeline de Datos

Flujo completo desde los CSVs crudos hasta el consumo en entrenamiento.
Todas las rutas de salida son relativas a `~/Data/INER/`.

---

## Comportamiento de rutas por script

| Script | Entrada | Salida | Rutas explícitas necesarias |
|---|---|---|---|
| `run_preprocessing.py` | `RAW_FILES` en `config.py` (fija) | `processed/<perfil>/` | Ninguna — deriva todo de `--perfil` |
| `run_dataset.py` | `processed/<perfil>/` | `processed/<perfil>[_sin_tokens]/` | Ninguna — deriva todo de `--perfil` y flags |
| `run_dataset_v2.py` | `processed/<perfil>/*_clean.csv` | `processed/<perfil>/{dataset_v2.parquet, pairs_for_review.xlsx, interim/}` | Ninguna — deriva todo de `--perfil` |
| `run_splitting.py` | `processed/<perfil>/dataset.parquet` | mismo directorio, `<stem>_split.parquet` | Opcional: `--dataset` para usar `dataset_v2.parquet` u otro |
| `evaluate_zeroshot.py` | `processed/tesis0_sin_tokens/dataset.parquet` | `outputs/evaluation/` | Opcional: `--dataset` si se usa otro parquet |
| `run_train_biencoder.py` | `processed/tesis1/dataset_split.parquet` | `models/checkpoints/<model>_mnrl/` | **Recomendado** pasar `--output` para nombrar el run; `--dataset` si el perfil no es `tesis1` |
| `evaluate_finetuned.py` | `processed/tesis1/dataset_split.parquet` | `outputs/evaluation/finetuned/` | Opcional: `--dataset`; usar `--checkpoint` o `--all` |
| `plot_training_curves.py` | `models/checkpoints/<run>/training_history.json` | `outputs/figures/` | Ninguna — deriva todo de `--checkpoint` |
| `build_consolidated_json.py` | `tesis/{output/<variant>/dataset.parquet, interim/records_interim.parquet}` + `RAW_FILES` | `iner/consolidated_entities_<v>.json` | Opcional: `--source-perfil`, `--out-perfil`, `--variant`, `--schema-version` |
| `build_data_dictionary.py` | `docs/consolidated_entities.schema.json` + `comparison_methods.REGISTRY` | `iner/{Diccionario_Final_INER.csv, metodos_comparacion.json, consolidated_entities.schema.json}` | Opcional: `--out-perfil` |

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

## Etapa 2-bis — Pipeline de etiquetado v2 (`run_dataset_v2.py`)

Alternativa a la Etapa 2 con clasificación explícita de pares cross-CSV y revisión manual.
Produce el mismo esquema de parquet que v1 (`record_id, source_db, text, entity_id`) — Etapa 3 en adelante consume el output igual (ver nota sobre integración con `run_splitting.py` al final).

### Diferencias clave vs v1 (`run_dataset.py`)

| Aspecto | v1 (`dataset.py`) | v2 (`dataset_v2.py`) |
|---|---|---|
| Asignación de `entity_id` | groupby `(exp, nombre_norm)` directo | union-find sobre pares positivos + intra-source grouping |
| Clasificación de pares | Implícita (todo o nada) | Explícita: `llave_exacta`, `metrica_clasica`, `no_confirmado`, `revision_manual` |
| Revisión manual | No | Sí — vía `pairs_for_review.xlsx` editable |
| Trazabilidad | Solo `entity_id` final | `criterio` + `decision` por par, auditable |
| Capa semántica (BETO) | — | Descartada empíricamente (zero-shot da falsos positivos masivos sobre nombres en español) |

### Flujo de dos pasos

```bash
# Paso 1: clasificar pares y producir xlsx editable
python scripts/run_dataset_v2.py --step classify --perfil tesis1

# [Revisión manual: abrir pairs_for_review.xlsx, marcar 'decision' = match / no_match
#  en los 'no_confirmado' que merezcan veredicto humano. Guardar con Ctrl+S.]

# Paso 2: leer xlsx editado, aplicar transiciones, producir parquet final
python scripts/run_dataset_v2.py --step finalize --perfil tesis1
```

### Salidas

```
~/Data/INER/processed/<perfil>/
├── pairs_for_review.xlsx        ← superficie de edición humana
├── dataset_v2.parquet           ← entregable tesis (esquema v1 compatible)
└── interim/
    ├── records_interim.parquet  ← internal del pipeline
    └── pairs_classified.parquet ← auditoría del estado original (sin decisiones)
```

### Esquema del xlsx editable

Columnas (las primeras dos ocultas, las demás visibles):

| record_id_a (hidden) | record_id_b (hidden) | source_a | source_b | exp | nombre_norm_a | nombre_norm_b | jw | lev | criterio | decision |

Después de `--step finalize` se agregan al final: `entity_id_a`, `entity_id_b` (para auditar el resultado del union-find).

**Validación de la columna `decision`**: dropdown con valores `match` / `no_match` (o vacío). Vacío significa "usa lo que dijo el pipeline".

**Paleta visual** (filas, excepto columna `decision` que queda blanca):
- 🟢 verde: `decision=match` o `criterio` auto-confirmado con decision vacío
- 🔴 rojo: `decision=no_match`
- 🟡 amarillo: `criterio=no_confirmado` con decision vacío (pendiente)

### Reglas de transición en finalize

| Estado pre-finalize | Estado post-finalize |
|---|---|
| `criterio` auto-confirmado, `decision` vacío | mismo `criterio`, `decision=match` |
| `criterio=no_confirmado`, `decision=match`/`no_match` | `criterio=revision_manual`, `decision` preservada |
| `criterio=no_confirmado`, `decision` vacío | sin cambio (pendiente, dispara ⚠ warning) |
| `criterio` auto-confirmado, `decision=no_match` (override raro) | `decision` preservada como `no_match` |

### Defaults calibrados empíricamente

| Flag | Default | Observación |
|---|---|---|
| `--umbral-jw` | 0.88 | Calibrado sobre los 9,855 `llave_exacta` — distingue typos reales de personas distintas |
| `--umbral-lev` | 0.85 | Mismo proceso |

### Flags

| Flag | Default | Descripción |
|---|---|---|
| `--step` | (requerido) | `classify` o `finalize` |
| `--perfil` | `tesis1` | Determina la ruta `processed/<perfil>/` |
| `--econo` / `--comor` / `--ts` | None | Override de rutas a CSVs limpios |
| `--output` | None | Override del directorio de salida |
| `--umbral-jw` / `--umbral-lev` | 0.88 / 0.85 | Override de umbrales calibrados |
| `--no-special-tokens` | False | Serializar sin tokens `[BLK_*]` (modo zero-shot) |

---

## Etapa 2-ter — Entregables de consultoría INER

Construyen los dos entregables finales del eje consultoría (schema **v2**) a partir de los artefactos
de etiquetado del perfil canónico `tesis` (`output/<variant>/dataset.parquet` → entity_id,
`interim/records_interim.parquet`) y los CSV crudos. Escriben en el perfil **`iner`** (carpeta separada
de los ejes tesis).

```bash
# JSON consolidado entity-centric v2 (Producto 3 — Base de Datos Consolidada)
python scripts/build_consolidated_json.py                 # tesis → iner/consolidated_entities_v2.json (indent=2)
python scripts/build_consolidated_json.py --indent -1     # JSON compacto (sin sangría)
python scripts/build_consolidated_json.py --schema-version v1   # → consolidated_entities_v1.json (histórico)

# Documentación del entregable (Producto 4): proyecta el schema + catálogo de métodos
python scripts/build_data_dictionary.py                   # → Diccionario_Final_INER.csv + metodos_comparacion.json + copia del schema
```

**Salidas (en `processed/iner/`):**
- `consolidated_entities_v2.json` (oficial) y `consolidated_entities_v1.json` (histórico): un objeto por
  `entity_id` con `cluster_size`, `decision`, `items[]` (`{item, source, linking_values, record}`), `scores[]`.
- `consolidated_entities.schema.json`: el **JSON Schema** formal del entregable (spec validable; fuente de
  verdad de las descripciones, se edita a mano en `docs/`).
- `Diccionario_Final_INER.csv`: vista plana del schema (`campo|tipo|descripcion`, rutas con puntos), **derivada
  del schema** por `build_data_dictionary.py` — no se edita a mano.
- `metodos_comparacion.json`: catálogo de métodos de `scores`, desde `comparison_methods.REGISTRY`.

Detalle del schema en `propuesta_entregable_JSON.md` → "Diseño v2".

---

## Etapa 3 — Partición train/val/test (`run_splitting.py`)

Divide el dataset a nivel de entidad (sin data leakage). Produce un único parquet con columna `split`.

```bash
# Perfil principal de tesis (lee dataset.parquet)
python scripts/run_splitting.py --perfil tesis1

# Dataset v2 (etiquetado robusto post-Ruta A)
python scripts/run_splitting.py --perfil tesis1 --dataset dataset_v2.parquet

# Con proporciones personalizadas
python scripts/run_splitting.py --perfil tesis1 --train 0.70 --val 0.15 --seed 42

# Para el dataset sin tokens (evaluación zero-shot)
python scripts/run_splitting.py --perfil tesis0_sin_tokens

# Otros perfiles disponibles
python scripts/run_splitting.py --perfil tesis0
python scripts/run_splitting.py --perfil tesis2
python scripts/run_splitting.py --perfil iner
```

**Salida:** `processed/<perfil>/<stem>_split.parquet` (ej. `dataset_split.parquet`, `dataset_v2_split.parquet`)

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
python scripts/evaluate_zeroshot.py --model BETO --dataset /ruta/custom/dataset.parquet
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

## Etapa 4C — Evaluación del Bi-Encoder fine-tuneado (`evaluate_finetuned.py`)

Evalúa checkpoints entrenados con MNRL sobre el **split de test**.  
Métricas: Recall@K (K=1,5,10,20,50), MRR y Δseparabilidad.  
Requiere `dataset_split.parquet` (Etapa 3) — **no** `dataset.parquet`.

```bash
# Un run (por defecto evalúa best/ sobre split=test)
python scripts/evaluate_finetuned.py --checkpoint beto_mnrl_hpc_run_e

# Época específica en lugar de best/
python scripts/evaluate_finetuned.py --checkpoint beto_mnrl_hpc_run_e --epoch 15

# Varios runs en una pasada (genera tabla resumen al final)
python scripts/evaluate_finetuned.py \
    --checkpoint beto_mnrl_hpc_run_e beto_mnrl_hpc_run_f \
              roberta_bio_hpc_run_a roberta_bio_hpc_run_b

# Todos los checkpoints con best/ disponible
python scripts/evaluate_finetuned.py --all

# Evaluar sobre val en lugar de test (útil para debug)
python scripts/evaluate_finetuned.py --checkpoint beto_mnrl_hpc_run_e --split val
```

**Dataset por defecto:** `processed/tesis1/dataset_split.parquet`  
**Salida:** `outputs/evaluation/finetuned/finetuned_results_<run>.json`

### HPC — lanzar como job SLURM

```bash
# Un job por run (corren en paralelo si hay nodos libres)
sbatch run_evaluate_biencoder.sh beto_mnrl_hpc_run_e
sbatch run_evaluate_biencoder.sh beto_mnrl_hpc_run_f
sbatch run_evaluate_biencoder.sh roberta_bio_hpc_run_a
sbatch run_evaluate_biencoder.sh roberta_bio_hpc_run_b

# Monitorear
squeue -u est_posgrado_uziel.lujan
```

### Bajar resultados del cluster

```bash
rsync -avz labcimatexterno:~/Data/INER/outputs/evaluation/finetuned/ \
    ~/Data/INER/outputs/evaluation/finetuned/ --mkpath
```

---

## Etapa 5 — Entrenamiento Bi-Encoder (`run_train_biencoder.py`)

Entrena el Bi-Encoder con MNRL. Consume `dataset_split.parquet` (Etapa 3).

**Parámetros clave:**

| Flag | Default | Descripción |
|---|---|---|
| `--model` | BETO | Nombre del modelo en `models/pretrained/` |
| `--dataset` | `tesis1/dataset_split.parquet` | Ruta al parquet con columna `split` |
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
    --dataset ~/Data/INER/processed/tesis0_sin_tokens/dataset_split.parquet \
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

## Visualización — Curvas de pérdida (`plot_training_curves.py`)

Genera gráficas de train_loss y val_loss a partir de `training_history.json`.  
Los JSON deben estar disponibles localmente (bajarlos del cluster si es necesario).

```bash
# Bajar training_history.json del cluster (archivos pequeños)
for run in beto_mnrl_hpc_run_e beto_mnrl_hpc_run_f roberta_bio_hpc_run_a roberta_bio_hpc_run_b; do
    rsync -avz "labcimatexterno:~/Data/INER/models/checkpoints/${run}/training_history.json" \
        ~/Data/INER/models/checkpoints/${run}/ --mkpath
done

# Un run — subplots separados (perspectiva por defecto)
python scripts/plot_training_curves.py --checkpoint beto_mnrl_hpc_run_e

# Varios runs con figura de comparación val_loss
python scripts/plot_training_curves.py \
    --checkpoint beto_mnrl_hpc_run_e beto_mnrl_hpc_run_f \
              roberta_bio_hpc_run_a roberta_bio_hpc_run_b \
    --compare

# Twin axes: train y val en el mismo eje con escalas independientes
# Guarda con sufijo _twin — no sobreescribe las figuras existentes
python scripts/plot_training_curves.py \
    --checkpoint beto_mnrl_hpc_run_e beto_mnrl_hpc_run_f \
              roberta_bio_hpc_run_a roberta_bio_hpc_run_b \
    --compare --twin-axes

# Todos los runs con history disponible
python scripts/plot_training_curves.py --all
```

**Salida:** `outputs/figures/training_curves_<run>.png` (subplots) /  
`outputs/figures/training_curves_<run>_twin.png` (twin axes) /  
`outputs/figures/training_curves_comparison.png`

> El script nunca sobreescribe figuras existentes — omite con aviso si el archivo ya existe.

---

## Flujo completo (tesis con v2 — recomendado)

```bash
# 1. Descargar modelos (una sola vez)
python scripts/download_model.py --all

# 2. Preprocesar
python scripts/run_preprocessing.py --perfil tesis1

# 3. Clasificar pares y producir xlsx editable
python scripts/run_dataset_v2.py --step classify --perfil tesis1

# 4. [Revisar pairs_for_review.xlsx — marcar 'decision' en los 'no_confirmado']

# 5. Producir dataset_v2.parquet con decisiones aplicadas
python scripts/run_dataset_v2.py --step finalize --perfil tesis1

# 6. Particionar (ver nota abajo sobre integración con dataset_v2.parquet)
python scripts/run_splitting.py --perfil tesis1

# 7. Entrenar
python scripts/run_train_biencoder.py --model BETO --epochs 10 --batch-size 64 --n-aug 0
```

> **Integración con `run_splitting.py`:** splitting acepta `--dataset dataset_v2.parquet` para usar el output etiquetado de Ruta A. La salida se deriva del input (`dataset_v2_split.parquet`), por lo que los splits v1 y v2 coexisten sin pisarse.

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
