# Decisiones de diseño — `entity-resolution-nlp`

Este documento registra las decisiones arquitectónicas del proyecto de tesis
y el razonamiento detrás de cada una.

La nota personal sobre entorno y dependencias modernas se encuentra en [[entorno_y_dependencias.md]] y complementa la justificación de `pyproject.toml`, `micromamba` y `uv`.

---

## Estructura de carpetas

```text
entity-resolution-nlp/
├── .env                    ← rutas locales (nunca en git)
├── .env.example            ← plantilla de variables de entorno
├── .gitignore
├── pyproject.toml          ← definición del paquete y dependencias (uv)
│
├── train_biencoder.sh      ← script de lanzamiento SLURM — Etapa 1
├── train_crossencoder.sh   ← script de lanzamiento SLURM — Etapa 2
│   (los .sh DEBEN vivir en la raíz del repo; requerimiento del Lab-SB
│    para que --chdir funcione correctamente y las rutas no se rompan)
│
├── logs/                   ← outputs de SLURM (en .gitignore, solo carpeta vacía en git)
├── docs/                   ← markdowns de contexto y decisiones de diseño
├── notebooks/              ← exploración y prototipado (no código de producción)
├── tests/                  ← unit tests
│
├── scripts/                ← scripts Python invocados desde .sh o local
│   ├── download_model.py      ← descarga BETO y RoBERTa-biomedical como SentenceTransformer
│   │                            Uso: python scripts/download_model.py --all
│   ├── run_preprocessing.py
│   ├── run_dataset.py         ← soporta --perfil B1|B2|zero_shot
│   ├── evaluate_zeroshot.py   ← Recall@K, MRR y Δ separabilidad sobre pares cross-database
│   │                            Uso: python scripts/evaluate_zeroshot.py --all
│   ├── train_biencoder.py
│   └── train_crossencoder.py
│
└── src/
    └── record_linkage/     ← paquete instalable principal
        ├── __init__.py
        ├── config.py       ← rutas centralizadas via pathlib + dotenv
        │
        ├── data/
        │   ├── __init__.py
        │   ├── preprocessing.py ← limpieza de CSVs crudos (Perfiles A y B)
        │   ├── dataset.py      ← serialización + etiquetado + guardado .parquet
        │   ├── augmentation.py ← transformaciones on-the-fly durante entrenamiento
        │   └── splitting.py    ← partición train/val/test a nivel de entidad
        │
        ├── models/
        │   ├── __init__.py
        │   ├── biencoder.py    ← Bi-Encoder siamés (SBERT) — Etapa 1
        │   └── crossencoder.py ← Cross-Encoder (DITTO) — Etapa 2
        │
        ├── training/
        │   ├── __init__.py
        │   ├── mnrl.py         ← Multiple Negatives Ranking Loss (Etapa 1)
        │   └── bce.py          ← Binary Cross-Entropy Loss (Etapa 2)
        │
        └── inference/
            ├── __init__.py
            ├── retrieval.py    ← indexación vectorial y búsqueda de candidatos
            └── reranking.py    ← clasificación fina y decisión final
```

---

## Separación repo / datos

Los datos **nunca** entran al repo. Viven en `~/Data/INER/` fuera de git,
tanto en local como en el cluster (con rutas distintas resueltas vía `.env`).

```text
~/Projects/entity-resolution-nlp/   ← repo git (código, configs, docs)
~/Data/INER/
    ├── raw/                ← CSVs originales del INER
    ├── processed/          ← CSVs limpios y dataset.parquet
    ├── ground_truth/       ← pares candidatos sujetos a revisión manual
    ├── models/
    │   ├── pretrained/     ← pesos descargados de HuggingFace como SentenceTransformer
    │   │   ├── BETO/               (dccuchile/bert-base-spanish-wwm-cased)
    │   │   └── RoBERTa-biomedical/ (PlanTL-GOB-ES/roberta-base-biomedical-clinical-es)
    │   │                     se descargan local y se transfieren al cluster
    │   └── checkpoints/    ← checkpoints entrenados por nosotros (.pt, .bin)
    ├── embeddings/         ← vectores precalculados (.faiss)
    └── outputs/
        ├── figures/     ← todas las visualizaciones (EDA, curvas de aprendizaje, etc.)
        ├── training/    ← métricas numéricas del entrenamiento (.json, .csv)
        └── evaluation/  ← reportes de clasificación (.txt, .csv, .json)
```

`config.py` actúa como puente entre el repo y los datos externos mediante
variables de entorno, con fallback automático a `~/Data/INER` si no existe `.env`.

---

## Flujo del dato

```text
raw CSVs (~/Data/INER/raw/)
   │
   ↓  data/preprocessing.py  — una vez, offline
   │  · Perfil A:  M0→M1→M2→M3→M4→M5→M6→M7 (base analítica INER)
   │  · Perfil B1: M1→M4(si TS) (mínima intervención para tesis)
   │  · Perfil B2: M1→M2→M4(si TS)→M7 (limpieza + renombrado semántico)
   │  · salida: ~/Data/INER/processed/<csv>_clean.csv
   │
   ↓  data/dataset.py  — una vez, offline
   │  · serializa cada registro tabular a secuencia de texto
   │  · use_block_tokens=True  → tokens [BLK_*] (fine-tuning)
   │  · use_block_tokens=False → Clave:Valor, nulos omitidos (zero-shot)
   │  · asigna entity_id usando llave determinista (expediente, nombre_norm)
   │  · guarda en ~/Data/INER/processed/<perfil>/dataset.parquet
   │
   ↓  scripts/evaluate_zeroshot.py  — una vez, antes del fine-tuning
   │  · codifica registros vinculables con BETO y RoBERTa-biomedical (zero-shot)
   │  · calcula Recall@K, MRR y Δ separabilidad sobre pares cross-database
   │  · establece el baseline a superar con fine-tuning
   │  · resultados en ~/Data/INER/outputs/evaluation/zeroshot_results.json
   │
   ↓  data/splitting.py  — una vez, offline
   │  · particiona a nivel de entidad (sin data leakage)
   │  · produce splits train / val / test
   │
   ↓  data/augmentation.py  — on-the-fly durante entrenamiento
   │  · span deletion, block shuffling, typo injection,
   │    attribute masking, input swapping
   │
   ↓  models/ + training/
      · Etapa 1: biencoder.py entrenado con mnrl.py
      · Etapa 2: crossencoder.py entrenado con bce.py
```

---

## Flujo local → cluster (Lab-SB CIMAT)

El cluster Lab-SB no tiene acceso a internet en los nodos de cómputo.
Todo modelo, tokenizer y dato debe estar disponible localmente en el cluster
antes de lanzar cualquier job.

```text
1. DESCARGAR MODELO (local, con internet)
   python scripts/download_model.py \
       --model_name "dccuchile/bert-base-spanish-wwm-cased" \
       --output_dir "~/Data/INER/models/pretrained/BETO"

2. TRANSFERIR DATOS Y MODELO AL CLUSTER (local → cluster)
   rsync -avz ~/Data/INER/processed/   labcimatexterno:~/Data/INER/processed/
   rsync -avz ~/Data/INER/models/      labcimatexterno:~/Data/INER/models/
   # Archivos pequeños (código): Dolphin vía SFTP

3. LANZAR JOB EN EL CLUSTER
   cd ~/entity-resolution-nlp
   sbatch train_biencoder.sh "~/Data/INER/models/pretrained/BETO" "run_01" 5

4. MONITOREAR
   squeue -u est_posgrado_uziel.lujan    # ver cola (R=Running, PD=Pending)
   tail -f logs/train_biencoder-JOB_ID.log

5. BAJAR RESULTADOS (cluster → local)
   rsync -avz labcimatexterno:~/Data/INER/outputs/     ~/Data/INER/outputs/
   rsync -avz labcimatexterno:~/Data/INER/models/checkpoints/  ~/Data/INER/models/checkpoints/
```

**Notas críticas del Lab-SB:**
- Los `.sh` deben vivir en la raíz del repo — `--chdir` en SLURM los requiere ahí.
- Usar `torchrun --nproc_per_node=2` para aprovechar las 2 GPUs Titan RTX por nodo.
- El entorno en el cluster usa Anaconda: `conda run -n env python script.py`.
- Nunca imprimir a pantalla en ejecuciones — todo debe ir a archivos de log.
- No existen respaldos externos — la integridad de pesos y datasets es responsabilidad del usuario.
- Cuota de maestría: máximo 4 nodos GPU simultáneos (`est_posgrado_uziel.lujan`).

---

## Decisiones tomadas y su justificación

### `src` layout con paquete `record_linkage/`

Usar `src/record_linkage/` en lugar de módulos sueltos en `src/` permite
instalar el paquete en modo editable (`uv pip install -e .`), lo que hace
que Python lo trate igual que cualquier dependencia instalada. Esto elimina
el uso de `sys.path` en notebooks y garantiza imports consistentes desde
cualquier contexto: notebook, script, test o cluster HPC.

El nombre `record_linkage` es el paquete Python; `entity-resolution-nlp`
es el nombre del repo en GitHub. Son capas distintas — igual que `scikit-learn`
(repo) vs `sklearn` (paquete).

### `preprocessing.py` — limpieza modular con perfiles A, B1 y B2

El pipeline de limpieza se implementa como un catálogo de funciones independientes
y componibles (M0–M7), cada una respondiendo a un hallazgo concreto del EDA.
Se exponen tres perfiles de ejecución:

- **Perfil A** (`run_profile_a`): limpieza completa orientada al entregable del
  INER (M0→M1→M2→M3→M4→M5→M6→M7).
- **Perfil B1** (`run_profile_b1`): mínima intervención para la serialización de la
  tesis (M1→M4 si TS). Columnas originales del CSV crudo, compatible con
  `SEMANTIC_BLOCKS` en `dataset.py`.
- **Perfil B2** (`run_profile_b2`): limpieza de caracteres + renombrado semántico
  (M1→M2→M4 si TS→M7). `SEMANTIC_BLOCKS` para B2 pendiente de definición.

**Decisión de nomenclatura — M7 al final:**
El módulo de renombrado de columnas (antes M4a) fue reubicado como M7 y declarado
siempre opcional y siempre al final. La razón: cualquier módulo que referencie
columnas por nombre asume los nombres originales del CSV crudo. Colocar el renombrado
en medio del pipeline creaba dependencias de orden frágiles (M4b buscaba columnas
que M4a ya había renombrado). Con M7 al final, todos los módulos anteriores son
agnósticos al renombrado.

### `dataset.py` — serialización y etiquetado consolidados

Inicialmente se consideraron dos módulos separados (`serialization.py` y
`labeling.py`), pero se consolidaron en `dataset.py` por una razón crítica:
`labeling.py` depende de que los `record_id` sean consistentes con los generados
por `serialization.py`. Al tenerlos en el mismo módulo, los IDs se generan y
consumen en el mismo contexto, eliminando el riesgo de desincronización silenciosa.

Internamente `dataset.py` mantiene funciones separadas con responsabilidades claras:

```python
serialize_record(row) -> str         # tabular → secuencia con tokens
assign_entity_ids(df, pairs_df)      # asigna entity_id desde ground truth
build_dataset(csv_paths, ...)        # pipeline completo → .parquet
```

### Construcción del Ground Truth y Asignación de `entity_id`

**✅ IMPLEMENTADO** en `src/record_linkage/data/dataset.py` — función `assign_entity_ids()`.

**Estrategia:**

La construcción del ground truth se basa en un criterio determinista:
**(expediente, nombre_v2_normalizado)** — sin `source_db`, para que el mismo paciente
en distintas bases de datos reciba el mismo `entity_id` y genere pares positivos
cross-database durante el entrenamiento MNRL.

**Normalización robusta (`normalizar_nombre_v2()`):**
Basada en análisis de `notebooks/Duplicados_INER.ipynb` §4.3:
1. Reemplaza `?` → `N` (encoding roto de Ñ en Comorbilidad)
2. Desacentúa vía NFD (quita diacríticos)
3. Limpia caracteres no alfabéticos (`.`, `|`, `/`, NBSP)
4. **Ordena tokens alfabéticamente** (invariante al orden AP/AM/NOMBRE entre CSVs)

**Pipeline de clustering:**
```python
def assign_entity_ids(df):
    # Normaliza nombres con normalizar_nombre_v2()
    # Crea tupla (expediente_int, nombre_norm) — sin source_db
    # Agrupa registros con tupla idéntica → mismo entity_id
    # Retorna df con columna entity_id (int64)
```

**Resultado Ground Truth:**
- **9,855 pares positivos confirmados** (EXP + nombre coinciden entre CSVs)
- **4,341 entidades vinculables** (presentes en 2 o 3 CSVs)
- **1,569 pares pendientes** (EXP igual, nombre diferente tras normalización)
  — candidatos para revisión manual o Zero-Shot SBERT

El DataFrame final (Parquet) tiene columnas `record_id`, `source_db`, `text`, `entity_id`.

**Notas arquitectónicas:**
- Evita pre-computar y almacenar pares explícitos ($O(N^2)$).
- Durante entrenamiento MNRL: cualquier batch con registros que comparten `entity_id`
  genera automáticamente pares positivos (in-batch negatives).
- Registros con distinto `entity_id` son negativos implícitos.
- Agnóstico al nivel de preprocesamiento: verificado con B0 (crudos) y B1 — ambos
  producen exactamente los mismos 9,855 pares y 4,341 vinculables.
- **Fuente de validación:** `notebooks/Duplicados_INER.ipynb` secciones 4.3–5.5.

**Reconciliación de métricas entre notebook y pipeline (2026-04-24):**

Durante la revisión se detectó una discrepancia aparente en el conteo de entidades únicas
entre el notebook de consultoría y el pipeline de tesis. La inspección reveló que miden
cosas distintas, ambas correctas para su propósito:

| Métrica | Valor | Definición | Propósito |
|---|---|---|---|
| Expedientes únicos (notebook) | 15,221 | `len(exp_eco ∪ exp_comor ∪ exp_ts)` — solo EXP, sin NaN | Reporte INER |
| Identidades únicas con EXP (notebook corregido) | 16,141 | `len(pares_eco ∪ pares_comor ∪ pares_ts)` — (EXP, nombre_norm), sin NaN | Análisis preciso |
| Identidades únicas (pipeline) | 16,222 | Igual que anterior + 81 registros con NaN expediente en Econo | Tesis — todos los registros |

El notebook original tenía un bug en la fila TOTAL de la tabla de distribución por regiones:
copiaba `total_entidades` (expedientes únicos) en la columna "EXP + Nombre" en lugar de
calcular la unión real de pares. Corregido en `notebooks/Duplicados_INER.ipynb` §5.3.

Los 81 registros con NaN expediente en Econo se excluyen del análisis de vinculación del
notebook (no pueden hacer match sin expediente) pero sí se incluyen en el pipeline de
tesis — cada uno recibe su propio `entity_id` y será visible al modelo durante inferencia.

### `augmentation.py` — transformaciones on-the-fly

La aumentación de datos se aplica durante el entrenamiento y no se persiste en
disco. Esto evita almacenar múltiples versiones alteradas de cada registro y
garantiza variabilidad estocástica en cada época. Los operadores implementados
son: span deletion, block shuffling, typo injection, attribute masking e
input swapping.


### Selección de backbone para zero-shot y fine-tuning

**Decisión (2026-04-28):** El segundo backbone es `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
en lugar de `PlanTL-GOB-ES/roberta-base-bne`.

**Motivo:** `roberta-base-bne` fue removido de HuggingFace Hub — el repositorio solo contiene
README sin pesos. El modelo biomédico-clínico es superior para este dominio:

| Modelo | Corpus | Relevancia para INER |
|---|---|---|
| `dccuchile/bert-base-spanish-wwm-cased` (BETO) | Wikipedia + noticias ES | Baseline estándar |
| `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | Texto clínico-biomédico ES | Alta — mismo dominio |

Ambos son RoBERTa/BERT base (12 capas, 768 dim), por lo que la comparación es justa.
Se guarda localmente como `~/Data/INER/models/pretrained/RoBERTa-biomedical/`.

---

### Separación `mnrl.py` / `bce.py`

Las dos etapas del pipeline tienen objetivos de entrenamiento distintos:

- **Etapa 1 (Bi-Encoder):** optimizada con MNRL para construir un espacio métrico
  adecuado para búsqueda vectorial eficiente (alto recall).
- **Etapa 2 (Cross-Encoder):** optimizada con BCE como clasificador binario de alta
  precisión sobre los candidatos recuperados.

### `.sh` en la raíz del repo

Los scripts de lanzamiento SLURM deben vivir en la raíz del proyecto en el
cluster. La directiva `--chdir` de SLURM fija ese punto como directorio de
trabajo, y todas las rutas relativas dentro del `.sh` (incluyendo `logs/`,
`scripts/`, `src/`) se resuelven desde ahí. Colocarlos en una subcarpeta
como `scripts/` rompe esta resolución sin configuración adicional no probada.

---

## Pendientes de diseño

- **Revisar en profundidad los modelos descargados (BETO y RoBERTa-biomedical).**
  En particular RoBERTa-biomedical: su `args.json` revela que fue entrenado con
  vocabulario BPE de 52k tokens (`bio-biomedical-clinical-vocab-52k`) construido
  sobre un corpus clínico-biomédico en español. Pendiente confirmar:
  - Fuentes exactas del corpus de preentrenamiento
  - Si incluye texto mexicano o solo español peninsular
  - Cuántas épocas / tokens procesados durante el preentrenamiento
  - Comparativa con BETO en benchmarks de NLP clínico en español (BioASQ-es, etc.)
- Definir si el entrenamiento en el cluster usará 1 o 2 GPUs por nodo
  (DDP con `torchrun --nproc_per_node=2` vs entrenamiento estándar).
- Decidir la estrategia de logging y experiment tracking (MLflow, W&B, etc.)
- Crear el entorno de Conda en el cluster y verificar compatibilidad con
  las versiones de CUDA disponibles en los nodos g-0-X.
