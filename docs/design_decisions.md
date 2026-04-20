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
│   ├── download_model.py   ← descarga modelos HuggingFace localmente antes de subir al cluster
│   ├── run_preprocessing.py
│   ├── run_dataset.py
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
    │   ├── pretrained/     ← pesos descargados de HuggingFace (BETO, RoBERTa-bne)
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
   │  · Perfil A: M1→M2→M3→M4a→M4b→M4c→M5 (base analítica INER)
   │  · Perfil B: M1→M2→M4a→M4b (mínima intervención para serialización)
   │  · salida: ~/Data/INER/processed/<csv>_clean.csv
   │
   ↓  data/dataset.py  — una vez, offline
   │  · serializa cada registro tabular a secuencia de texto con tokens especiales
   │  · asigna entity_id usando los pares del ground_truth/
   │  · guarda el resultado en ~/Data/INER/processed/dataset.parquet
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

### `augmentation.py` — transformaciones on-the-fly

La aumentación de datos se aplica durante el entrenamiento y no se persiste en
disco. Esto evita almacenar múltiples versiones alteradas de cada registro y
garantiza variabilidad estocástica en cada época. Los operadores implementados
son: span deletion, block shuffling, typo injection, attribute masking e
input swapping.

### `preprocessing.py` — limpieza modular con perfiles A y B

El pipeline de limpieza se implementa como un catálogo de funciones independientes
y componibles (M1–M8), cada una respondiendo a un hallazgo concreto del EDA.
Se exponen dos perfiles de ejecución:

- **Perfil A** (`run_profile_a`): limpieza completa orientada al entregable del
  INER (M1→M2→M3→M4a→M4b→M4c→M5).
- **Perfil B** (`run_profile_b`): mínima intervención para la serialización de la
  tesis (M1→M2→M4a→M4b). Preserva el ruido léxico deliberadamente para que el
  modelo aprenda a superarlo sin sesgo de limpieza.

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

- Definir si el entrenamiento en el cluster usará 1 o 2 GPUs por nodo
  (DDP con `torchrun --nproc_per_node=2` vs entrenamiento estándar).
- Decidir la estrategia de logging y experiment tracking (MLflow, W&B, etc.)
- Crear el entorno de Conda en el cluster y verificar compatibilidad con
  las versiones de CUDA disponibles en los nodos g-0-X.
