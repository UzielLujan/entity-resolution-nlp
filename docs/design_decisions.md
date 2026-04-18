# Decisiones de diseño — `entity-resolution-nlp`

Este documento registra las decisiones arquitectónicas del proyecto de tesis y el razonamiento detrás de cada una.

---
## Estructura de carpetas

```text
entity-resolution-nlp/
├── .env                    ← rutas locales (nunca en git)
├── .env.example            ← plantilla de variables de entorno
├── .gitignore
├── pyproject.toml          ← definición del paquete y dependencias (uv)
│
├── docs/                   ← markdowns de contexto y decisiones de diseño
├── notebooks/              ← exploración y prototipado (no código de producción)
├── scripts/                ← entrenamiento, inferencia y preprocesamiento (HPC/local)
├── tests/                  ← unit tests
│
└── src/
    └── record_linkage/     ← paquete instalable principal
        ├── __init__.py
        ├── config.py       ← rutas centralizadas via pathlib + dotenv
        │
        ├── data/
        │   ├── __init__.py
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

Los datos **nunca** entran al repo. Viven en `~/Data/INER/` fuera de git.

```text
~/Projects/entity-resolution-nlp/   ← repo git (código, configs, docs)
~/Data/INER/
    ├── raw/           ← CSVs originales del INER (intocables)
    ├── processed/     ← outputs exploratorios del EDA
    ├── ground_truth/  ← pares candidatos sujetos a revisión manual
    ├── models/        ← checkpoints entrenados (.pt, .bin)
    └── embeddings/    ← vectores precalculados (.faiss)
```

`config.py` actúa como puente entre el repo y los datos externos mediante variables de entorno, con fallback automático a `~/Data/INER` si no existe `.env`.

---

## Flujo del dato

```text
raw CSVs (~/Data/INER/raw/)
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

## Decisiones tomadas y su justificación

### `src` layout con paquete `record_linkage/`

Usar `src/record_linkage/` en lugar de módulos sueltos en `src/` permite instalar el paquete en modo editable (`uv pip install -e .`), lo que hace que Python lo trate igual que cualquier dependencia instalada. Esto elimina el uso de `sys.path` en notebooks y garantiza imports consistentes desde cualquier contexto: notebook, script, test o cluster HPC.

El nombre `record_linkage` es el paquete Python; `entity-resolution-nlp` es el nombre del repo en GitHub. Son capas distintas — igual que `scikit-learn` (repo) vs `sklearn` (paquete).

### `dataset.py` — serialización y etiquetado consolidados

Inicialmente se consideraron dos módulos separados (`serialization.py` y `labeling.py`), pero se consolidaron en `dataset.py` por una razón crítica: `labeling.py` depende de que los `record_id` sean consistentes con los generados por `serialization.py`. Al tenerlos en el mismo módulo, los IDs se generan y consumen en el mismo contexto, eliminando el riesgo de desincronización silenciosa entre archivos intermedios.

Internamente `dataset.py` mantiene funciones separadas con responsabilidades claras:

```python
serialize_record(row) -> str         # tabular → secuencia con tokens
assign_entity_ids(df, pairs_df)      # asigna entity_id desde ground truth
build_dataset(csv_paths, ...)        # pipeline completo → .parquet
```

### `augmentation.py` — transformaciones on-the-fly

La aumentación de datos se aplica durante el entrenamiento y no se persiste en disco. Esto evita almacenar múltiples versiones alteradas de cada registro y garantiza variabilidad estocástica en cada época. Los operadores implementados son los descritos en la metodología: span deletion, block shuffling, typo injection, attribute masking e input swapping.

### Separación `mnrl.py` / `bce.py`

Las dos etapas del pipeline tienen objetivos de entrenamiento distintos:

- **Etapa 1 (Bi-Encoder):** optimizada con MNRL para construir un espacio métrico adecuado para búsqueda vectorial eficiente (alto recall).
- **Etapa 2 (Cross-Encoder):** optimizada con BCE como clasificador binario de alta precisión sobre los candidatos recuperados.

Mantenerlos en módulos separados refleja esta separación conceptual y facilita iterar sobre cada etapa de forma independiente.

---

## Pendientes de diseño

- Definir si `scripts/` tendrá un script por etapa o un pipeline unificado.
- Decidir la estrategia de logging y experiment tracking (MLflow, W&B, etc.)
- Confirmar compatibilidad arquitectónica con el cluster HPC de CIMAT (SLURM).
