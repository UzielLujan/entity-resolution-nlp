# entity-resolution-nlp

Sistema de ligado de registros (Record Linkage) basado en aprendizaje profundo y NLP, aplicado a bases de datos de pacientes COVID-19 del INER. Proyecto de tesis de maestría en Cómputo Estadístico — CIMAT Unidad Monterrey.

---

## Contexto

El INER cuenta con tres bases de datos independientes de pacientes COVID-19 que no comparten una llave de identificación 100% confiable. Este proyecto construye un sistema moderno para vincularlas a nivel semántico usando modelos de lenguaje pre-entrenados, sin depender de llaves de identificación ni coincidencias exactas de campos. La arquitectura es un pipeline híbrido **Retrieve & Rerank**: Bi-Encoder (SBERT + MNRL) en Etapa 1 y Cross-Encoder (DITTO + BCE) en Etapa 2.

| CSV | Registros | Contenido |
|-----|-----------|-----------|
| Diagnósticos y Comorbilidades | 4,278 | diagnóstico principal, comorbilidades, fechas |
| Costos y Económico | 4,632 | costos de atención, datos socioeconómicos |
| Trabajo Social | 14,796 | datos demográficos, familia, situación social |

El dataset preparado contiene **un total de 23,706 registros**, **15,283 entidades únicas** y
**4,605 entidades vinculables**. Estas entidades generan **11,447 pares positivos entre bases
de datos a nivel entidad**. Al expandir cada entidad a todas las combinaciones cross-db de sus
registros, el conjunto contiene **12,224 pares positivos entre bases de datos a nivel registro**,
utilizados por el pipeline en sus respectivos splits. Este ground truth es un *silver standard*
construido y revisado en el repositorio independiente `consultoria-iner`.

---

## Estructura del repositorio

```text
entity-resolution-nlp/
├── src/record_linkage/
│   ├── config.py                  # Contrato de rutas bajo INER_DATA_ROOT
│   ├── data/
│   │   ├── augmentation.py        # 5 operadores on-the-fly: shuffle, mask, typos, delete
│   │   ├── splitting.py           # Partición train/val/test estratificada por entidad
│   ├── models/
│   │   ├── biencoder.py           # build_biencoder() + encode_texts() — backbone intercambiable
│   │   └── crossencoder.py        # Cross-Encoder DITTO (CLS + linear head)
│   ├── training/
│   │   ├── mnrl.py                # Multiple Negatives Ranking Loss (SBERT)
│   │   ├── bce.py                 # Binary Cross-Entropy con pos_weight
│   │   ├── train_biencoder.py     # Entrenamiento BE: warm init, LR diferencial, fp16
│   │   └── train_crossencoder.py  # Entrenamiento CE: BCE + early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # Hit@K, Recall@K, MRR, Δsep, F1/PR-AUC/ROC-AUC
│   │   ├── biencoder_eval.py      # Pipeline de evaluación BE (ranking + espacio métrico)
│   │   ├── crossencoder_eval.py   # Pipeline de evaluación CE (binaria + threshold óptimo)
│   │   └── calibration.py         # Temperature scaling + entropía por vínculo (Vía A)
│   └── inference/
│       ├── retrieval.py           # Búsqueda vectorial ANN (pendiente — blocking BE+ANN)
│       └── reranking.py           # Re-ranking Cross-Encoder en producción (pendiente)
│
├── scripts/
│   ├── download_model.py            # Descarga modelos HF como SentenceTransformer
│   ├── run_splitting.py             # CLI: partición train/val/test
│   ├── run_train_biencoder.py       # CLI: entrenamiento BE con MNRL
│   ├── train_crossencoder.py        # CLI: entrenamiento CE con BCE
│   ├── evaluate_zeroshot.py         # Evaluación zero-shot del BE
│   ├── evaluate_finetuned.py        # Evaluación del BE fine-tuneado (4 métricas + Δsep)
│   ├── evaluate_crossencoder.py     # Evaluación del CE + threshold óptimo
│   ├── calibrate_crossencoder.py    # Temperature scaling + extracción de pares inciertos
│   ├── mine_hard_pairs.py           # Hard Negative Mining sobre BE fine-tuneado
│   ├── visualize_embeddings.py      # UMAP 3D interactivo (HTML Plotly) del espacio métrico
│   ├── plot_training_curves.py      # Curvas de loss desde training_history.json
│   ├── measure_token_distribution.py # Diagnóstico de longitud tokenizada por CSV
│   ├── export_embeddings.py         # Exportación canónica por record_id
│   └── sanity_check_paraphrase.py   # Smoke test del baseline paraphrase-multilingual
├── docs/                            # Contrato, metodología y guía operativa
├── notebooks/                       # Inspección de artefactos preparados
├── pyproject.toml                   # Dependencias base y extras opcionales
├── uv.lock                          # Resolución reproducible del entorno local
└── *.sh                             # Wrappers SLURM históricos; revisión pendiente
```

---

## Configuración de inicio

### 0. Variables de entorno
Primero, copie `.env.example` como `.env` y configure la variable `INER_DATA_ROOT` con una ruta absoluta a la raíz externa de artefactos. Esta raíz debe contener al menos una variante del dataset etiquetado en `processed/default/output/<variant>/dataset.parquet`.
```bash
cp .env.example .env
```

### 1. Entorno de python

Se recomienda usar `uv`:

```bash
uv sync
source .venv/bin/activate
python -m record_linkage.config
```

`uv sync` crea `.venv`, instala el paquete en modo editable y sincroniza las dependencias base desde `uv.lock`. Después de activar el entorno, los comandos se ejecutan con `python` normalmente.

Dependencias opcionales:

```bash
uv sync --extra notebook   # Jupyter e ipykernel
uv sync --extra dev        # pytest y Ruff
uv sync --all-extras       # Todos los extras
```

### 2. Descargar modelos

Los modelos se descargan localmente como SentenceTransformer con tokens especiales ya registrados, para poder transferirlos al cluster de cómputo sin acceso a internet.

```bash
python scripts/download_model.py --all
```

Modelos disponibles: `BETO`, `RoBERTa-biomedical`, `paraphrase-multilingual`.

### 3. Entrada y partición

Este repositorio no lee datos crudos, no preprocesa CSV y no construye el ground truth.
Consume una o más variantes preparadas con el esquema
`[record_id, source_db, text, entity_id]`:

```text
$INER_DATA_ROOT/processed/default/output/<variant>/dataset.parquet
```

Para generar la partición canónica:

```bash
python scripts/run_splitting.py --variant tok_skipnull
```

La salida vive en `modeling/data/<variant>/split.parquet`. El algoritmo asigna entidades
completas con semilla 42 y proporciones 70/15/15; valida que todos los registros reciban
split y que ningún `entity_id` aparezca en más de uno.

### 4. Hard Negative Mining

HNM requiere un checkpoint Bi-Encoder previamente entrenado, que no forma parte del
repositorio. Por defecto, el checkpoint `best/` de cada run se resuelve desde:

```text
modeling/models/biencoder/<variant>/<run>/best/
```

```bash
python scripts/mine_hard_pairs.py \
  --checkpoint beto_mnrl \
  --variant tok_skipnull \
  --dataset "$INER_DATA_ROOT/modeling/data/tok_skipnull/split.parquet" \
  --top-k 20
```

Produce `pairs_train.parquet`, `pairs_val.parquet` y `pairs_test.parquet` en
`modeling/data/<variant>/`, con esquema
`[record_id_a, record_id_b, label, similarity, type]`. La opción `--output-dir` permite
sobrescribir el directorio de salida para ejecuciones experimentales.

Los comandos de entrenamiento, evaluación, calibración y exportación están documentados
en [`docs/comandos_proyecto.md`](docs/comandos_proyecto.md).

---

## Serialización

Cada registro tabular se convierte en una secuencia de texto estructurada con bloques semánticos:

```
[BLK_ID] [COL] nombre [VAL] GARCIA LOPEZ MARIA [BLK_ADMIN] [COL] expediente [VAL] 12345 [BLK_CLIN] ...
```

Con `--no-special-tokens` (zero-shot):

```
nombre: GARCIA LOPEZ MARIA expediente: 12345 ...
```

Los tokens `[BLK_ID]`, `[BLK_CLIN]`, `[BLK_GEO]`, `[BLK_ADMIN]`, `[BLK_SOCIO]`, `[COL]` y `[VAL]` se registran en el tokenizador al descargar los modelos. La configuración oficial para entrenar el CE (y la ganadora del experimento 2×2) es **`tok_skipnull`**: tokens activos + nulos omitidos, lo que mantiene los pares dentro del presupuesto de 256 tokens por lado para BERT.

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| `docs/data_contract.md` | Contrato de entrada, salida y propiedad de artefactos |
| `docs/design_decisions.md` | Decisiones técnicas tomadas y su justificación |
| `docs/Metodologia_arquitectura.md` | Arquitectura neuronal: SBERT, DITTO, MNRL, serialización |
| `docs/comandos_proyecto.md` | Comandos operativos del pipeline neuronal |
| `docs/contexto_manuscrito_tesis.md` | Correspondencia entre repositorio y manuscrito |

---

## Estado actual

**Pipeline empírico completo; estabilización y auditoría en curso.**

- [x] Partición train/val/test estratificada por entidad (sin entity leakage)
- [x] Augmentación on-the-fly (5 operadores: shuffle_blocks, shuffle_columns, mask, typos, delete_span)
- [x] Bi-Encoder — `build_biencoder()` con backbone intercambiable (BETO, RoBERTa-biomedical)
- [x] Pipeline de entrenamiento MNRL (warm init, LR diferencial por capa, fp16, early stopping)
- [x] Evaluación zero-shot y fine-tuneada (Hit@K, Recall@K, RecallNorm@K, Precision@K, MRR, Δsep)
- [x] Experimento de serialización 2×2 (tokens × nulos) — ganador `tok_skipnull` (val=1.1029, Δsep=11.28)
- [x] Modelo final Bi-Encoder — BETO sobre `tok_skipnull`
- [x] Hard Negative Mining sobre el BE fine-tuneado (`mine_hard_pairs.py`)
- [x] Cross-Encoder DITTO entrenado y evaluado (BCE + pos_weight=8) — **F1=1.0000** sobre test (16,350 pares)
- [x] Calibración del CE + incertidumbre por vínculo (Vía A — temperature scaling + entropía)
- [x] Visualización UMAP 3D del espacio métrico (`visualize_embeddings.py`)
- [ ] Rediseño de augmentación — operadores actuales no simulan variación real entre CSVs del INER
- [ ] Blocking semántico BE+ANN (FAISS) — sustituye al filtro por expediente en producción
- [ ] Indexación vectorial e inferencia en producción (`inference/retrieval.py`, `reranking.py`)
- [ ] Demo interactiva Streamlit + RAG (LLM auditor, human-in-the-loop)
- [ ] Manuscrito de tesis, foco actual.
