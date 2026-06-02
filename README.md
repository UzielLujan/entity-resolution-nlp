# entity-resolution-nlp

Sistema de ligado de registros (Record Linkage) basado en aprendizaje profundo y NLP, aplicado a bases de datos de pacientes COVID-19 del INER. Proyecto de tesis de maestría en Cómputo Estadístico — CIMAT Unidad Monterrey.

---

## Contexto

El INER cuenta con tres bases de datos independientes de pacientes COVID-19 que no comparten una llave de identificación 100% confiable. Este proyecto construye un sistema moderno para vincularlas a nivel semántico usando modelos de lenguaje pre-entrenados, sin depender de coincidencias exactas de campos. La arquitectura es un pipeline híbrido **Retrieve & Rerank**: Bi-Encoder (SBERT + MNRL) en Etapa 1 y Cross-Encoder (DITTO + BCE) en Etapa 2.

| CSV | Registros | Contenido |
|-----|-----------|-----------|
| Diagnósticos y Comorbilidades | 4,278 | diagnóstico principal, comorbilidades, fechas |
| Costos y Económico | 4,632 | costos de atención, datos socioeconómicos |
| Trabajo Social | 14,796 | datos demográficos, familia, situación social |

Ground truth final del dataset v2 (tras revisión manual de los pares):
**15,283 entidades únicas** y **11,447 pares cross-DB confirmados** = 9,855 por llave exacta + 1,118 por métrica clásica + 493 hard positives + 21 hard negatives marcados manualmente.

---

## Estructura del repositorio

```
entity-resolution-nlp/
├── train_biencoder_beto.sh        # SLURM — Bi-Encoder BETO (MNRL)
├── train_biencoder_roberta.sh     # SLURM — Bi-Encoder RoBERTa-biomedical (MNRL)
├── train_crossencoder_beto.sh     # SLURM — Cross-Encoder BETO (BCE)
├── eval_biencoder.sh              # SLURM — evaluación BE (ranking + espacio métrico)
├── eval_crossencoder.sh           # SLURM — evaluación CE (binaria + threshold)
├── calibrate_crossencoder.sh      # SLURM — calibración por temperatura + incertidumbre
│
├── src/record_linkage/
│   ├── config.py                  # Rutas centralizadas (DATA_ROOT, PROCESSED_DIR, perfiles)
│   ├── data/
│   │   ├── preprocessing.py       # Módulos M0–M7, perfiles iner/tesis0/tesis1/tesis2
│   │   ├── dataset.py             # Dataset v1 (legacy, baseline)
│   │   ├── dataset_v2.py          # Dataset v2: classify → finalize con xlsx de revisión humana
│   │   ├── serialization.py       # Reglas de serialización (tokens [BLK_*], col:val, skipnull)
│   │   ├── augmentation.py        # 5 operadores on-the-fly: shuffle, mask, typos, delete
│   │   ├── splitting.py           # Partición train/val/test estratificada por entidad
│   │   ├── comparison_methods.py  # Registro extensible de métodos de score (JW, Lev, …)
│   │   └── consolidation.py       # Generador JSON entity-centric INER (Perfil iner)
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
│   ├── utils/
│   │   ├── normalization.py       # Normalización de nombres y campos categóricos
│   │   ├── entities.py            # Asignación de entity_id (Union-Find)
│   │   └── pairs.py               # classify_pairs(): llave exacta + métrica clásica + límites
│   └── inference/
│       ├── retrieval.py           # Búsqueda vectorial ANN (pendiente — blocking BE+ANN)
│       └── reranking.py           # Re-ranking Cross-Encoder en producción (pendiente)
│
├── scripts/
│   ├── download_model.py            # Descarga modelos HF como SentenceTransformer
│   ├── run_preprocessing.py         # CLI: limpieza de CSVs crudos por perfil
│   ├── run_dataset.py               # CLI: dataset v1 (legacy)
│   ├── run_dataset_v2.py            # CLI: dataset v2 (classify → revisión humana → finalize)
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
│   ├── merge_review_decisions.py    # Preserva decisiones humanas al regenerar el xlsx
│   ├── show_pair.py                 # Inspección on-demand de un par desde *_clean.csv
│   ├── report_linking_numbers.py    # Reporte de cifras de vinculación para el manuscrito
│   ├── sanity_check_paraphrase.py   # Smoke test del baseline paraphrase-multilingual
│   ├── build_consolidated_json.py   # Genera consolidated_entities_v2.json (entregable INER)
│   └── build_data_dictionary.py     # Genera Diccionario_Final_INER.csv (entregable INER)
│
├── notebooks/                       # EDAs, análisis de duplicados, diccionarios hardcodeados
└── docs/                            # Documentación del proyecto
```

---

## Inicio rápido

### 1. Entorno

```bash
micromamba create -n tesis python=3.11 -c conda-forge -y
micromamba activate tesis
pip install uv
uv pip install -e .
```

El flag `-e` instala `record_linkage` en modo editable — los cambios en `src/` se reflejan sin reinstalar. Dependencias de desarrollo (jupyter, pytest, ruff) se instalan con `uv pip install -e ".[dev]"`.

### 2. Descargar modelos

Los modelos se descargan localmente como SentenceTransformer con tokens especiales ya registrados, para poder transferirlos al cluster de cómputo sin acceso a internet.

```bash
python scripts/download_model.py --all
```

Modelos disponibles: `BETO`, `RoBERTa-biomedical`, `paraphrase-multilingual`.

### 3. Preprocesamiento

```bash
# Verificar rutas
python scripts/run_preprocessing.py --perfil tesis1 --check-paths

# Limpiar CSVs
python scripts/run_preprocessing.py --perfil tesis1
```

Perfiles disponibles:

| Perfil | Módulos | Uso |
|--------|---------|-----|
| `iner` | M0(upper)→M1→M2→M3→M4→M5→M6→M7 | Entregables de consultoría INER |
| `tesis0` | M0(strip)→M1 | Base para evaluación zero-shot |
| `tesis1` | M0(strip)→M1→M4(si TS)→M5 | Entrenamiento — columnas originales del CSV |
| `tesis2` | M0(strip)→M1→M2→M4(si TS)→M5→M7 | Limpieza + renombrado semántico |

### 4. Construir dataset (v2 — flujo oficial de tesis)

```bash
# Paso 1: clasificar pares y emitir el xlsx de revisión
python scripts/run_dataset_v2.py --perfil tesis1 --step classify

# Paso 2: editar pairs_for_review.xlsx (decisiones match / no_match)

# Paso 3: finalizar — emite dataset_v2.parquet integrando las decisiones humanas
python scripts/run_dataset_v2.py --perfil tesis1 --step finalize
```

Salida: `dataset_v2.parquet` con columnas `record_id`, `source_db`, `text`, `entity_id`. La versión legacy `run_dataset.py` (dataset v1, sin revisión humana) sigue disponible pero no es la oficial.

### 5. Partición train/val/test

```bash
python scripts/run_splitting.py --perfil tesis1
```

Estratificada por entidad (sin entity leakage). Proporciones por defecto: 70/15/15.

### 6. Evaluación zero-shot (baseline)

```bash
python scripts/evaluate_zeroshot.py --model BETO
python scripts/evaluate_zeroshot.py --all
```

Métricas: Hit@K, Recall@K, RecallNorm@K, Precision@K (K∈{1,5,10,20,50}), MRR y Δsep.

### 7. Entrenamiento del Bi-Encoder (local / HPC)

```bash
# Smoke test local (1 época, batch pequeño)
python scripts/run_train_biencoder.py --model BETO --epochs 1 --batch-size 8 --n-aug 0

# HPC — editar train_biencoder_beto.sh con los parámetros y lanzar
sbatch train_biencoder_beto.sh
```

### 8. Evaluación del Bi-Encoder fine-tuneado

```bash
python scripts/evaluate_finetuned.py --run beto_mnrl_hpc_run_e
sbatch eval_biencoder.sh
```

### 9. Hard Negative Mining

```bash
python scripts/mine_hard_pairs.py --run beto_mnrl_hpc_run_e --top-k 20
```

Emite `pairs_hard.parquet` con triples (positivos + hard_positives + hard_negatives) por split, listos para entrenar el Cross-Encoder.

### 10. Entrenamiento y evaluación del Cross-Encoder

```bash
sbatch train_crossencoder_beto.sh
sbatch eval_crossencoder.sh
```

Calcula además el **threshold óptimo** sobre val (F1-max) y reporta F1, Precision, Recall, PR-AUC y matriz de confusión sobre test.

### 11. Calibración + incertidumbre (Vía A)

```bash
sbatch calibrate_crossencoder.sh
```

Aplica temperature scaling sobre los logits del CE en val y reporta ECE, NLL, T óptima, y la entropía por vínculo de los pares de test para identificar candidatos a revisión humana.

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
| `MEMORY.md` | Bitácora del proyecto: historial de fases, métricas, preguntas abiertas |
| `docs/design_decisions.md` | Decisiones técnicas tomadas y su justificación |
| `docs/Metodologia_arquitectura.md` | Arquitectura neuronal: SBERT, DITTO, MNRL, serialización |
| `docs/Contexto_Maestro_Proyecto.md` | Visión general, roadmap, rutas de artefactos externos |
| `docs/Contexto_Consultoria_INER.md` | Objetivos y entregables de la consultoría INER |
| `docs/entorno_y_dependencias.md` | Entorno Python: micromamba, uv, pyproject.toml |
| `docs/comandos_proyecto.md` | Comandos canónicos del pipeline end-to-end |
| `docs/plan_ruta_a_etiquetado.md` | Ruta A — rediseño del etiquetado y dataset v2 |
| `docs/plan_ruta_b.md` | Ruta B — pipeline BE + Cross-Encoder, experimento 2×2 |
| `docs/propuesta_entregable_JSON.md` | Schema entity-centric del JSON consolidado INER |
| `docs/Anexos/metricas_evaluacion.md` | Métricas (Recall@K, PR-AUC, Δsep) — teoría y operacionalización |
| `docs/Anexos/propuesta_incertidumbre.md` | Vía A/B de incertidumbre + auditoría de etiquetas |
| `docs/Anexos/metricspace-insight.md` | Lectura estadística del espacio métrico (X_pos/X_neg, Δsep) |
| `docs/Anexos/demo-propuesta.md` | Demo Streamlit + RAG (LLM auditor, human-in-the-loop) |

---

## Estado actual

**Pipeline empírico completo. Foco actual: redacción del manuscrito.**

- [x] EDA de las tres bases (Comorbilidad, Económico, Trabajo Social)
- [x] Ground truth — 15,283 entidades únicas, 11,447 pares cross-DB confirmados (dataset v2)
- [x] Pipeline de preprocesamiento (perfiles iner, tesis0, tesis1, tesis2)
- [x] Serialización con bloques semánticos → `dataset_v2.parquet`
- [x] Revisión manual de pares limítrofes (493 hard positives + 21 hard negatives)
- [x] Partición train/val/test estratificada por entidad (sin entity leakage)
- [x] Augmentación on-the-fly (5 operadores: shuffle_blocks, shuffle_columns, mask, typos, delete_span)
- [x] Bi-Encoder — `build_biencoder()` con backbone intercambiable (BETO, RoBERTa-biomedical)
- [x] Pipeline de entrenamiento MNRL (warm init, LR diferencial por capa, fp16, early stopping)
- [x] Evaluación zero-shot y fine-tuneada (Hit@K, Recall@K, RecallNorm@K, Precision@K, MRR, Δsep)
- [x] Experimento de serialización 2×2 (tokens × nulos) — ganador `tok_skipnull` (val=1.1029, Δsep=11.28)
- [x] Modelo final Bi-Encoder — BETO `lr=2e-5`, `temp=0.07`, val_loss=1.0224
- [x] Hard Negative Mining sobre el BE fine-tuneado (`mine_hard_pairs.py`)
- [x] Cross-Encoder DITTO entrenado y evaluado (BCE + pos_weight=8) — **F1=1.0000** sobre test (16,350 pares)
- [x] Calibración del CE + incertidumbre por vínculo (Vía A — temperature scaling + entropía)
- [x] Visualización UMAP 3D del espacio métrico (`visualize_embeddings.py`)
- [x] Eje consultoría INER cerrado — `consolidated_entities_v2.json` + `Diccionario_Final_INER.csv`
- [x] Presentación de avance (mayo 2026) — Beamer XeLaTeX, 27 slides
- [ ] Rediseño de augmentación — operadores actuales no simulan variación real entre CSVs del INER
- [ ] Blocking semántico BE+ANN (FAISS) — sustituye al filtro por expediente en producción
- [ ] Indexación vectorial e inferencia en producción (`inference/retrieval.py`, `reranking.py`)
- [ ] Demo interactiva Streamlit + RAG (LLM auditor, human-in-the-loop)
- [ ] Manuscrito de tesis, foco actual.
