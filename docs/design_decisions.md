# Decisiones de diseño — `entity-resolution-nlp`

Este documento registra las decisiones arquitectónicas del proyecto de tesisy el razonamiento detrás de cada una.

Las notas personales sobre aspectos técnicos y metodológicos sobre este proyecto se encuentran en `docs/Anexos/*.md`, por ejemplo, en [[entorno_y_dependencias.md]] complementa la justificación de `pyproject.toml`, `micromamba` y `uv`.

---

## Estructura de carpetas

```text
entity-resolution-nlp/
├── .env                    ← rutas locales (nunca en git)
├── .env.example            ← plantilla de variables de entorno
├── .gitignore
├── pyproject.toml          ← definición del paquete y dependencias (uv)
│
├── run_beto_baseline.sh    ← script de lanzamiento SLURM — Bi-Encoder BETO
├── run_roberta_bio.sh      ← script de lanzamiento SLURM — Bi-Encoder RoBERTa-biomedical
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
│   ├── run_dataset.py         ← soporta --perfil tesis0|tesis1|tesis2|iner; modo zero-shot con --no-special-tokens
│   ├── run_splitting.py       ← partición train/val/test por entidad
│   ├── evaluate_zeroshot.py   ← Hit@K, MRR y Δ separabilidad sobre pares cross-database
│   │                            Uso: python scripts/evaluate_zeroshot.py --all
│   ├── run_train_biencoder.py ← CLI delgado para Etapa 1 (importa training/train_biencoder.py)
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
        │   ├── train_biencoder.py ← Entrenamiento MNRL completo: warm init, LR diferencial, fp16
        │   └── bce.py             ← Binary Cross-Entropy Loss (Etapa 2)
        │
        ├── utils/
        │   ├── __init__.py
        │   └── mnrl.py    ← dump_mnrl_batch() + render_sim_matrix() — diagnóstico visual MNRL
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
   ↓ preprocessing.py — aplica perfil (iner/tesis0/tesis1/tesis2)
   │  salida: processed/<perfil>/<csv>_clean.csv
   ↓ dataset.py — serializa registros y asigna entity_ids
   │  use_block_tokens=True  → tokens [BLK_*]   (fine-tuning)
   │  use_block_tokens=False → col:val sin tokens (zero-shot)
   │  salida: processed/<perfil>/dataset.parquet
   ↓ evaluate_zeroshot.py — baseline pre-MNRL
   │  Recall@K, MRR, Δseparabilidad sobre pares cross-database
   ↓ splitting.py — partición train/val/test a nivel de entidad
   ↓ augmentation.py — on-the-fly durante entrenamiento
   ↓ models/ + training/
      Etapa 1: biencoder.py + mnrl.py
      Etapa 2: crossencoder.py + bce.py
```

---

## Flujo local → cluster (Lab-SB CIMAT)

El cluster Lab-SB no tiene acceso a internet en los nodos de cómputo.
Todo modelo, tokenizer y dato debe estar disponible localmente en el cluster
antes de lanzar cualquier job.

```text
1. DESCARGAR MODELO (local, con internet)
   python scripts/download_model.py --model dccuchile/bert-base-spanish-wwm-cased --name BETO
   python scripts/download_model.py --all   # descarga todos los modelos conocidos

2. TRANSFERIR DATOS Y MODELO AL CLUSTER (local → cluster)
   rsync -avz ~/Data/INER/processed/   labcimatexterno:~/Data/INER/processed/
   rsync -avz ~/Data/INER/models/      labcimatexterno:~/Data/INER/models/
   # Archivos pequeños (código): Dolphin vía SFTP

3. LANZAR JOB EN EL CLUSTER
   cd ~/entity-resolution-nlp
   sbatch run_beto_baseline.sh                          # BETO con defaults
   sbatch run_roberta_bio.sh 0.07 roberta_bio_run_b     # RoBERTa con temp y nombre custom

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
- El entorno en el cluster usa Anaconda. Invocar Python directamente: `~/.conda/envs/tesis/bin/python -u script.py` — `conda run` bufferiza stdout y no muestra logs en tiempo real.
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

### `preprocessing.py` — limpieza modular con perfiles iner, tesis0, tesis1 y tesis2

El pipeline de limpieza se implementa como un catálogo de funciones independientes
y componibles (M0–M7), cada una respondiendo a un hallazgo concreto del EDA.
Se exponen tres perfiles de ejecución:

- **Perfil iner** (`profile_iner`): limpieza completa orientada al entregable del
  INER (M0→M1→M2→M3→M4→M5→M6→M7).
- **Perfil tesis1** (`profile_tesis1`): mínima intervención para la serialización de la
  tesis (M0(strip)→M1→M4 si TS→M5). Columnas originales del CSV crudo, compatible con
  `SEMANTIC_BLOCKS` en `dataset.py`.
- **Perfil tesis2** (`profile_tesis2`): limpieza de caracteres + renombrado semántico
  (M1→M2→M4 si TS→M7). `SEMANTIC_BLOCKS` para tesis2 pendiente de definición.
- **Modo Zero-Shot** (usa `profile_tesis0`): M0(strip)→M1. Sin M4 —
  los campos de nombre de Trabajo Social (`APELLIDO PATERNO`, `APELLIDO MATERNO`,
  `NOMBRE`) permanecen separados con sus nombres originales. `build_dataset` los
  concatena on-the-fly para la asignación de `entity_id`. Produce CSVs en
  `processed/tesis0/`, desacoplados de tesis1.

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
serialize_record(row, csv_name)      # tabular → secuencia con tokens [BLK_*]/[COL]/[VAL] (B1/B2)
serialize_record_zeroshot(row)       # tabular → "col: val" sin tokens (ZS)
assign_entity_ids(df)                # asigna entity_id por llave (expediente, nombre_norm)
build_dataset(csv_paths, ...)        # pipeline completo → .parquet
```

**Serialización zero-shot separada (2026-04-29):**

`serialize_record_zeroshot` itera `row.index` directamente (sin `SEMANTIC_BLOCKS` ni
`SERIALIZATION_ORDER`) para preservar el orden natural de columnas del CSV. Campos
nulos se omiten. Se llama **antes del `pd.concat`** en `build_dataset`, mientras cada
DataFrame individual tiene su orden de columnas original — después del concat pandas
produce un orden mezclado de la unión de columnas de los 3 CSVs.

```python
# En build_dataset — ZS serializa por CSV antes de concatenar
for csv_path in csv_paths:
    df = pd.read_csv(csv_path)
    if not use_block_tokens:
        df["text"] = df.apply(serialize_record_zeroshot, axis=1)
    dfs.append(df)
```

### Construcción del Ground Truth y Asignación de `entity_id`

Implementado en `assign_entity_ids()` — `src/record_linkage/data/dataset.py`.

**Estrategia:** llave determinista `(expediente_int, nombre_v2_normalizado)` sin `source_db`,
para que el mismo paciente en distintas bases reciba el mismo `entity_id`.

**`normalizar_nombre_v2()`:** `?`→`N`, desacentuación NFD, limpia no-alfa, ordena tokens
alfabéticamente (invariante al orden AP/AM/NOMBRE entre CSVs). Ver `notebooks/Duplicados_INER.ipynb` §4.3.

**Ground Truth:**
- 9,855 pares positivos confirmados (EXP + nombre coinciden entre CSVs)
- 4,341 entidades vinculables (presentes en 2+ CSVs)
- 1,569 pares residuales (EXP igual, nombre distinto tras normalización)
- Diferencia de +31 pares vs notebook — notebook excluye expedientes NaN, pipeline los incluye

Parquet final: `record_id`, `source_db`, `text`, `entity_id`.

**tesis2:** requiere actualizar `_COL_MAP` en `build_dataset` para los nombres post-M7
(ej. `EXP` → `EXPEDIENTE` en Económico). Pendiente hasta confirmar si tesis2 se usará.

### Tokens especiales y serialización B1/B2 — estrategia DITTO (2026-04-29)

**Lista definitiva de tokens especiales:**
```
[BLK_ID]  [BLK_ADMIN]  [BLK_CLIN]  [BLK_GEO]  [BLK_SOCIO]
[COL]  [VAL]
```

**Nulos — estrategia DITTO:** los campos nulos se serializan como `[COL] columna [VAL] NULL`,
donde "NULL" es texto plano (no token especial). Esto sigue la implementación de DITTO
(Mudgal et al., 2022) y aprovecha el conocimiento previo de los modelos sobre el término
"NULL" adquirido durante preentrenamiento. Se descartó `[VAL_NULL]` como token especial
separado y variantes en español ("Vacío", "ND", "AUSENTE") tras verificar que los
modelos en español tienen representación adecuada de "NULL" como término técnico.

**Registro de tokens en `download_model.py`:**
Los tokens especiales se registran al guardar los modelos localmente mediante
`tokenizer.add_special_tokens()` + `model.resize_token_embeddings()`. Sin este paso,
el tokenizador fragmenta cada token especial en subwords:
```
[BLK_ID]   → '[', 'BLK', '_', 'ID', ']'     (5 tokens → 1)
[COL]      → '[', 'COL', ']'                 (3 tokens → 1)
[VAL]      → '[', 'VAL', ']'                 (3 tokens → 1)
```
**Estado:** implementado en `scripts/download_model.py` (`add_special_tokens` + `resize_token_embeddings`).

### Distribución de tokens y decisión de `max_seq_length`

`max_seq_length=512` para todos los modelos. Análisis sobre dataset zero-shot (tesis0):
ningún CSV supera 512 tokens con serialización col:val. Con tokens especiales sin registrar,
Económico rozaba 532 tokens con BETO — con tokens registrados la cifra baja (pendiente remedir).
El default de 128 de `paraphrase-multilingual` truncaba el 98–100% de registros de Económico y TS.

### `augmentation.py` — transformaciones on-the-fly

La aumentación de datos se aplica durante el entrenamiento y no se persiste en
disco. Esto evita almacenar múltiples versiones alteradas de cada registro y
garantiza variabilidad estocástica en cada época.

**Operadores implementados:**
- `shuffle_blocks` — permuta bloques `[BLK_*]` como unidades atómicas
- `shuffle_columns` — permuta pares `[COL]/[VAL]` dentro de cada bloque
- `mask_attributes` — reemplaza valores de campo con NULL (simula datos faltantes)
- `inject_typos` — introduce errores tipográficos en valores de texto
- `delete_span` — elimina una secuencia contigua de tokens no-especiales (**desactivado**, `prob=0.0`)

**Decisión (2026-05-01) — campos protegidos:**
Experimentos de ablación revelaron que `mask_attributes` puede enmascarar NOMBRE_DEL_PACIENTE
y NO_EXPEDIENTE — los campos que sirvieron como llave para etiquetar el ground truth.
Enmascarar estos campos en el positive crea pares sintéticos contradictorios: el modelo
no puede aprender que anchor y positive son el mismo paciente si el campo más discriminativo
desaparece. **Implementado:** `AugmentationConfig` tiene `protected_fields` con los 6 campos
llave de los 3 CSVs (`nombre`, `NOMBRE_DEL_PACIENTE`, `NOMBRE_COMPLETO`, `expediente`, `EXP`, `EXPEDIENTE`).

**Decisión (2026-05-01) — delete_span desactivado:**
Borrar tokens contiguos sin discriminar tipo de campo puede eliminar dígitos de expediente
o letras de nombre — información de identidad crítica. El operador requiere rediseño
(ej. restringirlo a campos no protegidos) antes de activarlo.


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

### Separación `train_biencoder.py` / `bce.py`

Las dos etapas del pipeline tienen objetivos de entrenamiento distintos:

- **Etapa 1 (Bi-Encoder):** optimizada con MNRL para construir un espacio métrico
  adecuado para búsqueda vectorial eficiente (alto recall). Implementada en
  `src/record_linkage/training/train_biencoder.py`; CLI en `scripts/run_train_biencoder.py`.
- **Etapa 2 (Cross-Encoder):** optimizada con BCE como clasificador binario de alta
  precisión sobre los candidatos recuperados. Pendiente de implementar en `bce.py`.

### `.sh` en la raíz del repo

Los scripts de lanzamiento SLURM deben vivir en la raíz del proyecto en el
cluster. La directiva `--chdir` de SLURM fija ese punto como directorio de
trabajo, y todas las rutas relativas dentro del `.sh` (incluyendo `logs/`,
`scripts/`, `src/`) se resuelven desde ahí. Colocarlos en una subcarpeta
como `scripts/` rompe esta resolución sin configuración adicional no probada.

---

## Pendientes de diseño

- **Rediseñar augmentación** — simular diferencias reales entre CSVs del INER (orden invertido de nombre Com→Eco, campos ausentes por diseño, codificaciones distintas). Rehacer `delete_span` field-aware.
- **Pipeline de evaluación** — diseño completo pendiente de discutir con asesor (ver Pregunta 6 en `MEMORY.md`). Incluye: qué métricas reportar, direccionalidad, separación de responsabilidades entre scripts.
- **DDP multi-GPU** — implementar `train_biencoder_ddp.py` + script SLURM con `torchrun --nproc_per_node=2` para aprovechar las 2 GPUs Titan RTX por nodo.
- **Resolver falsos negativos** — revisar los ~11,500 pares con expediente compartido con similitud difusa; ver §Deuda técnica en `MEMORY.md`.
- Decidir la estrategia de logging y experiment tracking (MLflow, W&B, etc.)
