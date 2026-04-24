# entity-resolution-nlp

Sistema de ligado de registros (Record Linkage) basado en aprendizaje profundo y NLP, aplicado a bases de datos de pacientes COVID-19 del INER. Proyecto de tesis de maestría en Cómputo Estadístico — CIMAT Unidad Monterrey.

---

## Contexto

El INER cuenta con tres bases de datos independientes de pacientes COVID-19 que no comparten una llave de identificación 100% confiable. Este proyecto construye un sistema moderno para vincularlas a nivel semántico usando modelos de lenguaje pre-entrenados (Bi-Encoder + Cross-Encoder), sin depender de coincidencias exactas de campos.

Las tres bases:

| CSV | Registros | Contenido |
|-----|-----------|-----------|
| Diagnósticos y Comorbilidades | 4,278 | diagnóstico principal, comorbilidades, fechas |
| Costos y Económico | 4,632 | costos de atención, datos socioeconómicos |
| Trabajo Social | 14,796 | datos demográficos, familia, situación social |

---

## Estructura del repositorio

```
entity-resolution-nlp/
├── src/record_linkage/
│   ├── config.py          # Rutas centralizadas (DATA_ROOT, PROCESSED_DIR, etc.)
│   ├── data/
│   │   ├── preprocessing.py   # Módulos M0–M6, perfiles A / B1 / B2
│   │   ├── dataset.py         # Serialización a bloques semánticos + .parquet
│   │   └── consolidation.py   # (en desarrollo)
│   ├── models/            # Definición de Bi-Encoder y Cross-Encoder
│   ├── training/          # Loops de entrenamiento MNRL
│   └── inference/         # Búsqueda ANN e inferencia en producción
├── scripts/
│   ├── run_preprocessing.py   # CLI: limpieza de CSVs crudos
│   └── run_dataset.py         # CLI: construcción del dataset .parquet
├── notebooks/             # EDAs y análisis de duplicados
└── docs/                  # Documentación del proyecto
```

---

## Inicio rápido

### 1. Entorno

Crear el entorno con micromamba e instalar dependencias con uv:

```bash
# Crear entorno con Python 3.11
micromamba create -n tesis python=3.11 -c conda-forge -y
micromamba activate tesis

# Instalar uv dentro del entorno
pip install uv

# Instalar dependencias del proyecto (incluye record_linkage como paquete editable)
uv pip install -e .
```

El flag `-e` instala `record_linkage` en modo editable — cualquier cambio en `src/` se refleja sin reinstalar. Las dependencias de desarrollo (jupyter, pytest, ruff) están declaradas en `pyproject.toml` bajo `[project.optional-dependencies]` y se instalan por separado si se necesitan.

Una vez creado el entorno, puedes activarlo en sesiones posteriores con:

```bash
micromamba activate tesis
```

### 2. Preprocesamiento

```bash
# Verificar rutas
python scripts/run_preprocessing.py --perfil B1 --check-paths

# Limpiar CSVs (Perfil B1: mínima intervención, solo fix en encoding, nombres de columna originales)
python scripts/run_preprocessing.py --perfil B1
```

Perfiles disponibles:

| Perfil | Módulos | Uso |
|--------|---------|-----|
| `A` | M0→M1→M2→M3→M4a→M4b→M4c→M5 | Entregables de consultoría INER |
| `B1` | M1→(M4b si TS) | Tesis — columnas originales, compatible con SEMANTIC_BLOCKS |
| `B2` | M1→M2→M4a→M4b | Tesis — columnas renombradas (SEMANTIC_BLOCKS pendiente) |

### 3. Dataset

```bash
# Verificar rutas
python scripts/run_dataset.py --output ~/Data/INER/processed/dataset.parquet --check-paths

# Construir dataset serializado
python scripts/run_dataset.py --output ~/Data/INER/processed/dataset.parquet
```

Salida: `dataset.parquet` con columnas `record_id`, `source_db`, `expediente`, `nombre`, `text`, `entity_id`.

---

## Serialización

Cada registro tabular se convierte en una secuencia de texto estructurada con bloques semánticos:

```
[BLK_ID] nombre: GARCIA LOPEZ MARIA [BLK_ADMIN] expediente: 12345 fechaing: 2021-03-10 [BLK_CLIN] diagnosticoprincipal: COVID-19 ...
```

Los tokens `[BLK_*]` se incluyen para fine-tuning; se omiten (`use_block_tokens=False`) para evaluación zero-shot con modelos pre-entrenados.

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| `docs/Contexto_Maestro_Proyecto.md` | Visión general, roadmap, estado de fases |
| `docs/Metodologia_arquitectura.md` | Arquitectura neuronal: SBERT, DITTO, MNRL |
| `docs/design_decisions.md` | Decisiones técnicas tomadas y su justificación |
| `docs/Contexto_Consultoria_INER.md` | Objetivos y entregables de la consultoría |

---

## Estado actual

- [x] EDA de las tres bases (Comorbilidad, Econo, Trabajo Social)
- [x] Extracción de verdad base — 4,110 entidades vinculables, ~9,855 pares confirmados
- [x] Pipeline de preprocesamiento (Perfiles A, B1, B2)
- [x] Serialización con bloques semánticos → `dataset.parquet` (23,706 registros)
- [ ] Data augmentation
- [ ] Entrenamiento Bi-Encoder con MNRL (HPC CIMAT)
- [ ] Evaluación zero-shot vs. fine-tuned
- [ ] Cross-Encoder como re-ranker
