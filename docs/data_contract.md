# Contrato de datos

Frontera técnica entre `consultoria-iner` y
`entity-resolution-nlp`.
Ambos repositorios se comunican exclusivamente mediante Parquet bajo `INER_DATA_ROOT`.

## Responsabilidades

- `consultoria-iner` limpia las fuentes, construye y revisa el ground truth, asigna `record_id` y `entity_id`, serializa los registros y publica las variantes de `dataset.parquet`.
- `entity-resolution-nlp` no lee datos crudos, no reconstruye el ground truth y no escribe dentro de `processed/default/output/`.
Produce splits, pares minados, modelos, entrenamiento, evaluaciones y embeddings.
`consultoria-iner` puede consumir los embeddings.

## Entrada: consultoría a tesis

Ruta:

```text
$INER_DATA_ROOT/processed/default/output/<variant>/dataset.parquet
```

Esquema obligatorio:

| Columna | Tipo | Restricción |
|---|---|---|
| `record_id` | `int64` | No nulo y único; identifica el mismo registro en todas las variantes |
| `source_db` | string | No nulo; `Económico`, `Comorbilidad` o `Trabajo Social` |
| `text` | string | No nulo ni vacío; serialización definida por la variante |
| `entity_id` | `int64` | No nulo; clúster de identidad del silver standard |

Variantes admitidas: `tok_skipnull`, `tok_keepnull`, `notok_skipnull` y
`notok_keepnull`. Las cuatro deben compartir exactamente `record_id`, `source_db` y
`entity_id`; solo cambia `text`. La variante canónica es `tok_skipnull`.

`record_id` se asigna por posición global en el orden Económico, Comorbilidad y Trabajo
Social. Si el productor cambia el preprocesamiento, el número u orden de los registros o
las decisiones de vinculación, deben regenerarse los datasets y todos los artefactos
neuronales derivados.

`entity_ids.parquet` no forma parte de esta interfaz. Es una proyección de
`dataset.parquet` utilizada internamente por `consultoria-iner` para sus entregables.

## Salida: tesis a consultoría

Ruta:

```text
$INER_DATA_ROOT/modeling/embeddings/<variant>/embeddings.parquet
```

Esquema:

| Columna | Tipo | Restricción |
|---|---|---|
| `record_id` | `int64` | Único; debe corresponder al dataset que originó el embedding |
| `embedding` | lista de `float32` | Vector L2-normalizado de dimensión constante |

La metadata `biencoder_provenance` registra checkpoint, ruta del dataset, variante,
número de registros, dimensión, normalización y fecha de creación. El productor de datos
consume este artefacto de forma opcional y solo mediante `record_id`.

## Propiedad de rutas

| Ruta relativa a `INER_DATA_ROOT` | Escritura | Lectura |
|---|---|---|
| `processed/default/output/` | `consultoria-iner` | Ambos repositorios |
| `modeling/` | `entity-resolution-nlp` | `entity-resolution-nlp`; embeddings opcionalmente por consultoría |
| `raw/`, directorios `clean/` e `interim/` | `consultoria-iner` | `consultoria-iner` |


## Estructura oficial de directorios

```text
INER_DATA_ROOT/
├── raw/                                      # Entrada protegida de consultoría
├── processed/                                # Artefactos de consultoria-iner
│   └── default/
│       ├── clean/                            # CSV limpios
│       ├── interim/                          # Pares candidatos y revisión manual
│       ├── output/                           # Interfaz consultoría → modeling
│       │   ├── entity_ids.parquet            # Uso interno de consultoría
│       │   └── <variant>/
│       │       └── dataset.parquet           # Entrada de entity-resolution-nlp
│       └── deliverables/
│           └── audits/                       # Auditorías producidas por consultoría
│
└── modeling/                                 # Artefactos de entity-resolution-nlp
    ├── data/
    │   └── <variant>/
    │       ├── split.parquet                 # Dataset con asignación train/val/test
    │       ├── pairs_train.parquet           # Pares canónicos del Cross-Encoder
    │       ├── pairs_val.parquet
    │       └── pairs_test.parquet
    │
    ├── models/
    │   ├── pretrained/                       # Backbones preparados para uso offline
    │   ├── biencoder/
    │   │   └── <variant>/
    │   │       └── <run>/
    │   │           ├── best/
    │   │           └── training_history.json
    │   └── crossencoder/
    │       └── <variant>/
    │           └── <run>/
    │               ├── best/
    │               ├── training_history.json
    │               └── calibration.json      # Si se ejecuta calibración
    │
    ├── embeddings/
    │   └── <variant>/
    │       └── embeddings.parquet            # Exportación canónica del modelo ganador
    │
    └── outputs/
        ├── evaluation/
        │   ├── biencoder/
        │   │   ├── zeroshot/
        │   │   └── finetuned/
        │   └── crossencoder/
        │       ├── classification/
        │       └── calibration/              # Métricas e incertidumbre por vínculo
        ├── figures/
        │   ├── embeddings/
        │   └── training_curves/
        └── diagnostics/
            ├── mnrl_batches/
            │   └── <run>/
            └── tokenization/
```

Nota:
- `processed/default/deliverables/audits/` queda reservado para auditorías generadas por consultoría después de consumir los embeddings.