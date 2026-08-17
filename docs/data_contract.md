# Contrato de datos

Frontera técnica entre `consultoria-iner`, productor de los datos preparados, y
`entity-resolution-nlp`, consumidor del dataset y productor de artefactos neuronales.
Ambos repositorios se comunican exclusivamente mediante Parquet bajo `INER_DATA_ROOT`.

## Responsabilidades

`consultoria-iner` limpia las fuentes, construye y revisa el ground truth, asigna
`record_id` y `entity_id`, serializa los registros y publica las variantes de
`dataset.parquet`. `entity-resolution-nlp` no lee datos crudos, no reconstruye el ground
truth y no escribe dentro de `processed/default/output/`.

`entity-resolution-nlp` produce splits, pares minados, modelos, evaluaciones y embeddings.
`consultoria-iner` puede consumir los embeddings, pero no modifica los artefactos del
pipeline neuronal.

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

## Cambios de interfaz

Un cambio en columnas, tipos, valores válidos, semántica u orden de `record_id` requiere
actualizar este contrato y regenerar los artefactos dependientes. Un cambio exclusivo en
`text` debe publicarse como una variante nueva o reemplazar explícitamente una variante
existente, notificando qué modelos y evaluaciones deben repetirse.
