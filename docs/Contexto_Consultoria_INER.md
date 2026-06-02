# Consultoría INER: Integración y Homologación de Bases de Datos

**Consultor:** Ing. Uziel Isaí Luján López (Maestría en Cómputo Estadístico, CIMAT)

**Organización Receptora:** Instituto Nacional de Enfermedades Respiratorias (INER)

**Responsable:** Dra. Mariana Esther Martínez Sánchez
**Marco:** Protocolo B40-25

---

## 1. Definición del Problema
El INER enfrenta un desafío crítico de interoperabilidad y calidad de datos derivado de la emergencia sanitaria por COVID-19. La información de los pacientes se encuentra dispersa en tres fuentes principales. Estos archivos contienen datos relacionados con el **costo económico** de la atención médica, los **diagnósticos y comorbilidades** de los pacientes, y la información de **trabajo social** en un periodo que comprende desde marzo de 2020 hasta mayo de 2023, los cuales se cree que carecen de identificadores únicos (llaves primarias/foráneas) 100% confiables, presentan alta duplicidad y nula estandarización tipográfica.

El objetivo de esta consultoría es resolver el problema de calidad e integración desde una perspectiva de **Análisis e Ingeniería de Datos**, sentando las bases limpias y estructuradas que posteriormente alimentarán el sistema moderno de ligado de registros basado en Procesamiento de Lenguaje Natural y Aprendizaje Profundo (Proyecto de Tesis). El encuadre general de ambos ejes se resume en [[Contexto_Maestro_Proyecto.md]], y la arquitectura técnica de la etapa de tesis se desarrolla en [[Metodologia_arquitectura.md]].

---

## 2. Objetivos Específicos de la Consultoría
Para resolver la problemática institucional, los esfuerzos técnicos se centrarán en:
1. **Análisis exploratorio de Datos:** Analizar la estructura, completitud, información y calidad de las tres bases de datos proporcionadas. Definir estrategias de limpieza y consolidación basadas en los hallazgos.
3. **Análisis de duplicados y desarrollo de métricas sintácticas:** Identificar columnas susceptibles a ser llaves foráneas para identificar duplicados y vinculaciones. Implementar métodos de comparación de cadenas de texto (similitud), con especial enfoque en la resolución de entidades, utilizando los identificadores detectados (ej. nombres y expedientes).
2. **Estandarización y Diccionarios:**  Creación de diccionarios de datos que documenten el estado original y las transformaciones aplicadas.
4. **Consolidación Relacional:** Unificar las tablas dispersas en un esquema relacional estructurado, disponible para sistemas de procesamiento automatizado.

---

## 3. Entregables de la estancia

El proyecto de consultoría se considerará exitoso tras la entrega de los siguientes productos:

### Producto 1: Reporte de Metodología y EDA
Documento analítico detallando:
* Los hallazgos del Análisis Exploratorio de Datos individual de cada base.
* Las decisiones de consolidación tomadas.
* Los métodos de comparación sintáctica utilizados y la justificación de las métricas aplicadas para nombres propios con el objetivo de detectar duplicados y vinculación entre bases.

Este producto también sirve como puente hacia los lineamientos del proyecto de tesis descritos en [[ElProtocolodeInvestigacion.md]].

### Producto 2: Pipeline de limpieza y procesamiento (Código Fuente)
Módulos desarrollados en Python en un repositorio organizado que:
* Implementen la limpieza y preprocesamiento de los CSV crudos según indicaciones del INER (pendiente de confirmar el nivel)
* Estén justificados en los hallazgos del análisis exploratorio de datos
* Garanticen la total reproducibilidad del proceso.

### Producto 3: Base de Datos Consolidada
Un esquema relacional limpio, libre de redundancias y estructurado en múltiples tablas normalizadas (pendiente de definir el diseño final) vinculadas lógicamente mediante llaves primarias y foráneas (PK/FK). Este diseño garantiza la integridad referencial y la disponibilidad de la información para su consulta directa o para sistemas de procesamiento automático.

### Producto 4: Diccionarios de Datos
El INER propuso "Creación de un diccionario de datos de las bases de datos identificando las columnas susceptibles a ser consideradas llaves foráneas". Se entregan dos documentos técnicos complementarios que documentan el ciclo de vida del dato, desde la exploración de su estado original hasta su consolidación final:

1. **`Diccionario_Origen_INER.csv` (El Diagnóstico):** Bitácora de auditoría que perfila la calidad y anomalías detectadas en las bases de datos originales. Versión final lean de 6 columnas:
   ```| Archivo_Origen | Columna | Tipo_Dato_Detectado | Unicos | Nulos | Descripcion | Hallazgos |```

   * *Definiciones clave:*
        * **Descripcion:** Significado semántico de la columna y panorama general (categorías dominantes, rango, formato), derivado del Cap 1 del reporte (Caracterización de Columnas).
        * **Hallazgos:** Anomalías y aspectos de calidad: typos, inconsistencias tipográficas, valores faltantes, duplicados, valores fuera de rango, formatos incorrectos, redundancias entre columnas, etc., derivados del Cap 2 del reporte (Calidad de los Datos).

   * *Columnas consideradas durante el diseño y excluidas en la versión final:*
        * **`Transformacion`** — descartada porque las reglas de limpieza viven en el código del pipeline (`preprocessing.py`, `consolidation.py`) y replicarlas en el diccionario crearía dos fuentes de verdad divergentes. El JSON consolidado entrega datos crudos del registro, las transformaciones aplicadas para la vinculación son trazables en los scripts.
        * **`Total_Registros`** y **`Pct_Nulos`** — `Total_Registros` es redundante (igual a `Unicos + Nulos` para campos sin duplicados, y a una constante por archivo); `Pct_Nulos` se deriva trivialmente de `Nulos / Total_Registros`. No aportan información nueva.
        * **`Tipo_Variable`** y **`Bloque_Semantico`** — útiles internamente para el diseño de la serialización del eje tesis, pero ajenas a la auditoría que pide el INER. Quedan documentadas en `docs/Metodologia_arquitectura.md` y en el código (`SEMANTIC_BLOCKS`).

2. **`Diccionario_Final_INER.csv` (El Entregable):** Mapa técnico de la estructura del JSON consolidado entity-centric. La especificación formal vive en **`consolidated_entities.schema.json`** (JSON Schema Draft 2020-12), que es la fuente de verdad validable; el CSV es una proyección legible de 3 columnas derivada del Schema:
   ```| campo | tipo | descripcion |```

   El schema describe la jerarquía `entity → linked_items[] → scores[]` y se complementa con `metodos_comparacion.json` (catálogo de métricas: nombre, campos sobre los que opera, rango, semántica). El esquema relacional planteado originalmente (`Tabla_SQL | Columna_Final | Tipo_Dato_Final | Llave | Descripcion_Limpia`) se descartó porque el entregable final es JSON anidado, no tabular — una proyección a SQL crearía pérdida de información en los campos array (`linked_items`, `scores`).

---

## 4. Impacto Esperado

Los productos entregables de esta consultoría resuelven la necesidad operativa del INER, pero estratégicamente constituyen el analisis exploratorio de los datos que utilizará el proyecto de tesis y más importante: **el cimiento del conjunto de datos etiquetado** indispensable para el aprendizaje del modelo:

* **Insumo para la Arquitectura Neuronal:** El análisis exploratorio de los datos justificará la estructura final de los registros serializados (secuencias de texto y *Early Fusion*) requerida por los modelos (SBERT y DITTO).
* **Fundamento para el Aprendizaje Auto-Supervisado:** El perfilado exhaustivo de datos (EDA) y la tipificación de errores de captura (typos) proveerán el contexto necesario para diseñar la estrategia de *Data Augmentation*, permitiendo entrenar la red neuronal multiplicando sintéticamente los pares de registros vinculados.
* **Institucional:** Se transiciona a un pipeline moderno, entregando al INER un sistema que garantiza la vinculación de pacientes en bases de datos fragmentadas para un seguimiento clínico confiable.


## 5. Actualización de pasos siguientes

* **Confirmación del nivel de limpieza requerido por el INER:** Pendiente de definir el nivel de limpieza requerido por el INER, lo cual impactará directamente en la estructura de la base de datos consolidada y del diccionario final. Esto se definirá en conjunto con el asesor de la estancia, considerando sus necesidades operativas. Por el momento, los módulos de limpieza implementados en el pipeline son robustos, generales y basados en los hallazgos del análisis exploratorio de datos, pero se pueden ajustar para incluir reglas más estrictas o específicas según las indicaciones del INER.

|Source DB | Entity_id| Registros |

23706 repartidos en 3 CSV según su source DB

$Entity_{id} = [0 , ...., e_i ..., , N-1]$ $N$: Entidades únicas $e_i >= 1$, no esta ordenado en orden creciente


* **Implementación de métricas de similitud para vinculación de registros:**
El análisis exploratorio de datos reveló que los nombres propios y expedientes son los campos más susceptibles a ser considerados como llaves foráneas para la vinculación entre bases. Sin embargo, dada la ausencia de identificadores únicos 100% confiables, se definió que el enfoque de comparación sintáctica y métricas clásicas se centrará en la inspección de los registros vinculados entre bases pendientes a confirmar manualmente. Actualmente hay 1569 pares de registros en esta situación, la idea es que utilizando estas métricas, se puedan reducir el número de registros a revisar manualmente, estableciendo umbrales de similitud para automatizar la vinculación de registros con alta confianza.
Pendiente de definir con el asesor de la estancia la interfaz de visualización de estos pares para su revisión manual, se ha propuesto una tabla de excel que muestre los registros lado a lado con las métricas de similitud calculadas para facilitar la inspección y la toma de decisiones. Esto permitirá acelerar el proceso de revisión manual, enfocándose en los casos más difíciles de vincular.


Un `.json` :

- Sirve como log para identificar la llave (expediente + nombre normalizado)
- (exp, nombre) |

Un archivo que narre el proceso de identificación de los pares, comenzando con solo filtrando expediente compartido, esto produjo 11424 pares de registros positivos, luego agregué el campo nombre normalizado robusto ->  identificación directa de la llave (exp, nombre) que dió 9855 , restaron 1569 y siguiendo con el cálculo de distancias Levenshtein, Jaro Winkler, para mostrar cómo se filtran los pares.

Pasos de identificación utilizados para llegar a los pares finales (por Uziel):
1. Expedientes coincidentes
2. Expediente + Nombre normalizado Version 1
3. Expediente + Nombre normalizado Version 2
4. Calculo de métricas clásicas con un umbral (pendiente de implementar y definir umbral)

---

## 6. Diseño del entregable de vinculación (propuesto post reunión con asesora, 2026-04-30)

Este entregable resuelve el pendiente de **Implementación de métricas de similitud** y la solicitud de la Dra. de visualizar singletons, duplas y tríadas. Se propone un Excel con dos hojas complementarias vinculadas por `entity_id`:

### Hoja 1 — Pares (11,424 filas)

Una fila por par de registros que comparten expediente entre distintos CSV. Narra el proceso de filtrado progresivo:

| `entity_id` | `source_a` | `source_b` | `exp` | `nombre_a` | `nombre_b`| `paso_identificacion` | `levenshtein` | `jaro_winkler` |
|---|---|---|---|---|---|---|---|---|

- **`paso_identificacion`** toma 3 valores (criterio de confirmación del par):
  - `llave_exacta` — expediente + nombre_norm_v2 coinciden (9,855 pares confirmados automáticamente)
  - `metrica_clasica` — expediente coincide, nombre no exacto pero supera umbral Jaro-Winkler/Levenshtein (subconjunto de los 1,569)
  - `no_confirmado` — expediente coincide pero nombre no supera ningún umbral; candidatos a revisión manual o descartar como vinculados
- **`levenshtein` y `jaro_winkler`** se calculan para los 11,424 pares — permite verificar que `llave_exacta` tiene métricas altas y `no_confirmado` valores más bajos o mixtos.
- Las tríadas aparecen aquí como múltiples pares (C(3,2) = 3 filas por entidad en 3 CSV).
- Los **singletons no aparecen** en esta hoja por construcción (necesitan al menos 2 registros para formar un par).

### Hoja 2 — Entidades (una fila por `entity_id`)

Vista resumida a nivel de paciente. Aquí sí aparecen los tres tipos:

| `entity_id` | `exp` | `source_db_list` | `tipo_entidad` |
|---|---|---|---|

- **`tipo_entidad`**: `singleton` (1 CSV), `dupla` (2 CSV), `tríada` (3 CSV).
- **`source_db_list`**: lista de CSV donde aparece la entidad, ej. `['Comorbilidad', 'Económico']` para una dupla. Para facilitar el filtrado en Excel se propone reemplazarla por columnas booleanas individuales: `en_comorbilidad`, `en_econo`, `en_trabajo_social` — así la asesora puede filtrar directamente por base sin parsear listas.
- Complementa la Hoja 1: permite navegar de un par a su clasificación de entidad y viceversa.

### Nota de diseño
- La Versión 1 de normalización de nombre se omite del proceso — no aportó mejoras respecto a v2 y añade ruido a la narrativa.
- Pendiente de aprobación final por la asesora antes de implementar.

---

## 7. Entregable oficial aprobado: JSON consolidado entity-centric (2026-05-12)

**Estado:** confirmado por la Dra. Mariana Esther Martínez Sánchez el 2026-05-12. Supersede el diseño Excel de 3 hojas de la sección 6 (conservada como histórico). Revisión final con los VIC del INER programada para el 2026-05-14.

**Documento formal enviado a la asesora:** `~/Documents/Maestria/Tesis/Propuesta_Entregables_INER/propuesta_entregable.tex` — sección `Propuesta unificada: JSON por entidad`.

### 7.1 Diseño técnico

Un único archivo `.json` consolidado: arreglo de **16,222 objetos**, uno por entidad única (`entity_id`). Cada objeto agrupa los registros originales de las 3 bases que corresponden al mismo paciente, sus pares de vinculación bilateral con métricas y un campo `decision` editable para la revisión manual de la asesora.

```json
{
  "entity_id": 1219,
  "exp": 55810,
  "tipo_entidad": "dupla",
  "decision": null,
  "pairs": [
    {
      "source_a": "Comorbilidad",
      "source_b": "Trabajo Social",
      "criterio": "no_confirmado",
      "nombre_a": "APOLINAR DANIEL PRADO",
      "nombre_b": "APOLINAR ARMANDO PRADO",
      "lev": 7,
      "jw": 0.812
    }
  ],
  "records": [
    { "source": "Comorbilidad",   "exp": 55810, "nombre_norm": "...", "edad": 54, ... },
    { "source": "Trabajo Social", "exp": 55810, "nombre_norm": "...", "diagnostico": "...", ... }
  ]
}
```

**Tratamiento de heterogeneidad y casos límite:**
- Cada elemento de `records` conserva solo los campos propios de su fuente (no se imponen columnas comunes ni nulos artificiales).
- Expedientes faltantes (los 31 casos NaN-TS de la base Económico) se serializan como `"exp": null`.
- Tríadas: el array `pairs` contiene los $\binom{3}{2}=3$ pares bilaterales (uno por cruce entre bases).
- Duplas: `pairs` con un solo par.
- Singletons: `pairs = []`.

### 7.2 Insumos del repo para construir el JSON

| Campo del JSON | Fuente en el repo |
|---|---|
| `entity_id`, `exp`, `tipo_entidad` | derivado de `pairs_classified.parquet` y `dataset_v2.parquet` (artefactos del pipeline v2 en `~/Data/INER/processed/tesis1/`) |
| `pairs[]` | filas de `pairs_classified.parquet` agrupadas por `entity_id` (ya contienen `criterio`, `lev`, `jw`, fuentes y nombres) |
| `records[]` | filas originales de los 3 CSV crudos en `~/Data/INER/raw/`, unidas por `(source, source_row_id)` y filtradas por las que pertenecen a cada `entity_id` |
| `decision` | inicializado en `null`; se actualiza vía `overrides.csv` tras revisión manual de la Dra. |

`tipo_entidad` se calcula directamente como `len(records)`: 1 → `singleton`, 2 → `dupla`, 3 → `tríada`. Las columnas booleanas de presencia por base (`en_comorbilidad`, `en_econo`, `en_trabajo_social`) que pedía la versión Excel se derivan inspeccionando `records[].source`, no se serializan en el JSON.

### 7.3 Posición en el pipeline

El JSON pasa a ser el **tronco común** del que cuelgan ambos ejes del proyecto:

```
pairs_classified.parquet  +  CSV crudos
              │
              ▼
   build_consolidated_json.py     ← (por implementar)
              │
              ▼
    consolidated_entities.json    ← fuente de verdad
              │
              ├─→ pairs_classified.xlsx   (vista derivada para revisión manual de la Dra.)
              ├─→ dataset_v2.parquet      (serialización para ingesta del modelo — eje tesis)
              └─→ Diccionario_Final_INER  (derivado del schema del JSON)
```

`overrides.csv` sigue siendo el puente bidireccional: la Dra. anota en el Excel, las anotaciones se traducen a `overrides.csv`, y un `finalize` regenera el JSON con `decision` actualizado.

### 7.4 Cobertura de los entregables originales

Este único artefacto resuelve directamente 3 de los 4 productos de la sección 3:

| Entregable original | Cómo lo cubre el JSON |
|---|---|
| Producto 2 — Pipeline de limpieza | El script generador (`build_consolidated_json.py`) más los módulos ya existentes en `src/record_linkage/data/` y `utils/` constituyen el pipeline reproducible. |
| Producto 3 — Base consolidada | La estructura entity-centric integra los registros de las 3 fuentes bajo `entity_id` común, con integridad referencial sin redundancias. Reemplaza el esquema relacional multi-tabla originalmente propuesto. |
| Producto 1 — Reporte de metodología | Los campos `criterio`, `lev`, `jw` dentro de `pairs` documentan in-situ el método de vinculación; el reporte ya entregado (`Reporte_INER.pdf`) cubre el resto. |
| Producto 4 — Diccionarios | El schema del JSON es la base directa para derivar `Diccionario_Final_INER.csv` (entregable separado pendiente). |

### 7.5 Pendientes

1. ✅ **(2026-05-27) Implementado — generador del JSON consolidado (schema v2).** `src/record_linkage/data/consolidation.py` (`build_entity_objects`, lógica pura) + `scripts/build_consolidated_json.py`. Salida: `~/Data/INER/processed/iner/consolidated_entities.json` (15,283 entidades, 29 MB). Schema **v2** = `items` anidado (`{item, source, linking_values, record}`) sin la redundancia de arrays paralelos; `scores` es un protocolo de métodos extensible (módulo `data/comparison_methods.py`), recalculado y solo cross-source. Diseño completo en `propuesta_entregable_JSON.md` → "Diseño v2"; decisiones en `design_decisions.md` → bloque ESTADO 2026-05-27.
2. ✅ **(2026-05-27) Implementado — documentación del entregable.** Spec formal = **JSON Schema** (`docs/consolidated_entities.schema.json`, Draft 2020-12, editado a mano y validable). `scripts/build_data_dictionary.py` lo proyecta a `~/Data/INER/processed/iner/Diccionario_Final_INER.csv` (vista plana `campo|tipo|descripcion`, ~14 filas, derivada del schema), emite `metodos_comparacion.json` (catálogo desde `comparison_methods.REGISTRY`) y copia el schema al bundle.
3. **Revisión con los VIC del INER** — programada para 2026-05-14.