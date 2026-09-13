# Contexto del Manuscrito de Tesis

Documento puente entre el repositorio de tesis (`entity-resolution-nlp`), su manuscrito LaTeX versionado en `manuscript/` y el repositorio separado de consultoría (`consultoria-iner`). Su objetivo es indicar qué fuentes usar para redactar cada parte de la tesis sin reconstruir contexto desde cero.

---

## 1. Rutas canónicas

### Manuscrito de tesis

- Carpeta: `manuscript/`
- Archivo principal: `manuscript/Tesis_UzielLujan.tex`
- PDF compilado: `manuscript/out/Tesis_UzielLujan.pdf` (artefacto local ignorado por Git).
- El manuscrito se versiona en este repositorio; incluye fuentes, bibliografía, figuras, portada y logotipos.

### Reporte oficial de consultoría

- Carpeta: `~/Documents/Maestria/Consultoria/Reporte_Final/`
- Archivo principal: `~/Documents/Maestria/Consultoria/Reporte_Final/Reporte_INER.tex`
- PDF compilado: `~/Documents/Maestria/Consultoria/Reporte_Final/out/Reporte_Final_INER.pdf`
- Esta es la fuente canónica para extraer material de datos, análisis exploratorio, vinculación determinística y ground truth.

### Repo separado de consultoría

- Repo: `~/Projects/consultoria-iner/`
- Bitácora: `~/Projects/consultoria-iner/MEMORY.md`
- Documento puente clave: `~/Projects/consultoria-iner/docs/interfaz_consultoria_tesis.md`

---

## 2. Estado actual del manuscrito

El manuscrito versionado contiene la versión actual de la tesis. Se compila de forma autónoma desde `manuscript/`.

Archivos activos desde `Tesis_UzielLujan.tex`:

| Archivo | Estado | Rol actual |
|---|---|---|
| `Capitulos/01.tex` | Primera versión | Introducción, planteamiento, objetivos y contribución |
| `Capitulos/02.tex` | Primera versión | Marco teórico y estado del arte; requiere revisión bibliográfica |
| `Capitulos/03.tex` | Redactado | Datos y preprocesamiento |
| `Capitulos/04.tex` | Redactado | Metodología neuronal Retrieve & Rerank |
| `Capitulos/05.tex` | Redactado | Resultados y discusión |
| `Capitulos/06.tex` | Primera versión | Conclusiones, limitaciones y trabajo futuro |

Carpetas del manuscrito:

- `Figuras/`: diagramas TikZ y figuras de resultados incluidas en el manuscrito.
- `Bibliografia/referencias.bib`: bibliografía versionada del manuscrito.

---

## 3. Capítulo 3: Datos y Preprocesamiento

Este capítulo puede avanzar de forma sustancial usando el reporte oficial de consultoría casi como fuente directa, adaptando el tono de consultoría a tesis.

Fuentes principales:

- `~/Documents/Maestria/Consultoria/Reporte_Final/Capitulos/1_Introduccion.tex`
- `~/Documents/Maestria/Consultoria/Reporte_Final/Capitulos/3_2_Metodos.tex`
- `~/Documents/Maestria/Consultoria/Reporte_Final/Capitulos/4_1_Resultados.tex`
- `~/Documents/Maestria/Consultoria/Reporte_Final/Capitulos/5_Resumen_Comparativo.tex`
- `~/Projects/consultoria-iner/docs/interfaz_consultoria_tesis.md`

Contenido transferible:

- Contexto INER: tres bases de pacientes COVID-19, sistemas heterogéneos, ausencia de llave primaria confiable.
- Volumen de datos: 23,706 registros totales.
- Fuentes:
  - Económico: 4,632 registros, 24 columnas.
  - Diagnóstico/Comorbilidad: 4,278 registros, 24 columnas.
  - Trabajo Social: 14,796 registros, 20 columnas.
- Caracterización semántica de columnas: identificación, clínica, geográfica, socioeconómica y administrativa.
- Problemas de calidad:
  - Diferente orden de nombres entre fuentes.
  - Encoding roto `Ñ -> ?` en Comorbilidad.
  - NBSP y caracteres problemáticos.
  - Expediente faltante en Económico.
  - Expediente no confiable como llave única.
- Normalización robusta:
  - uppercase.
  - descomposición NFD.
  - eliminación de caracteres no alfabéticos.
  - ordenamiento alfabético de tokens.
- Construcción del ground truth determinístico + revisión humana:
  - 11,487 pares candidatos.
  - 9,855 `llave_exacta`.
  - 1,118 `metrica_clasica`.
  - 514 revisión manual: 493 `match` + 21 `no_match`.
  - 11,466 pares positivos finales.
  - 15,283 entidades únicas.
  - 4,605 entidades vinculables.
   - 10,678 entidades de una sola base de datos.

Lectura metodológica para tesis:

- Este ground truth debe describirse como *silver standard*, no como verdad absoluta.
- PRADO APOLINAR demuestra empíricamente que incluso la revisión humana puede contener errores.
- La consultoría resuelve la etapa de datos y produce el insumo que consume el eje neuronal.

---

## 4. Capítulo 4: Metodología

Fuentes principales dentro de este repo:

- `docs/Metodologia_arquitectura.md`
- `docs/design_decisions.md`
- `docs/Anexos/historial_generacion_dataset_etiquetado.md`
- `docs/Anexos/historial_desarrollo_pipeline_neuronal.md`
- `docs/Anexos/metricas_evaluacion.md`

Secciones redactables:

- Serialización tabular como texto mediante Early Fusion.
- Bloques semánticos: `[BLK_ID]`, `[BLK_CLIN]`, `[BLK_ADMIN]`, `[BLK_GEO]`, `[BLK_SOCIO]`.
- Tokens estructurales `[COL]` y `[VAL]`.
- Dataset final: `record_id`, `source_db`, `text`, `entity_id`.
- Variante canónica: `tok_skipnull`.
- Arquitectura Retrieve & Re-rank:
  - Bi-Encoder/SBERT para recuperación.
  - Cross-Encoder/DITTO-like para clasificación fina.
- Entrenamiento del Bi-Encoder:
  - BETO como backbone principal.
  - Multiple Negatives Ranking Loss.
  - split por entidad para evitar data leakage.
- Hard Negative Mining.
- Entrenamiento del Cross-Encoder:
  - BCEWithLogitsLoss.
  - `pos_weight=8`.
  - threshold optimizado sobre validación.

Matiz importante:

- `Metodologia_arquitectura.md` contiene diseño temprano y algunas partes prospectivas. Para la tesis final, priorizar los resultados confirmados en `docs/Anexos/historial_desarrollo_pipeline_neuronal.md` y `MEMORY.md` sobre formulaciones antiguas.

---

## 5. Resultados disponibles para redacción preliminar

Fuentes principales:

- `docs/Anexos/historial_desarrollo_pipeline_neuronal.md`
- `MEMORY.md`, Fases 10-12.
- `docs/Anexos/metricas_evaluacion.md`
- `docs/Anexos/propuesta_incertidumbre.md`

Resultados ya documentados:

- Experimento 2x2 de serialización:
  - Ganador oficial: `tok_skipnull`.
  - `val_loss=1.1029`.
  - `Recall@1=0.9837`.
  - `Recall@5=1.0000`.
  - `MRR=0.9998`.
  - `Delta_sep=11.2609`, promedio no ponderado sobre las seis direcciones cross-source.
  - La evaluación acumula todos los negativos disponibles por consulta; ya no usa submuestreo aleatorio.
- Hard Negative Mining sobre `tok_skipnull`:
  - Train: 8,513 positivos + 69,122 hard negatives.
  - Val: 1,851 positivos + 14,415 hard negatives.
  - Test: 1,860 positivos + 14,490 hard negatives.
  - 0 hard positives en los tres splits: el Bi-Encoder recupera todos los positivos conocidos en top-20.
- Cross-Encoder oficial:
  - `beto_bce_hpc_v2_tok_skipnull`.
  - 3 epochs, batch 16, max_seq 512, `pos_weight=8`.
  - threshold óptimo en validación: 0.12.
  - Test: F1=0.9997, Precision=1.0000, Recall=0.9995, 0 FP, 1 FN sobre 16,350 pares.
- Caso PRADO APOLINAR:
  - El único FN del Cross-Encoder resultó ser un error del ground truth.
  - Sirve para discusión de *silver standard*, auto-auditoría y human-in-the-loop.
- Visualización UMAP:
  - Evidencia visual del espacio métrico aprendido.
  - Islas compactas con registros cross-source mezclados por entidad.
- Calibración / incertidumbre Vía A:
  - CE ya estaba casi perfectamente calibrado en extremos (`ECE≈0`).
  - Temperature scaling degeneró por separabilidad perfecta.
  - Solo ~9 pares con entropía cruda apreciable.
  - PRADO confirma que confianza no equivale a corrección de etiqueta.

Pendientes antes de resultados finales:

- Implementar/ejecutar ablación de bloques semánticos si se desea una sección de explicabilidad fuerte.
- Decidir baseline formal con asesor.

---

## 6. Capítulo 2: Marco Teórico

El manuscrito ya tiene un borrador de marco teórico, pero requiere revisión bibliográfica formal.

Fuentes internas útiles:

- `docs/Anexos/Fundamentos_Transformers*` si existen en `docs/Anexos/`.
- `docs/Anexos/Sentence_BERT*` si existen en `docs/Anexos/`.
- `docs/Anexos/modelos-baseline*` si existen en `docs/Anexos/`.
- `docs/Metodologia_arquitectura.md`, sección bibliográfica final.
- `docs/Anexos/metricas_evaluacion.md` para formalizar métricas.

Temas mínimos a cubrir:

- Record Linkage clásico y Fellegi-Sunter.
- Métricas léxicas: Levenshtein, Jaro-Winkler.
- Entity Matching neuronal.
- Serialización de tablas para Transformers: DITTO.
- Sentence-BERT y Bi-Encoders.
- Cross-Encoders y atención cruzada.
- Multiple Negatives Ranking Loss.
- Blocking/retrieval y evaluación con Recall@K/MRR.
- Calibración, incertidumbre y *silver standard*.

Este capítulo probablemente debe redactarse después de una lectura bibliográfica dirigida para evitar depender solo de documentación interna.

---

## 7. Contrato consultoría -> tesis

Fuente canónica:

- `~/Projects/consultoria-iner/docs/interfaz_consultoria_tesis.md`

Resumen del contrato:

- Consultoría produce `output/<variant>/dataset.parquet` con columnas `[record_id, source_db, text, entity_id]`.
- Tesis consume la variante canónica `tok_skipnull`.
- Las cuatro variantes de serialización comparten `record_id` y `entity_id`; solo cambia `text`.
- Las 514 decisiones manuales son estables ante cambios de serialización, pero no ante cambios de preprocessing.
- La tesis exporta embeddings hacia consultoría en `$INER_DATA_ROOT/modeling/embeddings/tok_skipnull/embeddings.parquet` para habilitar `cos_biencoder` en el JSON consolidado.

---

## 8. Próximo orden recomendado de redacción

1. Capítulo 3: Datos y preprocesamiento, partiendo del reporte oficial de consultoría.
2. Capítulo 4: Metodología, usando `Metodologia_arquitectura.md`, `docs/Anexos/historial_generacion_dataset_etiquetado.md` y `docs/Anexos/historial_desarrollo_pipeline_neuronal.md`.
3. Resultados preliminares: 2x2, BE, HNM, CE, UMAP, calibración.
4. Discusión: silver standard, PRADO, limitaciones del BE/CE, alcance clínico, human-in-the-loop.
5. Marco teórico refinado con literatura.
6. Conclusiones finales cuando se cierren auditoría de métricas y ablación opcional.
