# Bitácora — entity-resolution-nlp

## Identidad del proyecto

| Campo | Valor |
|---|---|
| **Proyecto** | Tesis MCE — Record Linkage en bases de datos clínicas |
| **Título** | *Aprendizaje métrico y modelos de lenguaje pre-entrenados para la resolución de entidades en bases de datos clínicas* |
| **Institución** | CIMAT Unidad Monterrey |
| **Autor** | Uziel Isaí Luján López (Uzi) |
| **Director** | Dr. Víctor Hugo Muñiz Sánchez |
| **Co-director** | Dr. Víctor Mireles Chávez |
| **Datos** | INER — 3 bases de pacientes COVID-19 (Comorbilidad, Econo, TrabajoSocial), marzo 2020 – mayo 2023 |
| **Stack** | Python · PyTorch · HuggingFace Transformers · Sentence-Transformers · Micromamba · UV |


**Repositorio de consultoría:** `~/Projects/consultoria-iner/` contiene el pipeline de datos, la vinculación determinística y los entregables del INER. El repositorio es autocontenido y quedaron resueltos sus pendientes de reproducibilidad.

**Alcance de este repositorio:** `~/Projects/entity-resolution-nlp` contiene exclusivamente el sistema de record linkage retrieve and rerank: El entrenamiento del Bi-Encoder, el Hard Negative Mining como paso intermedio, el entrenamiento del Cross-Encoder, las evaluaciones de cada etapa, la calibración del umbral y la exportación de embeddings están implementados y evaluados. El foco inmediato es la limpieza de código, la documentación y la redacción del manuscrito de tesis. Hay una rama `feature/llm-chatbot-prototype` separada como futuro trabajo de AI Engineering para un proceso de inferencia asistida por inteligencia artificial.

## Contrato operativo

- El schema, las rutas, la propiedad de artefactos y los invariantes entre repositorios se definen en `docs/data_contract.md`.
- Los comandos y rutas vigentes están en `docs/comandos_proyecto.md`.
- La historia de caracterización, preprocesamiento, serialización, etiquetado, ground truth y entregables del INER vive en `consultoria-iner/MEMORY.md`.

---

## Estado actual

**Objetivo inmediato:** consolidar la redacción del manuscrito a partir de los artefactos canónicos ya evaluados, sin implementar todavía las mejoras propuestas.

**Estado del repositorio (2026-09-13):**
- El entorno operativo y reproducible es `uv`, usando `.venv/bin/python`.
- Las rutas, splitting, aumento de datos y descarga de modelos fueron auditados; la ruta canónica contiene BETO, RoBERTa-biomedical y paraphrase-multilingual.
- El baseline zero-shot oficial sobre `notok_skipnull/test` quedó reproducido para los tres modelos y se guarda en `modeling/outputs/evaluation/biencoder/zeroshot/zeroshot_results_notok_skipnull_test.json`.
- El visualizador UMAP canónico lee `modeling/embeddings/tok_skipnull/embeddings.parquet`; no recodifica registros. Sus 23,706 vectores de 768 dimensiones fueron exportados con `max_seq_length=512`.
- El foco vigente es consolidar la documentación metodológica y el manuscrito; la auditoría integral de código continúa pendiente de cierre.


**Estado del manuscrito:**
- Capítulos 1 (Introducción), 2 (Marco Teórico) y 4 (Metodología): primera versión redactada; requieren revisión y no se consideran definitivos.
- Capítulo 3 (Datos): actualmente el capítulo más completo.
- Capítulo 5 (Resultados y Discusión): redactado; incluye resultados del experimento de serialización, recuperación, Cross-Encoder, UMAP y calibración.
- Capítulo 6 (Conclusiones): redactado con contribuciones, límites, trabajo futuro y cierre; requiere revisión académica final.
- La bibliografía oficial se carga exclusivamente desde `manuscript/Bibliografia/referencias.bib`; el archivo histórico `Capitulos/referencias.tex` era obsoleto y se retiró.
- Pendiente relevante: revisar referencias bibliográficas cruzadas.
- La prioridad es cerrar una primera versión basada en el pipeline neuronal ya ejecutado. El baseline clásico end-to-end es un desarrollo tentativo y preventivo: no se incorporará todavía como resultado formal y permanecerá mencionado como trabajo futuro.

## Pendientes y decisiones

### Resultados de tesis

1. **Revisión de métricas cerrada** — Δseparabilidad usa todos los negativos disponibles por consulta. Los cuatro Bi-Encoders y los tres baselines zero-shot fueron reevaluados sobre test; `tok_skipnull` conserva la mayor separación promedio (11.2609).
2. **Auditoría de Hard Negative Mining** — Auditar el dataset y reconstruir con precisión el procedimiento histórico de HNM antes de modificarlo. En particular, verificar el efecto de seleccionar el Top-K global y filtrar después los registros de la misma fuente, y contrastarlo con el protocolo que invalida candidatos intra-fuente antes del Top-K.
3. **Decisión sobre HNM y K** — Después de la auditoría, decidir si se conserva o corrige el orden del filtrado y justificar `K=20` experimentalmente o documentarlo como una decisión no optimizada. Si se adopta el filtrado cross-source antes del Top-K, será necesario regenerar los pares HNM, reentrenar el Cross-Encoder y actualizar sus resultados.

### Manuscrito

1. **Metodología del Bi-Encoder** — Corregir en el Capítulo 4 la longitud máxima del checkpoint canónico: los cuatro modelos oficiales fueron entrenados con `max_seq_length=512`, no 384.
2. **Referencias** — Revisar las referencias bibliográficas y las referencias cruzadas del manuscrito.
3. **Revisión académica** — Realizar la revisión académica final de los seis capítulos.
4. **Front matter** — Completar los agradecimientos y actualizar la fecha de portada cuando corresponda.
5. **Alcance acordado** — Mantener la ablación, K-fold, ANN/FAISS, el baseline clásico formal y el reentrenamiento bidireccional del Cross-Encoder como trabajo futuro; no bloquean la primera versión neuronal del manuscrito.

### Código y documentación

1. **Auditoría integral post-separación** — Tras la separación de repositorios (tesis y consultoría), auditar de principio a fin el código completo del proyecto antes de reanudar el desarrollo de la tesis. Como parte de ese cierre, retirar los `sys.path.insert(...)` de los scripts: el README exigirá ejecutar `uv sync`, que deja el proyecto editable e importable.
2. **Bi-Encoder** — Revisar entrenamiento y evaluación completos cuando se retome el acceso GPU/HPC; la carga y la salida del checkpoint canónico ya fueron verificadas localmente.
3. **Exportación** — Verificar la exportación canónica de embeddings y sus consumidores.
4. **Reproducibilidad** — Revisar pruebas mínimas, semillas, scripts y wrappers SLURM sin acceder al cluster hasta acordarlo explícitamente.
5. **Documentación** — Actualizar progresivamente README y documentacion completa.

### Consulta con asesor

1. **[Prioridad alta] Incertidumbre del ligado** — Definir la estrategia de cuantificación de incertidumbre y su interpretación ya hay propuestas implenentadas en el código, pero no se han formalizado.

### Decisión documentada

1. **Umbral del Cross-Encoder** — Se fijó en 0.12 tras maximizar F1 sobre validación en una malla de 51 valores uniformes entre 0 y 1; alcanzó F1=1.0000. La evaluación de test usó ese valor sin reajuste. La calibración post-hoc no modifica el umbral.

## Resultados experimentales consolidados

- El experimento de serialización seleccionó `tok_skipnull` como variante canónica. Con todos los negativos de test, su Δseparabilidad promedio sobre las seis direcciones es 11.2609.
- En los candidatos evaluados, el Bi-Encoder alcanzó Recall@20 perfecto y Hard Negative Mining no encontró hard positives.
- El Cross-Encoder, con umbral 0.12 optimizado en validación, obtuvo F1=0.9997 antes de corregir un error de etiqueta; tras la corrección humana, se **lograría** obtener un F1=1.0.
- Temperature scaling no aportó información útil por la separación perfecta en validación. La incertidumbre queda acotada al Cross-Encoder y al supuesto de silver standard.
- La configuración del Bi-Encoder (`lr=2e-5`, `temp=0.07`) se heredó de los experimentos iniciales; no se reoptimizó sobre el ground truth oficial.
- La base Económico contiene desde el origen 80 registros con `EXP` nulo y uno con el valor literal `S/E`, ambos sin expediente utilizable para el ligado. Los 81 registros se conservaron en los Parquet preparados, el etiquetado, la serialización y el splitting como ruido natural de los datos; no fueron excluidos por esta condición.
- Los cuatro checkpoints oficiales con los nombres históricos `beto_mnrl_hpc_v2_*` fueron entrenados con `max_seq_length=512`, según sus `training_history.json` remotos y locales. Existió un run separado `beto_mnrl_hpc_v2_b72_L384_tok_skipnull` para extender al máximo el batch size pero usando `max_seq_length=384` y no superó al oficial. Los HTML UMAP históricos fueron generados imponiendo 384; se conservan como evidencia de esa configuración previa, pero la referencia canónica son las visualizaciones que leen `embeddings.parquet`, exportado con 512.

## Historial de fases

### [Inicio de 2026] Fase 1 — Arquitectura neuronal

- Se definió la arquitectura Retrieve & Rerank: Bi-Encoder siamés con MNRL para recuperación y Cross-Encoder con BCE para reclasificación.
- La representación adoptó Early Fusion con los bloques `[BLK_ID]`, `[BLK_CLIN]`, `[BLK_GEO]`, `[BLK_ADMIN]` y `[BLK_SOCIO]`.
- BETO y RoBERTa-bne fueron los candidatos iniciales en español.

---

### [Abril 2026] Fase 2 — Baselines zero-shot y diagnóstico de representación

- Se implementó el módulo Bi-Encoder que toma un backbone Transformer intercambiable y mean pooling, junto con una evaluación zero-shot cuya experimentación se corrigió.
- El baseline utilizó la variante `notok_skipnull/test`: serialización sin tokens especiales y omitiendo valores nulos.
- Modelos:
    - BETO (`dccuchile/bert-base-spanish-wwm-cased`),
    - RoBERTa-biomedical (`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`) y
    - paraphrase-multilingual (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`).
- Las métricas resultantes fueron bajas todos los modelos. Esto motivó avanzar hacia fine-tuning con MNRL.

**Sanity check:** paraphrase-multilingual obtuvo μ_para=0.803, μ_neg=0.147 y margen=0.657 sobre paráfrasis y negativos en español. Se confirmó que el modelo resolvía bien su tarea original.

**Experimento exploratorio con solo nombres (Hit@1):**

Se realizó un diagnóstico usando solo los nombres de los registros para determinar si la señal de identidad estaba presente en los embeddings preentrenados. Los detalles se encuentran en `docs/Anexos/narrativa_baselines_y_propuesta.md`.

---

### [Mayo 2026] Fase 3 — Entrenamiento inicial MNRL y aumento de datos

- Se validó localmente el entrenamiento end-to-end del Bi-Encoder con MNRL, incluyendo partición por entidad, warm initialization, learning rates diferenciales y precisión mixta.
- Los primeros runs con batch pequeño sobreajustaron: pocos negativos in-batch por ancla limitaban la capacidad de generalización. El entrenamiento en HPC con batches grandes resolvió esta limitación.
- Se añadieron visualizaciones diagnósticas de batches MNRL para inspeccionar la matriz de similitud durante el entrenamiento. BETO preentrenado no separaba identidades al inicio; tras una época, la diagonal de los pares positivos se volvió dominante dentro del batch.
- En los experimentos iniciales, los tokens especiales mejoraron la generalización respecto a la serialización sin tokens. Esta tendencia se confirmó con el ground truth oficial.
- Los experimentos de aumento de datos revelaron que los operadores implementados y sus probabilidades de activación no aproximan la distribución subyacente de los pares positivos reales entre bases de datos del INER. Al entrenar el Bi-Encoder con pares reales y sintéticos, la pérdida de entrenamiento disminuyó, pero la pérdida sobre el conjunto de validación, conformado unicamente por pares reales, aumentó.
- La exploración se detuvo en este punto. Aplicar operadores sintéticos también en el conjunto de validación y prueba cambiaría el objetivo; si los modelos lograran un buen desempeño implicaría que aprendieron el patrón artifical de los datos sintéticos y no el de los reales. La alternativa requeriría diseñar y calibrar finamente los operadores para reproducir la distribución real de los datos. Ambas vías exceden el alcance actual del proyecto.

---

### [Mayo 2026] Fase 4 — Experimentación HPC inicial del Bi-Encoder

- Se habilitó entrenamiento reproducible en el cluster con batch=64, checkpoints `best/` y early stopping.

| Run exploratorio | n_aug | train_loss | val_loss |
|---|---:|---:|---:|
| Baseline | 0 | 1.3138 | 1.1070 |
| Augmentación suave | 1 | 0.3070 | 1.3986 |
| Augmentación agresiva | 2 | 0.2174 | 1.5306 |

- El batch grande proporcionó más negativos in-batch para MNRL y permitió que generalizara. La tabla confirma la conclusión de la Fase 3: al aumentar la proporción de pares sintéticos, la pérdida de entrenamiento disminuyó mientras la pérdida de validación aumentó.
---

### [Mayo 2026] Fase 6 — Pipeline Retrieve & Rerank

- El experimento 2×2 de serialización (tokens × nulos) seleccionó **`tok_skipnull`** como variante oficial. La reevaluación con todos los negativos confirmó la mayor Δseparabilidad promedio (11.2609); Recall@K y MRR se saturaron en el test disponible.
- Hard Negative Mining con top-K=20 produjo 12,224 positivos y 98,027 hard negatives, sin hard positives en ningún split. El Bi-Encoder alcanzó Recall@20 perfecto sobre estos candidatos.
- El Cross-Encoder `beto_bce_hpc_v2_tok_skipnull`, con umbral 0.12 optimizado en validación, obtuvo en test F1=0.9997, precisión=1.0, recall=0.9995, 0 falsos positivos y 1 falso negativo sobre 16,350 pares.
- **Caso PRADO APOLINAR** (el único FN): `entity_id=3062`, exp 237574 — Económico/Armando (defunción 2020-04-21) vs Comorbilidad/Daniel (egreso 2020-05-23). Matemáticamente imposible mismo paciente: **era un error de etiquetado humano, no del modelo**. El CE lo detectó (score 0.0001) por divergencia del nombre. Corregido → **F1=1.0**. Insight: un modelo cercano al techo no sustituye la validación humana, la complementa.
- La proyección UMAP 3D aportó evidencia visual del aprendizaje métrico y cuatro comprobaciones confirmaron aislamiento por entidad sin data leakage.

---

### [Mayo 2026] Fase 7 — Calibración e incertidumbre

- Se implementó y ejecutó la calibración post-hoc y la cuantificación de incertidumbre por vínculo. Temperature scaling produjo $T=0.065256$ porque la validación era perfectamente separable; la entropía calibrada resultó poco informativa. El artefacto canónico conserva entropía cruda, entropía calibrada y margen al umbral.
- Nueve de 16,350 pares presentaron entropía cruda $\geq0.01$ bits. El caso PRADO evidenció que alta confianza no implica una etiqueta correcta: el modelo podía estar seguro y, a la vez, discrepar legítimamente del silver standard.
- Se confirmó con el asesor que el dataset es un **silver standard** y que la incertidumbre evaluada se limita al Cross-Encoder; matches no recuperados por el Bi-Encoder quedan fuera de esta auditoría.
- La ablación de bloques semánticos permanece como experimento de baja prioridad para medir su contribución mediante la caída de F1.
- Confident learning y una tercera etapa auditora con LLM se difirieron como trabajo futuro de ingeniería.

---

### [Junio 2026] Fase 8 — Integración semántica con datos preparados

- Los embeddings del Bi-Encoder se integraron como el score opcional `cos_biencoder`: compara el registro completo y complementa Jaro-Winkler y Levenshtein sin reemplazar automáticamente la revisión humana.

---

### [Mayo-Julio 2026] Fase 9 — Comunicación y manuscrito

- Se preparó y presentó satisfactoriamente ante el comité una exposición de avance que integró el pipeline neuronal, sus resultados y la visualización UMAP.
- Se creó `docs/contexto_manuscrito_tesis.md` para mapear el repositorio con los capítulos y orientar el orden de redacción.
- Se redactó una primera versión del Capítulo 3, que describe los datos preparados y conduce desde las limitaciones tabulares hacia los bloques semánticos del Capítulo 4.
- El manuscrito migró a clase `book`, incorporó la portada oficial del CIMAT y adoptó la estructura `frontmatter` / `mainmatter` / `backmatter`. Los capítulos 2–4 quedaron en una primera versión, todavía no definitiva.

---

### [Agosto 2026] Fase 10 — Separación definitiva del repositorio neuronal

- `entity-resolution-nlp` quedó dedicado exclusivamente al pipeline de deep learning y consume datos previamente preparados en `processed/default/output/<variant>/dataset.parquet`.
- `config.py` y los scripts adoptaron la variante por default `tok_skipnull`; los splits producidos por este repositorio se migraron posteriormente a `modeling/data/<variant>/`.
- El split canónico fue generado y verificado, y `visualize_embeddings.py` dejó de depender de CSVs limpios al extraer la identidad desde el texto serializado.
- Se retiraron del repositorio el procesamiento tabular, la construcción del ground truth y la documentación operativa correspondiente a esas etapas.

---

### [Agosto 2026] Fase 11 — Estabilización del workspace y validación de HNM

- Se formalizó en `docs/data_contract.md` la frontera con `consultoria-iner`: este repositorio consume `processed/default/output/<variant>/dataset.parquet` y concentra todos sus artefactos derivados bajo `INER_DATA_ROOT/modeling/`.
- Se reorganizaron las rutas en `config.py`: datos derivados, modelos Bi-Encoder/Cross-Encoder, embeddings, evaluaciones, figuras y diagnósticos.
- El entorno local quedó reproducible con `uv sync`.
- Se prepararon y validaron en `modeling/models/pretrained/` los tres modelos zero-shot reproducibles: BETO, RoBERTa-biomedical y paraphrase-multilingual, cada uno con los siete tokens estructurales registrados.
- Se regeneraron los splits de las cuatro variantes. Cada uno contiene 23,706 registros y 15,283 entidades, sin registros sin asignación ni entity leakage. Las asignaciones fueron idénticas entre variantes y `tok_skipnull` coincidió exactamente con el split histórico.
- El reporte de splitting distingue 10,678 entidades de una sola base de datos, 10,665 singletons y 13 entidades con varios registros y de una sola base de datos. Se corrigió la terminología: una entidad de una sola base puede tener múltiples registros; un singleton estrictamente tiene solo un registro. También reporta 11,447 pares positivos entre bases a nivel entidad y 12,224 a nivel registro.
- El checkpoint histórico del Bi-Encoder se copió, sin eliminar el original, a `modeling/models/biencoder/tok_skipnull/beto_mnrl/`. La copia se verificó por checksum, conservó `best/` y `training_history.json`, y cargó correctamente el mejor epoch 17; sus embeddings de prueba fueron finitos y normalizados L2.
- Se repitió Hard Negative Mining con `top_k=20` sobre el checkpoint canónico. Su pool contiene 12,224 pares positivos entre bases de datos a nivel registro: los 11,466 pares confirmados directamente durante el etiquetado y 758 pares adicionales inducidos por el cierre transitivo de `union-find` sobre los `entity_id` finales. El resultado fue 0 hard positives y 98,027 hard negatives: 77,635 pares en train, 16,266 en val y 16,350 en test.
- Los pares minados se validaron sin duplicados, pares intra-fuente, cruces de split, IDs inválidos ni inconsistencias entre `label` y `type`. No existía una copia histórica de los Parquet para comparación fila por fila, pero los conteos consolidados coincidieron con la bitácora previa.
- El baseline zero-shot oficial se ejecutó sobre `notok_skipnull/test` para los tres modelos preentrenados y las seis direcciones entre bases. Sus resultados viven en `modeling/outputs/evaluation/biencoder/zeroshot/zeroshot_results_notok_skipnull_test.json`; confirmaron desempeño bajo, con el mejor resultado de paraphrase-multilingual en Comorbilidad → Trabajo Social (Hit@1=0.0197, MRR=0.0702).
- `mine_hard_pairs.py` ahora usa `pairs_path(split, variant)` para escribir por defecto en `modeling/data/<variant>/`; `--output-dir` permanece como override experimental. El README conserva solo el contrato operativo y esta bitácora concentra los resultados de la validación.

---

### [Septiembre 2026] Fase 12 — Visualización canónica y resultados del manuscrito

- `visualize_embeddings.py` se estabilizó como consumidor de los embeddings canónicos exportados. Produce HTML interactivos en `modeling/outputs/figures/embeddings/` con nombres explícitos por split, filtro de entidades vinculables, tema y entidad destacada.
- Las vistas activas incluyen el espacio de prueba completo y de entidades vinculables en tema oscuro, una vista clara que destaca la entidad 471 y una proyección oscura del universo completo de 23,706 registros. Los HTML históricos se preservan en `modeling/outputs/figures/embeddings/old/` como evidencia de la configuración previa con 384 tokens.
- La figura usada en el póster, que destaca los tres registros cross-source de la entidad 471 (JUAN SANCHEZ LAGUNES), se incorporó al Capítulo 5. La discusión describe UMAP como una técnica de preservación de vecindades locales para diagnóstico visual, no como una métrica de desempeño ni evidencia suficiente de cobertura.
- La comparación de curvas de pérdida de validación de las cuatro variantes de serialización se incorporó al Capítulo 5. Las variantes que omiten nulos conservaron mejoras durante más épocas: `tok_skipnull` alcanzó su mejor checkpoint en la época 17 y ejecutó 20, mientras que `notok_skipnull` alcanzó su mínimo en la época 13 y ejecutó 16. Las variantes que retienen nulos tuvieron sus mejores checkpoints en las épocas 5 y 8 y finalizaron por early stopping en las épocas 8 y 11.
- El manuscrito se compiló correctamente en 113 páginas después de incorporar ambas figuras y la referencia de UMAP.

---

### [Septiembre 2026] Fase 13 — Baseline clásico y diagnóstico metodológico

- Se implementó un reranker clásico con TF-IDF de palabras y caracteres, características simétricas de pares y regresión logística. Sobre el HNM neuronal histórico obtuvo F1=0.9811, precisión=0.9848 y recall=0.9774; el Cross-Encoder conserva ventaja con F1=0.9997.
- Se implementó el pipeline clásico end-to-end independiente: ajuste TF-IDF solo en train, selección de representación y umbral en validación, minería Top-K propia, evaluación condicional y end-to-end, hashes de artefactos y validaciones contra entity leakage.
- Se ejecutaron las cuatro variantes. F1 de test: `tok_skipnull`=0.9699, `tok_keepnull`=0.9677, `notok_skipnull`=0.9893 y `notok_keepnull`=0.9847. Todas alcanzaron Recall@20=1.0000 y cero falsos negativos de recuperación.
- La comparación directa sobre `notok_skipnull` confirmó que TF-IDF combined (MRR=0.9995, Recall@20=1.0000, Delta=5.8485) supera ampliamente a BETO, RoBERTa-biomedical y paraphrase-multilingual zero-shot, cuyos MRR macro fueron 0.0184, 0.0183 y 0.0248.
- Los cosenos TF-IDF son menores que los neuronales afinados, pero discriminativos: `tok_skipnull` obtuvo medias 0.3463 en positivos y 0.1204 en negativos, con Delta=6.2120. BETO + MNRL obtuvo 0.9626, 0.0092 y Delta=11.2609. Los zero-shot asignaron cosenos altos a ambas clases y Delta cercano a 0.15.
- Se confirmó una circularidad metodológica central: nombre y expediente construyen principalmente el silver standard y permanecen en el texto usado por los modelos. No hay `entity_id` ni `record_id` serializados, pero TF-IDF puede reproducir casi directamente la regla de etiquetado.
- Se detectó una desalineación en HNM: el flujo neuronal histórico selecciona Top-K global y filtra misma fuente después; el clásico corregido invalida misma fuente antes de Top-K. La primera regla desperdicia slots y no es equivalente para entrenar o evaluar rerankers.
- No se encontró una justificación experimental formal para K=20. Queda propuesto un análisis de sensibilidad sobre K en validación y una ablación clásica mínima: completo, solo nombre-expediente, sin expediente, sin nombre y sin ambos.
- El resumen completo, limitaciones, artefactos y decisiones pendientes viven en `docs/Anexos/resultados_baseline_clasico_y_diagnostico_metodologico.md`. El manuscrito no fue modificado; primero se discutirán los resultados con el asesor.

---

## Trabajo futuro fuera del alcance actual

- **Robustez de HNM y recuperación:** evaluar un pool de candidatos que incluya entidades de una sola base, filtrar candidatos inválidos antes del Top-K, conservar y reportar la direccionalidad de recuperación, y añadir validaciones, procedencia y métricas de distribución de los artefactos. Detalle y priorización: `docs/Anexos/propuestas_robustez_hnm_kfold.md`.
- **Ablación clásica de identidad:** ejecutar primero el estudio mínimo con registro completo, solo nombre-expediente, sin expediente, sin nombre y sin ambos; después decidir si se justifica repetir la ablación con modelos neuronales. Detalle: `docs/Anexos/resultados_baseline_clasico_y_diagnostico_metodologico.md`.
- **Invariancia al orden y truncamiento del Cross-Encoder:** Se revisó que hay truncamiento de los registros en el Cross-Encoder `A | B`, la concatenación supera el límite de 512 tokens. La propuesta es reentrenar y evaluar cada par en ambas orientaciones (`A | B` y `B | A`), truncando un único registro por pasada para que se observe completo en una de ellos; combinar ambas puntuaciones y contrastar con lo reportado actualmente.
- **Auditoría de los casos difíciles del Bi-Encoder:** buscar aquellos registros que no se hayan etiquetado correctamente, aquellos con coseno alto o bajo y que difieran según su `entity_id`. Detalle: `docs/Anexos/propuesta_incertidumbre.md`.
- **Robustez de la estimación de generalización:** decidir si ejecutar validación cruzada, preferentemente tras auditar el flujo actual; un K-fold del pipeline completo requiere reentrenar el Bi-Encoder (varias veces) y regenerar HNM en cada fold (depende del encoding). Detalle: `docs/Anexos/propuestas_robustez_hnm_kfold.md`.
- **Inferencia e indexación:** Revisar propuestas como ANN, Exact MIPS, FAISS cuando se haya implementado/diseñado un sistema o flujo de inferencia completo, actualmente pertenece a una futura rama de ingeniería con aspectos como un chatbot LLM o incluso un proceso agéntico que asista en la revisión humana y recuperación de los candidatos. Detalle: `docs/Anexos/propuesta_inferencia/*.md`.
