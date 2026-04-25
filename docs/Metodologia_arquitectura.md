# Arquitectura Híbrida para Ligado de Registros: Enfoque de Recuperación y Re-clasificación Semántica (Retrieve & Rerank)

**Uziel Isaí Luján López**
## 1. Resumen de la Propuesta

Se propone una arquitectura de Ligado de Registros basada en el estado del arte del Procesamiento de Lenguaje Natural y Aprendizaje Profundo, diseñada para resolver el compromiso entre escalabilidad computacional y precisión semántica en grandes volúmenes de datos heterogéneos. El sistema supera las limitaciones de las métricas de similitud léxica estáticas basadas en comparación de cadenas de caracteres en favor de un enfoque que aprende representaciones semánticas a partir de la **serialización de registros completos** [1] mediante un **pipeline de dos etapas**:

1. **Retrieval: Generación de Candidatos.**

    La primera etapa utiliza una arquitectura de **Bi-Encoder Siamesa (SBERT)** para procesar todos los registros y proyectarlos a un espacio vectorial denso, permitiendo la recuperación eficiente de candidatos de alta probabilidad mediante búsqueda de similitud coseno, reduciendo drásticamente el espacio de búsqueda de $O(N^2)$ a una complejidad lineal [2].

2. **Re-ranking: Clasificación Fina.**

    Los candidatos recuperados en la etapa anterior son procesados por un **Cross-Encoder basado en Transformers (DITTO)**, que aplica mecanismos de Auto-Atención sobre la serialización completa de los pares de registros para **capturar matices sintácticos y semánticos** [1].

Este enfoque permite integrar técnicas para manejar secuencias largas de texto sin saturar la ventana de contexto del modelo, estrategias de entrenamiento que superan la eficiencia de las funciones de perdida tradicionales [3] y técnicas de aumentación de datos para aprender representaciones robustas sin depender de procesos de etiquetado manual, gestionar la heterogeneidad y los datos faltantes.

## 2. Representación de las Entradas: Serialización Semántica (Input Layer)
En contraposición a los enfoques tradicionales de Record Linkage que requieren pipelines de vectorización heterogéneos según el tipo de dato (e.g., One-Hot Encoding para variables categóricas, normalización para numéricas o codificación fonética aislada), esta propuesta implementa una estrategia de Fusión Temprana (Early Fusion) que transforma cada registro tabular $R$ en una secuencia de texto unificada $S$. Esta técnica de **serialización**, fundamentada en la metodología DITTO [1], permite que el modelo capture dependencias semánticas latentes y el contexto global del registro sin la rigidez de una alineación estricta por columnas.

### 2.1. Estrategia de Serialización
Para preservar la estructura del esquema original dentro de una secuencia lineal, se emplean tokens especiales que delimitan tanto los atributos individuales como los Bloques Semánticos definidos (`Identidad Demográfica`, `Clínico`, `Geográfico`, `Administrativo`, `Socioeconómico`), concatenándolos en una única cadena de texto.

**Formato General de la Secuencia Serializada:**

$$S_i = \text{[BLK\_NAME]} \ \text{[COL]} \ Atributo_j \ \text{[VAL]} \ Valor_j \ \dots$$

Donde:
* **`[BLK_NAME]`**: Token de control propuesto (e.g., `[BLK_ID]`, `[BLK_CLIN]`) que agrupa los atributos de un mismo bloque semántico. Esto permite al modelo aprender la jerarquía y el contexto de los atributos relacionados, facilitando la atención contextualizada dentro de cada bloque.

* **`[COL]` y `[VAL]`**: Tokens especiales, adoptados de la arquitectura DITTO [1], que indican explícitamente el inicio del nombre del atributo y su valor respectivo, enseñando al modelo la estructura relacional clave-valor.

**Justificación Arquitectónica de los Tokens de Bloque:**

A diferencia de la propuesta original de DITTO [1], que aplana los registros en un solo nivel, la inclusión de `[BLK]` extiende la serialización a un esquema jerárquico adaptado a la complejidad de los datos del INER. A nivel de implementación, esta extensión es un procedimiento estándar: se añaden los nuevos tokens al vocabulario (`tokenizer.add_special_tokens()`) y se ajusta la capa de entrada del modelo neuronal (`model.resize_token_embeddings()`). Estos nuevos vectores se inicializan aleatoriamente y adquieren su representación semántica óptima mediante la actualización de pesos por retropropagación durante la fase de *Fine-Tuning*.


**Manejo de Datos Faltantes**

La arquitectura es inherentemente robusta a la dispersión de datos (sparsity). A diferencia de los métodos clásicos que requieren imputación (medias, modas), este enfoque aprovecha el **mecanismo de Auto-Atención (Self-Attention) de los Transformers**.

* **Ausencia de Datos:** Si un atributo es `NULL`, se omite o se representa con un token vacío `[VAL_NULL]` en la secuencia serializada.

* **Mecanismo:** El modelo aprende a redistribuir los pesos de atención hacia los tokens presentes, ignorando dinámicamente la ausencia de información sin introducir ruido numérico artificial en la representación vectorial.


**Casos de Uso y Manejo de Ausencias:**

* **Ejemplo Estándar:**
    ```text
    [BLK_ID] [COL] Nombre [VAL] Juan Pérez [COL] Edad [VAL] 45 [BLK_CLIN] [COL] Diagnóstico [VAL] Diabetes Tipo 2 ...
    ```
    *Justificación:* Este diseño maximiza la eficiencia en el uso de la ventana de contexto al condensar la jerarquía en tokens individuales.

* **Ejemplo con datos faltantes a nivel de atributo:**
    ```text
    [BLK_ID] [COL] Nombre [VAL] María López [COL] Edad [VAL_NULL] [BLK_CLIN] [COL] Diagnóstico [VAL] Hipertensión ...
    ```
    *Justificación:* El token `[VAL_NULL]` señala explícitamente la ausencia de información en una columna esperada. Esto instruye al modelo a redistribuir su atención sin introducir ruido numérico ni sesgos de imputación.

* **Ejemplo con datos faltantes a nivel de bloque completo:**
    ```text
    [BLK_ID] [COL] Nombre [VAL] Carlos Sánchez [COL] Edad [VAL] 38 [BLK_GEO] [COL] Ciudad [VAL] Ciudad de México ...
    ```
    *Justificación:* Si un bloque entero (en este caso, el Clínico) carece de datos, **se omite por completo de la secuencia**. El mecanismo de atención del modelo se adapta dinámicamente, saltando del *bloque identidad demográfica* al *bloque geográfico* sin desperdiciar tokens de procesamiento.

### 2.2 Estructura de Datos Serializados y Gestión de Etiquetas

Para la transición entre la fase de limpieza de datos crudos y la ingesta en el pipeline del modelo (PyTorch/Hugging Face), el sistema abandona el formato tabular tradicional CSV en favor del estándar Apache Parquet (`.parquet`). Esta decisión arquitectónica se fundamenta en su compresión columnar y su capacidad de *memory-mapping*. Esto previene la saturación de los 8GB de memoria RAM disponibles en el entorno local, permitiendo que la GPU se dedique exclusivamente al procesamiento de tensores durante el entrenamiento.

Para resolver la explosión combinatoria $\mathcal{O}(N^2)$ inherente a la vinculación de registros a nivel de almacenamiento, el archivo procesado no guarda pares de comparación explícitos. En su lugar, emplea un enfoque de asignación de clústeres mediante una llave maestra unificadora.

**Esquema de Datos Serializados (Formato Parquet):**

| Columna | Tipo de Dato | Descripción Funcional |
| :--- | :--- | :--- |
| `record_id` | String | Identificador primario único de la observación en su base de origen (ej. `Econo_1045`). |
| `source_db` | String | Base de datos de procedencia (`comorbilidades`, `economica`, `tsocial`). |
| `text` | String | Secuencia de texto crudo serializado con tokens especiales, sin saltos de línea (ej. `[BLK_ID] Nombre...`). |
| `entity_id` | String | **Etiqueta Maestra.** Identificador global único que agrupa a la misma entidad (paciente real) a través de las bases. |

**Lógica de Agrupamiento para el Entrenamiento (MNRL):**

Bajo este esquema de unificación, el archivo final consolida exactamente el total nominal de registros ($N$). Durante la construcción dinámica de los lotes (*batches*) en la fase de entrenamiento, cualquier subconjunto de filas que comparta el mismo `entity_id` se extrae automáticamente como un conjunto de **Pares Positivos**. Por consiguiente, cualquier cruce entre filas con distinto `entity_id` dentro del mismo lote se procesa algorítmicamente como un **Negativo Implícito** para la función de pérdida MNRL. Esta estructura elimina la necesidad de pre-computar y almacenar matrices dispersas gigantescas.


### 2.3. Inyección de Conocimiento de Dominio (Domain Knowledge Injection)
Para mitigar la ambigüedad en campos de **alta precisión sintáctica** donde la semántica general no es un criterio discriminativo suficiente (como identificadores únicos o fechas), se implementa la estrategia de **Tipificación de Segmentos (*Span Typing*)** propuesta por Li et al. [1].

Esta técnica consiste en insertar etiquetas especiales de apertura y cierre alrededor de segmentos críticos. El objetivo es forzar al mecanismo de atención a enfocarse en la **coincidencia exacta** de los caracteres delimitados, penalizando severamente las discrepancias superficiales.

* **Ejemplo de Tipificación:**
    ```text
    ... [COL] CURP [VAL] <UID> JUPZ900101 </UID> [BLK_CLIN] [COL] Ingreso [VAL] <DATE> 2026-02-14 </DATE> ...
    ```
    *Justificación:* Estas envolturas `<UID> </UID>`, `<DATE> </DATE>` actúan como *imanes de atención* estructurales. Ayudan al modelo a distinguir diferencias de un solo carácter (e.g., variaciones mínimas en folios o los últimos dígitos de un teléfono) que un modelo de lenguaje estándar podría suavizar o ignorar en el espacio vectorial [1].

Complementariamente, se aplica la estrategia de **Normalización de Segmentos (*Span Normalization*)**, la cual estandariza tramos de texto con alta variabilidad sintáctica pero equivalencia semántica estricta. Al llevar estos segmentos a un formato canónico, el modelo concentra su capacidad de aprendizaje en comparar la información real y no la variabilidad de su representación superficial [1].

* **Ejemplos de Normalización:**
    * **Numéricos:** Cadenas como `"5 %"` y `"5.00 %"` se normalizan al formato `"5.0%"`.
    * **Fechas:** Variaciones de captura humana o formatos como `"14/02/2026"` y `"14 Feb 2026"` se estandarizan al formato ISO `"2026-02-14"`.
    Esto previene falsos negativos derivados exclusivamente de inconsistencias de formato durante la captura manual en el entorno clínico [1].


### 2.4. Longitud de la Secuencia de Entrada: Resumen Estadístico Basado en TF-IDF

Dado que los modelos basados en BERT imponen una restricción de entrada de 512 tokens (sub-words), el truncamiento ingenuo de registros clínicos extensos podría resultar en la pérdida de información crítica situada al final del texto.

Para resolver esto, se aplica una técnica de Resumen Estadístico previo a la entrada al modelo [1]:

1. Se calculan los puntajes TF-IDF de los tokens en el corpus de los registros para identificar los términos más informativos y relevantes.

2. Se retienen obligatoriamente los tokens especiales de
estructura (`[BLK]`, `[COL]`, `[VAL]`, `<UID>`, `</UID>`, `<DATE>`, `</DATE>`).

3. Del texto libre restante, se seleccionan los tokens con mayor peso TF-IDF hasta completar la ventana de contexto.

Esto garantiza que términos poco frecuentes pero relevantes se preserven sobre palabras comunes (*stop words*) , maximizando la cantidad de **tokens informativos** que alimentan al modelo [1].

> Nota: Es importante destacar una restricción arquitectónica: dado que la Etapa 2 del Cross-Encoder (explicado más adelante) procesa pares de registros concatenados (`[CLS] S_a [SEP] S_b [SEP]`), la ventana máxima de 512 tokens del Transformer se **comparte entre ambos**. Por lo tanto, el algoritmo de resumen TF-IDF se parametriza para restringir cada secuencia individual a un máximo de **256 tokens**, garantizando que el par unificado nunca exceda el límite de memoria del modelo


## 3. Arquitectura del Sistema: Pipeline Híbrido de Recuperación y Re-clasificación

Dado el alto costo computacional de los modelos de Atención Cruzada (Cross-Encoders) como DITTO [1], cuya complejidad es cuadrática en función del número de pares a comparar, resulta inviable evaluar el producto cartesiano completo de dos bases de datos grandes ($|D_A| \times |D_B|$). Para resolver esto, se diseña un **Pipeline Híbrido de Dos Etapas (Retrieve & Rerank)**.

Esta arquitectura implementa un enfoque de ***embudo*** donde:

1. Una etapa de **Recuperación (Retrieval)** de alta velocidad y cobertura (*High Recall*) que filtra el espacio de búsqueda.
2. Una etapa de **Re-clasificación (Rerank)** de alta precisión (**High Precision**) que evalúa únicamente a los candidatos prometedores.

Este diseño se alinea con estrategias validadas en el estado del arte, permitiendo aprovechar la comprensión semántica profunda de los Transformers en un entorno de producción escalable [2].


### 3.1. Etapa 1: Generación de Candidatos (The Retrieval Layer)
El objetivo de esta etapa es proyectar los registros serializados a un espacio vectorial denso donde la proximidad geométrica refleje similitud semántica. Para ello, se utiliza la arquitectura **Sentence-BERT (SBERT)**.

#### 3.1.1. Arquitectura Siamesa (Bi-Encoder)
A diferencia de BERT estándar, que requiere que ambos registros se procesen simultáneamente (***Cross-Encoder***), SBERT emplea una red siamesa (***Siamese Network***). Esta configuración consta de dos redes BERT idénticas que comparten sus parámetros (pesos atados o tied weights).

Formalmente, sea $S_a$ un registro serializado (y resumido vía TF-IDF según la Sección 2.3). La red procesa el registro de forma independiente para generar una representación contextualizada de cada token.

#### 3.1.2. Estrategia de Pooling y Generación de Embeddings
Para derivar un vector de tamaño fijo $u \in \mathbb{R}^d$ (donde $d=768$ típicamente) a partir de la secuencia de tokens de longitud variable, SBERT introduce una operación de **Pooling** en la capa de salida.

Aunque existen estrategias como usar el token [CLS] o Max-Pooling, la literatura de SBERT [2] demuestra que la estrategia de **Mean-Pooling** (promedio de todos los vectores de salida de los tokens) genera las representaciones semánticas más robustas para tareas de similitud textual.

$$u = \text{MeanPooling}(\text{BERT}(S_a))$$

#### 3.1.3. Recuperación por Similitud Coseno
Una vez que todos los registros han sido codificados en vectores ($u$ para la base A, $v$ para la base B), la similitud entre dos registros se calcula mediante la Similitud Coseno:

$$\text{sim}(S_a, S_b) = \cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

Esta formulación permite transformar el problema de Linkage en una búsqueda de vecinos cercanos (Nearest Neighbor Search) en el espacio vectorial.

#### 3.1.4. Justificación de Eficiencia Computacional
El uso de SBERT es imperativo para la escalabilidad. Mientras que una arquitectura Cross-Encoder requeriría re-evaluar la red neuronal para cada par posible, la arquitectura Bi-Encoder permite pre-calcular los embeddings.

Reimers et al. (2019) demuestran que para encontrar los pares más similares en una colección de 10,000 elementos, BERT estándar (Cross-Encoder) requiere aproximadamente 65 horas de cómputo ($~50$ millones de inferencias), mientras que SBERT reduce este tiempo a aproximadamente **5 segundos**. Esto representa una reducción crítica de la complejidad computacional, pasando de $O(N^2)$ inferencias de red profunda a una búsqueda indexada eficiente [2].

**Nota:** Es importante aclarar que, aunque la literatura distingue entre estructuras "Siamesas" (dos ramas, para pares) y "Tripletes" (tres ramas, para ancla-positivo-negativo) según la estrategia de entrenamiento utilizada, ambas se fundamentan en el mismo principio arquitectónico de ***Compartir Parámetros***. Para efectos de inferencia y recuperación, el modelo opera como un Bi-Encoder unificado que genera representaciones vectoriales independientes para cada registro [2].


### 3.2. Etapa 2: Clasificación de Alta Precisión (The Matching Layer)

Los candidatos recuperados en la etapa anterior (que representan una fracción minúscula del espacio total, pero con alta probabilidad de coincidencia) son sometidos a un proceso de verificación exhaustiva mediante un modelo ***Cross-Encoder***.
Para esta etapa, se adopta la arquitectura **DITTO (Deep Entity Matching with Pre-Trained Language Models)** [1].

#### 3.2.1. Arquitectura Cross-Encoder con Atención Cruzada
A diferencia del Bi-Encoder de la etapa previa, que procesa los registros de forma aislada, el Cross-Encoder procesa el par de registros $(S_a, S_b)$ como una **única secuencia unificada**.

$$S_{pair} = \text{[CLS]} \ S_a \ \text{[SEP]} \ S_b \ \text{[SEP]}$$

Esta formulación permite que el mecanismo de **Auto-Atención (Self-Attention)** del Transformer opere sobre la concatenación de ambos registros. Esto habilita una **Atención Cruzada (Cross-Attention)** completa, donde cada token de $S_a$ puede "atender" e interactuar directamente con cada token de $S_b$ en todas las capas de la red, capturando relaciones no lineales imposibles de ver en una simple comparación de vectores [1].

#### 3.2.2. Ventaja sobre la Arquitectura Bi-Encoder
La principal limitación de la etapa 1 (Bi-Encoder) es que comprime toda la información de un registro en un único vector fijo. Esto puede generar **Falsos Positivos** en registros que tienen una alta coincidencia de palabras pero difieren en un detalle crítico.

* Ejemplo Intuitivo: Dos registros clínicos casi idénticos:

    * Registro A: `"Paciente Juan Pérez, Diagnóstico: Diabetes Tipo 1"`

    * Registro B: `"Paciente Juan Pérez, Diagnóstico: Diabetes Tipo 2"`

Un Bi-Encoder proyectaría ambos registros en vectores extremadamente cercanos (alta similitud coseno) debido a la gran superposición de palabras, sugiriendo erróneamente un match.

En contraste, DITTO (Cross-Encoder) analiza la **secuencia conjunta** y detecta que el token "1" y el token "2" están en conflicto directo dentro del mismo contexto, permitiendo clasificar correctamente los registros como **entidades distintas**.

Esta capacidad de discriminación fina es reportada por Li et al. [1], quienes demuestran en su Caso de Estudio que, mientras un modelo SBERT alcanza un F1-score del 92% (útil para filtrado rápido), no logra capturar las sutilezas necesarias para la decisión final. DITTO, al procesar la interacción completa, eleva el rendimiento hasta un **96.5% de F1**, justificando su rol como el **juez final** de la arquitectura [1].

#### 3.2.3. Clasificación final y Penalización de Discrepancias Sintácticas
La salida del token especial `[CLS]`, que ahora resume la relación semántica profunda entre el par de registros, se alimenta a una capa lineal final (*Fully Connected Layer*) seguida de una función Softmax para predecir la probabilidad de que los registros refieran a la misma entidad real (Clase 1: *Match*, Clase 0: *No-Match*).

Adicionalmente, esta etapa integra la **Inyección de Conocimiento de Dominio** (descrita en la Sección 2.2). Al envolver tokens críticos (como fechas o folios) con etiquetas especiales, se fuerza al mecanismo de atención a penalizar fuertemente las discrepancias sintácticas dentro de esos tramos, robusteciendo la decisión final ante errores tipográficos o variaciones menores [1].


### 3.3. Adecuación de la Arquitectura al Idioma Español

La implementación directa de arquitecturas de *Metric Learning* (SBERT) y *Cross-Attention* (DITTO) utilizando sus modelos base originales en inglés (e.g., `bert-base-uncased`) presenta una limitación crítica para el contexto de esta investigación: la fragmentación excesiva del vocabulario.

Al procesar expedientes de pacientes en México, un tokenizador optimizado para el inglés o uno multilingüe genérico destrozaría apellidos locales o nombres de ubicaciones mexicanas específicas en múltiples sub-tokens sin sentido semántico. Esto no solo destruye la representación del concepto, sino que satura rápidamente la restricción de la ventana de contexto de 256 tokens por registro (en DITTO), reduciendo drásticamente la cantidad de información útil que el modelo puede procesar.

Para resolver esto, el presente trabajo propone sustituir el codificador base por Modelos de Lenguaje Pre-entrenados en Español. Para garantizar la representación más robusta, se plantea una evaluación comparativa con los siguientes modelos:

1.  **BETO:** Desarrollado por la Universidad de Chile, basado en la arquitectura clásica de BERT y pre-entrenado con un corpus masivo que incluye la Wikipedia en español y noticias.
2.  **RoBERTa-bne (Proyecto MarIA):** Desarrollado por el Barcelona Supercomputing Center (BSC). Al estar basado en RoBERTa y entrenado con el vasto Archivo de la Web Española (BNE), elimina el objetivo de predicción de siguiente oración (NSP) y optimiza dinámicamente el enmascaramiento, reportando sistemáticamente un rendimiento superior a BETO en múltiples tareas de comprensión del lenguaje natural (NLU).


**Exploración del Estado del Arte (SOTA).**
Adicionalmente, sujeto a compatibilidad arquitectónica y restricciones de tiempo y cómputo, se contempla la evaluación de **RigoBERTa** [Vaca Serrano et al.]. Este modelo monolingüe en español está basado en la arquitectura DeBERTa, la cual introduce un mecanismo de "atención desenredada" (*disentangled attention*) que codifica el contenido y la posición relativa de forma separada. Aunque promete resultados superiores en tareas de alineamiento semántico, su integración con el ecosistema estándar de *Sentence-Transformers* requiere adaptaciones técnicas más profundas.

**Manejo Inherente de Variables Numéricas y Fechas.**
Finalmente, es importante destacar cómo estos modelos base abordan campos de naturaleza no estrictamente alfabética. Gracias a que sus vocabularios se construyen mediante tokenización por sub-palabras (BPE o WordPiece) aplicada sobre corpus a escala de internet, procesan de forma natural los dígitos universales y formatos de fecha estándar (ej. `DD/MM/YYYY`). El modelo comprende el contexto numérico (sabe que un token como "1990" suele acompañar a conceptos temporales).

Sin embargo, debido a que el pre-entrenamiento estadístico tiende a agrupar números similares en el espacio vectorial (suavizando la diferencia entre el folio "505" y el "506"), la precisión estricta a nivel de carácter que demanda el *Record Linkage* no recae en este pre-entrenamiento base, sino que se garantiza algorítmicamente mediante la **Inyección de Conocimiento** (etiquetas `<UID>`) detallada en la Sección 2.2.

## 4. Estrategia de Entrenamiento: Aprendizaje Auto-Supervisado y Supervisión Débil

Dado que los conjuntos de datos clínicos del INER carecen de un *Ground Truth* (pares de registros previamente etiquetados como "iguales" o "diferentes"), este proyecto propone una estrategia híbrida que combina **Supervisión Débil (Weak Supervision)** para la extracción de una verdad base inicial, seguida de **Aprendizaje Auto-Supervisado** mediante **aumentación de datos**.

El objetivo fundamental es aprender una **función de similitud robusta** en un espacio métrico compartido. El modelo debe aprender a estructurar el espacio vectorial de tal forma que las representaciones de una **misma entidad** (independientemente de sus variaciones sintácticas, errores o datos faltantes) estén cerca geométricamente, mientras que las de entidades distintas se alejen significativamente, maximizando el margen de decisión.

### 4.1. Fase 0: Detección Zero-Shot y Extracción de la Verdad Base
Para evitar el colapso de la función de pérdida durante el entrenamiento, al asumir erróneamente que dos registros de la misma entidad en un lote son entidades distintas, se implementa una fase preliminar de identificación de pares positivos reales.

La extracción de esta *Verdad Base* se realiza mediante un enfoque dual de filtrado, inspirado en las metodologías de emparejamiento tradicionales y semánticas [1].

#### 4.1.1. Extracción Básica: Reglas Deterministas y Similitud Estadística
Este primer filtro se apoya en heurísticas de **muy alta precisión pero bajo alcance (*Recall*)**, enfocadas en la coincidencia exacta o cuasi-exacta de atributos clave aislados. Las estrategias incluyen:
* **Reglas Lógicas Exactas:** Emparejamiento determinista de registros que comparten identificadores únicos absolutos (mismo expediente, mismo nombre) o combinaciones altamente restrictivas (coincidencia exacta simultánea).
* **Similitud Léxica (TF-IDF):** Cálculo de la similitud coseno basada en vectores de frecuencias (TF-IDF) sobre atributos de alta varianza (como direcciones o nombres completos), reteniendo únicamente los pares que superan un umbral de coincidencia léxica.

#### 4.1.2. Extracción Avanzada: Recuperación Semántica Zero-Shot
Dado que la Extracción Básica fracasa ante errores tipográficos severos, valores nulos o variaciones de formato, se implementa una capa secundaria de detección semántica sobre los registros completos.
* Se utiliza un modelo Bi-Encoder (SBERT) con pesos pre-entrenados en **tareas de similitud textual multilingüe** operando en un entorno *Zero-Shot*.
* El modelo procesa la secuencia serializada de los registros que no fueron emparejados en el paso anterior y calcula su similitud coseno en el espacio vectorial.
* Los pares que superan un umbral de confianza estadísticamente estricto son considerados como **candidatos positivos reales**.


**Consolidación de la Verdad Base**

* **Objetivo:** La unión de los pares obtenidos por ambos métodos de extracción confirmados posteriormente por **revisión humana experta** constituye el subconjunto de **Positivos Reales Identificados** (Verdad Base).

* **Supuesto Fuerte:** Dado que se espera que la fracción de positivos reales detectados sea estadísticamente pequeña, se asume operativamente que los registros no identificados en esta Fase 0 no tienen pares idénticos naturales en el resto de las bases de datos, lo cual permite tratarlos matemáticamente como **Negativos Implícitos** en el entrenamiento posterior.


### 4.2. Estrategia de Partición a Nivel de Entidad (Data Splitting)
Para garantizar la validez de las métricas de evaluación y evitar la fuga de datos (*Data Leakage*), el corpus se debe dividir **antes** de aplicar cualquier técnica de aumentación de datos.

Tanto el subconjunto de los Positivos Reales Identificados en la Fase 0 como el resto de los registros no identificados se particionan en conjuntos de Entrenamiento, Validación y Prueba.

* **Regla estricta:** Todos los registros pertenecientes al mismo clúster de una entidad real deben quedar asignados al mismo conjunto (Entrenamiento, Validación o Prueba). Si el registro, digamos, $A_{15}$ y el registro $B_{22}$ son identificados como la misma entidad, ambos deben estar en la misma partición, garantizando que el modelo jamás se evalúe con una entidad que ya vio durante el entrenamiento.

### 4.3. Generación de Pares Positivos Sintéticos (Data Augmentation)

Dado que el volumen de Positivos Reales Identificados en la Fase 0 será probablemente insuficiente para entrenar o evaluar métricas con significancia estadística, se aplican transformaciones estocásticas para generar Versiones Aumentadas ($R_{aug}$) a partir de un Registro de Referencia ($R_{ref}$), de esta forma generamos Pares Positivos Sintéticos que mitigan la desproporción entre pares positivos y negativos reales y enriquecen el espacio de entrenamiento y evaluación con variaciones realistas que el modelo debe aprender a reconocer como la misma entidad robusteciendo su capacidad de generalización.

Para preservar la pureza de la evaluación, la aumentación de datos se aplica de manera **independiente y aislada** a los registros no identificados dentro de cada una de las tres particiones (Train, Val y Test).

* **En el Conjunto de Entrenamiento:** Se genera un volumen masivo de pares sintéticos para que el modelo aprenda a ser invariante al ruido y a las discrepancias de captura.

* **En los Conjuntos de Validación y Prueba:** Se genera una cantidad controlada de pares sintéticos. Esto dota a las fases de evaluación de una **Estructura Híbrida**, evaluando al modelo tanto en los pocos positivos naturales (Verdad de Oro) como en un volumen estadísticamente significativo de positivos aumentados (Verdad de Plata) que no intervinieron en el entrenamiento.

Matemáticamente, buscamos que la función de similitud aprenda a mapear estas transformaciones estocásticas a un espacio donde los pares aumentados se mantengan cercanos a su referencia original:
$$sim(f_\theta(R_{ref}), f_\theta(R_{aug})) \approx 1$$

Esta metodología, adaptada de Li et al. [1] aplica las siguientes operaciones de aumentación:

1.  **Eliminación de Tokens (Span Deletion):** Se eliminan secuencias aleatorias de tokens con probabilidad $p$. Esto fuerza a la red a no sobre-depender de palabras específicas y a utilizar el contexto global.

2.  **Intercambio de Bloques (Block Shuffling):** Se altera el orden de los atributos serializados. Evita que el modelo memorice posiciones fijas ya sea a nivel de bloques semánticos o de atributos individuales, promoviendo la comprensión de la estructura semántica más que la sintáctica.

3.  **Inyección de Ruido Tipográfico:** Simulación de errores de tecleo (*typos*) para robustecer el modelo ante la "suciedad" de los datos reales.

4.  **Enmascaramiento de Atributos (Attribute Masking):** Se simula la ausencia de datos reemplazando el valor de un campo completo por un token de vacío `[VAL_NULL]` o eliminándolo de la secuencia serializada.
    * *Objetivo:* Aunque el análisis exploratorio de los datos mostró una alta presencia de datos faltantes, es crucial que el modelo aprenda a manejar esta situación, pues podría aprender a depender de campos que casi siempre están completos (sesgo de dependencia).
    * *Justificación:* Al forzar artificialmente la ausencia de información en ciertos atributos [1], obligamos a la red a redistribuir su atención hacia los tokens presentes y aprender a identificar a la entidad real **usando cualquier combinación de atributos disponibles**, mejorando así su capacidad de generalización en escenarios reales donde la incompletitud es común.

5.  **Intercambio de Entradas (Input Swapping):** Específicamente para el entrenamiento del Cross-Encoder (Etapa 2, detallada más adelante), se invierte el orden de las secuencias al concatenarlas en la entrada, procesando [$R_{ref} | R_{aug}$] y luego [$R_{aug} | R_{ref}$] para tomar su decisión, promoviendo la simetría en la función de similitud aprendida.

    $$sim((R_{ref}, R_{aug})) = sim((R_{aug}, R_{ref}))$$

    Esto garantiza que el mecanismo de atención no desarrolle sesgos posicionales, forzando matemáticamente a que la probabilidad de vinculación sea simétrica e independiente de qué registro se presenta primero:
    $$P(\text{Match} | R_{ref}, R_{aug}) = P(\text{Match} | R_{aug}, R_{ref})$$


> **Nota sobre el Enmascaramiento:** No hay que confundir la técnica de *Masked Language Modeling* (MLM) usado en el pre-entrenamiento de los Transformers para *estimar* tokens ocultos, con la fase de *Fine-Tuning* donde el enmascaramiento de atributos actúa puramente como **regularización**. En ese sentido, el modelo no intenta estimar el dato faltante, sino identificar la entidad real usando exclusivamente la evidencia disponible.


### 4.4. Entrenamiento de la Etapa 1: Selección de la Función de Pérdida del Bi-Encoder

Con los pares positivos generados (naturales y sintéticos), se entrena a Sentence-BERT utilizando la función **Multiple Negatives Ranking Loss (MNRL)** [3]. A continuación, se justifica la selección de esta función de pérdida a través de la evolución del estado del arte.


#### 4.4.1. Enfoques Clásicos: Clasificación, Contrastive y Triplet Loss
Históricamente, el aprendizaje de similitud semántica ha evolucionado desde enfoques de clasificación pura hasta funciones de pérdida diseñadas específicamente para espacios métricos.

**1. Cross-Entropy Loss (Enfoque de Clasificación)**
El enfoque más directo, utilizado en arquitecturas como DeepMatcher o la configuración de clasificación de SBERT, trata el problema como una clasificación binaria o multiclase. Se concatenan las representaciones de los registros $(u, v)$ junto con su diferencia $|u-v|$ y se pasan por una capa Softmax.

El objetivo es minimizar la **Entropía Cruzada ($N$ clases)**:
$$\mathcal{L}_{CE} = - \sum_{c=1}^{M} y_{o,c} \log(p_{o,c})$$
Donde $M$ es el número de clases, $y$ es el indicador binario (0 o 1) de si la etiqueta $c$ es correcta para la observación $o$, y $p$ es la probabilidad predicha. Para el caso binario (*Match/No-Match*), esto se simplifica a la **Binary Cross-Entropy**:
$$\mathcal{L}_{BCE} = - \left( y \cdot \log(p) + (1 - y) \cdot \log(1 - p) \right)$$

Aunque efectiva para precisión, esta función no optimiza directamente la distancia euclidiana necesaria para búsquedas rápidas.

**2. Contrastive Loss (Pérdida Contrastiva)**
Este enfoque, pionero en *Metric Learning*, toma pares de registros y una etiqueta binaria $y$ (0 si son similares, 1 si son disímiles). El objetivo es minimizar la distancia $D$ entre pares similares y maximizarla para los disímiles, siempre que esta distancia sea menor a un margen $m$.

$$\mathcal{L}_{Contr} = (1-y)\frac{1}{2}D^2 + (y)\frac{1}{2}\{\max(0, m - D)\}^2$$

Sin embargo, su limitación radica en que analiza cada par de forma aislada, sin considerar el contexto relativo de otros ejemplos negativos en el espacio vectorial.

**3. Triplet Loss (Pérdida de Triplete)**
Posteriormente, la **Triplet Loss** mejoró este enfoque estructurando el entrenamiento en tripletas formadas por tres elementos:
1.  **Ancla ($A$):** El registro de referencia.
2.  **Positivo ($P$):** Una versión aumentada o equivalente del ancla.
3.  **Negativo ($N$):** Un registro diferente (otra entidad).

> **Nota:** En este contexto, el término **"Ancla" (Anchor)** es un concepto técnico de la función de pérdida utilizado en la literatura que designa al registro de referencia. No debe confundirse con los metadatos o nombres de atributos (`[COL]`) del esquema tabular.

El objetivo matemático es asegurar que la distancia entre el ancla y el positivo sea menor que la distancia entre el ancla y el negativo por al menos un margen $\epsilon$:

$$\mathcal{L}_{Triplet} = \max\left( \|f(A) - f(P)\| - \|f(A) - f(N)\| + \epsilon, \ 0 \right)$$

A pesar de su efectividad teórica, la **limitación crítica** de la *Triplet Loss* es su ineficiencia computacional: requiere un proceso de "Minería de Negativos" (*Negative Mining*) para encontrar ejemplos que sean informativos para el modelo (aquellos donde $d(A, P) \approx d(A, N)$). Como señalan Reimers et al. [2], si los negativos son demasiado fáciles, el gradiente es cero y la red deja de aprender; si son demasiado difíciles, el entrenamiento puede volverse inestable.

#### 4.4.2. Enfoque moderno: Multiple Negatives Ranking Loss (MNRL)
Para superar la ineficiencia de minar tripletas explícitas, esta arquitectura implementa la función **Multiple Negatives Ranking Loss (MNRL)**. Aunque enfoques de clasificación binaria directa (como Binary Cross Entropy con concatenación) pueden ofrecer alta precisión en tareas aisladas [5], resultan computacionalmente inviables para la etapa de recuperación masiva. MNRL, en cambio, optimiza directamente el espacio métrico para búsqueda vectorial, ofreciendo un equilibrio óptimo entre **eficiencia de entrenamiento** y **capacidad de ranking** [3].

La premisa, validada industrialmente en el sistema *Smart Reply* de Google [3], aprovecha la eficiencia del lote (*batch*):
* En un lote de entrenamiento con $B$ pares positivos (naturales y sintéticos) $\{(a_i, p_i)\}_{i=1}^B$.
* Para un registro de referencia dado (ancla) $a_i$, su único positivo verdadero es $p_i$.
* **Innovación:** El sistema asume automáticamente que todos los demás positivos del lote $\{p_j\}_{j \neq i}$ son **Negativos Implícitos** (*In-batch Negatives*) para $a_i$.

Esto permite que, en una sola pasada hacia adelante y hacia atrás (*forward/backward pass*), el modelo aprenda a distinguir la entidad correcta de otras $B-1$ entidades distractores sin el costo de procesar pares negativos explícitos.

Además, gracias a la limpieza y al aislamiento a nivel de entidad realizados en la fase 0 (previa a la partición), se tiene una altísima certidumbre estadística de que estos registros paralelos corresponden efectivamente a entidades distintas. Esto mitiga el riesgo de los *Falsos Negativos en Lote* (penalizar a la red por acercar dos registros que resultan ser la misma entidad por casualidad al no haber sido detectados), lo que es un riesgo inherente en la función de pérdida de Triplet Loss tradicional.


**Formalismo Matemático**

El objetivo de entrenamiento se formula como un problema de clasificación de $B$ clases (*Learning to Rank*), donde se maximiza la similitud del par correcto mientras se minimiza la de los $B-1$ negativos.

La función de pérdida $\mathcal{L}$ para un lote de tamaño $B$ se define como la entropía cruzada negativa de las similitudes normalizadas:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{sim(a_i, p_i) / \tau}}{\sum_{j=1}^{B} e^{sim(a_i, p_j) / \tau}}$$

Donde:
* $sim(x, y)$ es la similitud coseno entre los embeddings $\frac{u \cdot v}{\|u\| \|v\|}$.
* $\tau$ es un hiperparámetro de temperatura que escala las similitudes para mejorar la separabilidad de las clases.

Esta configuración permite entrenar el **Bi-Encoder (SBERT)** de la Etapa 1 de manera eficiente. Chang et al. [5] exploran la viabilidad de integrar MNRL en arquitecturas BERT siamesas para tareas de similitud semántica, mientras que Henderson et al. [3] demuestran su superioridad crítica en términos de escalabilidad y convergencia rápida frente a métodos que requieren minería de negativos, justificando su elección para la generación del espacio vectorial denso en esta propuesta.


**Mitigación Estocástica de Falsos Negativos en Lote**

A pesar del filtro determinista aplicado en la Fase 0 para la extracción de la Verdad Base, el conjunto de entrenamiento inevitablemente podría conservar pares positivos reales ocultos. Durante la optimización masiva con MNRL, si dos registros correspondientes a la misma entidad coinciden aleatoriamente en el mismo lote, el algoritmo los tratará matemáticamente como entidades distintas. Esto inyecta ruido directo en el objetivo contrastivo y penaliza al modelo por alinear representaciones semánticamente correctas.

Para mitigar esta limitación, la arquitectura incorpora estrategias de evaluación de márgenes dentro del lote, extendiendo la función MNRL estándar. Se contempla la adopción de técnicas como el filtrado consciente del positivo (*Positive-aware filtering*) mediante umbrales relativos, o la implementación de funciones de pérdida asistidas como `GISTEmbedLoss`.

La mecánica subyacente de estas técnicas evalúa la similitud de los supuestos negativos implícitos en tiempo de entrenamiento; si un negativo alcanza un puntaje de similitud peligrosamente cercano al del par positivo aumentado (dentro de un margen predefinido), el algoritmo lo identifica como un probable falso negativo y lo excluye del cálculo del gradiente, evitando así una supervisión engañosa para la red neuronal.

> **Referencia Técnica:** *Mitigating False Negatives in Retriever Training* (Hugging Face, 2024). Disponible en: https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training

### 4.5. Entrenamiento de la Etapa 2: Clasificación Fina (Cross-Encoder Optimization)
Mientras que la primera etapa (SBERT) se optimiza mediante aprendizaje métrico para la recuperación eficiente de candidatos, la segunda etapa (DITTO) se entrena como un clasificador binario de alta precisión.

El entrenamiento del Cross-Encoder (DITTO) no se realiza evaluando combinaciones aleatorias del conjunto de entrenamiento, ya que la tarea resultaría trivial para la red y computacionalmente ineficiente. En su lugar, se implementa una estrategia de **embudo de entrenamiento**.

El modelo se optimiza minimizando la **Entropía Cruzada Binaria (Binary Cross-Entropy Loss)** sobre un conjunto de datos altamente curado que incluye:

1.  **Clase Positiva ($y=1$):** Los pares $(R_{ref}, R_{aug})$ generados sintéticamente y los positivos reales correspondientes a la partición de entrenamiento.
2.  **Clase Negativa Difícil ($y=0$ / *Hard Negatives*):** Una vez que la Etapa 1 (SBERT) ha sido entrenada, se utiliza para predecir similitudes sobre el propio set de entrenamiento. Los pares que SBERT clasifica erróneamente con alta similitud (falsos positivos) son extraídos explícitamente e inyectados al entrenamiento de DITTO. Esto garantiza que el Cross-Encoder se entrene con ejemplos que realmente desafían su capacidad de discriminación, evitando el riesgo de sobreajuste a ejemplos triviales.


$$\mathcal{L}_{BCE} = - \left( y \cdot \log(p) + (1 - y) \cdot \log(1 - p) \right)$$

Donde:
* $y \in \{0, 1\}$ es la etiqueta real (1 para Match, 0 para No-Match).
* $p = \text{Softmax}(W \cdot \text{[CLS]})$ es la probabilidad predicha por el modelo de que el par sea una coincidencia [1].

Esta estrategia de minería de negativos difíciles obliga a la red de Atención Cruzada a especializarse exclusivamente en discriminar discrepancias minúsculas y detalles finos que lograron engañar a la etapa de SBERT.

La separación de objetivos es crucial: MNRL entrena a la Etapa 1 para **"traer candidatos probables"** (Recall alto), mientras que Cross-Entropy entrena a la Etapa 2 para **"decidir la verdad"** (Precision alta) observando las interacciones finas entre tokens de ambos registros [1].


## 5. Fase de Inferencia: Pipeline Híbrido de Búsqueda y Vinculación

Si bien la arquitectura basada en SBERT captura eficazmente la equivalencia semántica, los modelos de lenguaje basados en *sub-word tokenization* pueden presentar limitaciones al discriminar diferencias sintácticas finas en tokens no semánticos, como identificadores numéricos, fechas o variaciones de un solo carácter en apellidos (ej. "ID-505" vs "ID-506").

Para mitigar esta "miopía sintáctica" y robustecer la decisión final sin sacrificar la escalabilidad, la fase de inferencia (producción) abandona las comparaciones exhaustivas $O(N^2)$ en favor de un sistema de embudo (*funnel*) de dos etapas: **Recuperación (Retrieval)** y **Re-clasificación (Reranking)**.

### 5.1. Indexación y Recuperación Vectorial (Fase de Búsqueda / Retrieval)
En un escenario del mundo real, comparar un registro entrante contra una base de datos histórica masiva requiere optimización extrema para evitar la complejidad cuadrática $O(N^2)$ de una búsqueda exhaustiva.

1. **Pre-computación e Indexación Vectorial:** Todos los registros de la base de datos de referencia son procesados *offline* por el Bi-Encoder (SBERT), generando representaciones vectoriales densas $v \in \mathbb{R}^d$. Para la etapa de inferencia en tiempo real, el estado del arte ofrece dos paradigmas principales para recuperar a los candidatos, cuya elección depende de los recursos de hardware disponibles y la tolerancia a la pérdida geométrica:
2. **Búsqueda Exacta mediante Multiplicación de Matrices (Exact MIPS):** Este es el enfoque original adoptado por arquitecturas como DITTO [1]. En lugar de construir índices aproximados de búsqueda, este método optimiza el cálculo matemático directo (*Maximum Inner Product Search*) mediante la multiplicación de matrices por bloques, aprovechando al máximo aceleradores de hardware modernos (GPUs/TPUs) [Abuzaid et al., 2019]. Su ventaja radica en garantizar una recuperación matemáticamente exacta basada en la Similitud Coseno, eliminando la posibilidad de falsos negativos derivados de la indexación.
3. **Búsqueda Aproximada de Vecinos Más Cercanos (ANN):** Para entornos clínicos masivos donde el uso exhaustivo de GPUs para inferencia no es viable, los vectores se almacenan utilizando algoritmos ANN implementados en librerías de alto rendimiento como FAISS. Técnicas como grafos jerárquicos (HNSW) o índices invertidos (IVF) pre-estructuran el espacio vectorial. Cuando ingresa una consulta $q$, el sistema navega el índice reduciendo el tiempo de búsqueda a $O(\log N)$. El compromiso (*trade-off*) consiste en una ganancia masiva de escalabilidad y velocidad en CPU, a cambio de sacrificar una fracción estadísticamente minúscula de precisión geométrica.
4. **Salida:** Independientemente del motor de búsqueda subyacente (Exact MIPS o ANN), esta primera etapa actúa como un filtro de alta cobertura (*High Recall*). Retorna en milisegundos únicamente una lista reducida con los identificadores (IDs) de los Top-$K$ candidatos más probables, evadiendo millones de comparaciones secuenciales de texto y alimentando eficientemente a la etapa de Re-clasificación profunda.

### 5.2. Re-clasificación Profunda (Fase de Vinculación / Reranking)
Los $K$ candidatos recuperados pasan a la etapa de escrutinio final utilizando el modelo Cross-Encoder (DITTO).

> **Aclaración Arquitectónica Crítica (Texto Crudo vs. Embeddings):** Es imperativo notar que el Cross-Encoder **no recibe los embeddings vectoriales** generados por SBERT. La Etapa 1 solo devuelve los identificadores (IDs) de los candidatos. Con estos IDs, el sistema recupera las secuencias serializadas en **texto crudo** originales. Esto permite que la Atención Cruzada de DITTO opere directamente sobre el vocabulario explícito y no sobre representaciones comprimidas, resolviendo así la "miopía sintáctica" [1].

Para evaluar cada par, el registro de consulta $q$ y un candidato $c_k$ se concatenan formando una secuencia unificada:
$$S_{pair} = \text{[CLS]} \ q \ \text{[SEP]} \ c_k \ \text{[SEP]}$$

**Gestión de la Ventana de Contexto:**
Dado que el Cross-Encoder procesa el par concatenado, la ventana máxima de 512 tokens del Transformer se comparte entre ambos registros. Por lo tanto, el algoritmo de resumen estadístico (TF-IDF) descrito en la Sección 2.3 se parametriza para restringir cada secuencia individual a un máximo de **256 tokens**, garantizando que el par unificado nunca exceda el límite de memoria del modelo [1].

Los $K$ pares concatenados se procesan a través de DITTO, el cual utiliza sus capas de auto-atención profunda para evaluar las discrepancias finas carácter por carácter. La salida es una probabilidad de coincidencia para cada candidato evaluado.

### 5.3. Criterio de Decisión y Resolución de Conflictos
La decisión final de *Record Linkage* no depende de un vector de características manual, sino de la probabilidad emitida por la capa Softmax del Cross-Encoder.

El sistema aplica un umbral de decisión estricto $\tau_{match}$ (ej. $\tau = 0.90$) sobre la probabilidad predicha:
$$P(\text{Match} | q, c_k) = \sigma(W \cdot \text{[CLS]}_{pair})$$

* **Match Directo:** Si un único candidato supera el umbral $\tau_{match}$, se declara una vinculación exitosa.
* **Resolución de Conflictos:** En el caso atípico de que múltiples candidatos superen el umbral (por ejemplo, gemelos o registros altamente duplicados), el sistema implementa una función *ArgMax*, seleccionando al candidato con la mayor probabilidad absoluta. Si ningún candidato supera el umbral, el registro de consulta $q$ se clasifica como una nueva entidad (*No-Match*).

* **Supervisión Experta (Human-in-the-Loop):** Es imperativo destacar que, dada la naturaleza crítica de los datos de salud del INER, el sistema no ejecuta fusiones o alteraciones destructivas en las bases de datos de forma autónoma. Las vinculaciones propuestas por el modelo que superen el umbral $\tau_{match}$ se emiten como **recomendaciones de alta confianza**, sujetas a la validación final y resolución de un administrador de datos o personal médico experto.

## Bibliografía y Sustento del Estado del Arte

Esta arquitectura se fundamenta en la literatura reciente de Neural Entity Matching y Metric Learning, alineándose con las metodologías que actualmente lideran los benchmarks de resolución de entidades en datos no estructurados.

1. **Sobre la Serialización y el Uso de Transformers (El "Qué")**
* **Paper Clave:** Ditto: Deep Entity Resolution with Pre-Trained Language Models
* Autores: Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHay Doan, Wang-Chiew Tan.
Año: 2020 (EMNLP).
* **Justificación para la Tesis:** Este es el paper fundacional que demostró que serializar registros tabulares como texto y pasarlos por un modelo tipo BERT supera a los métodos que comparan atributo por atributo. Justifica la decisión de concatenar los bloques semánticos y usar tokens especiales (`[BLK]`,`[COL]`, `[VAL]`). Es la base técnica de la Etapa 2 (Cross-Encoder) de esta propuesta.

2. **Sobre la Arquitectura Bi-Encoder Siamesa (El "Cómo")**
* **Paper Clave:** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
* Autores: Nils Reimers, Iryna Gurevych.
Año: 2019 (EMNLP).
* **Justificación para la Tesis:** Introduce la arquitectura de Dos Torres (Bi-Encoder) para generar embeddings de oraciones semánticamente significativos que pueden compararse con distancia coseno. Es la base técnica de nuestro modelo de la etapa 1. Esta es la etapa de "Recuperación" que permite filtrar candidatos de manera eficiente antes de la clasificación fina.
* **Por qué supera a BERT nativo:** Reduce el tiempo de búsqueda de $O(N^2)$ a algo manejable mediante la indexación de vectores densos.

3. **Sobre la Función de Pérdida (Multiple Negatives Ranking Loss)**
* **Paper Clave:** Efficient Natural Language Response Suggestion for Smart Reply (Base conceptual)
* Autores: Matthew Henderson et al. (Google Research).
Año: 2017 / Adaptado en SBERT (2020).
* **Justificación para la Tesis:** Justifica el uso de In-batch Negatives. En lugar de minar tripletas manualmente (que es costoso y difícil), este paper valida que usar los otros ejemplos del lote como negativos es una estrategia eficiente y efectiva para aprender ranking semántico.

4. **Sobre el Manejo de Datos Heterogéneos (Deep Learning vs. Reglas)**
* **Paper Clave:** Deep Learning for Entity Matching: A Design Space Exploration
* Autores: Sidharth Mudgal, Han Li, Theodoros Rekatsinas, AnHai Doan, Youngchool Park, Ganesh Krishnan, Raghavendra Deep, Ihab F. Ilyas, Jeffrey Naughton.
Año: 2018 (SIGMOD).
* **Justificación para la Tesis:** Aunque es previo a los Transformers, este trabajo establece que las redes neuronales (Deep Learning) son superiores a las reglas manuales y a los modelos de ML clásicos (SVM/Random Forest) cuando los datos son sucios, textuales y heterogéneos, exactamente como nuestro caso con INER.

5. **Sobre la Eficiencia de MNRL en Arquitecturas BERT Siamesas**
**Paper Clave:** Chang, J., Alate, K., & Grover, K. "That was smooth": Exploration of S-BERT with Multiple Negatives Ranking Loss and Smoothness-Inducing Regularization. Stanford CS224N Project.