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
El INER propuso "Creación de un diccionario de datos de las bases de datos identificando las columnas susceptibles a ser consideradas llaves foráneas". Me tomé la libertad de proponer entregar dos documentos técnicos (.csv) complementarios para documentar el ciclo de vida del dato, desde la exploración de su estado original hasta su consolidación final:

1. **`Diccionario_Origen_INER.csv` (El Diagnóstico):** Bitácora de auditoría que perfila la calidad, anomalías y reglas de limpieza de las bases de datos originales.
   * *Estructura:*
   ```| Archivo_Origen | Columna | Tipo_Dato_Detectado | Total_Registros |Unicos | Nulos | Pct_Nulos | Descripcion | Hallazgos | Transformacion |```

   * *Definiciones clave:*
        * **Hallazgos:** Anomalias, tipos de errores detectados, incluyendo typos, inconsistencias tipográficas, valores faltantes, duplicados, valores fuera de rango o no válidos, formatos incorrectos (ej. orden de nombres, tipo de dato inesperado).

        * **Transformación:** Reglas de limpieza y estandarización aplicadas, como conversión de tipos de datos (ej. casteo de string a int), conversión a minúsculas/mayúsculas, eliminación de espacios, eliminación de columnas redundantes, renombrado para mejorar identificación, limpieza por expresiones regulares, etc.

2. **`Diccionario_Final_INER.csv` (El Entregable):** Mapa técnico que describe la estructura de la nueva base de datos consolidada, sirviendo como guía para el departamento de datos del INER.
   * *Estructura:*
    ```| Tabla_SQL | Columna_Final | Tipo_Dato_Final | Llave | Descripcion_Limpia |```

Sin embargo, esto queda pendiente de confirmar por parte del INER, puede ser que este entregable se reduzca solo a un diccionario descriptivo de los datos originales (en cuyo caso estaría listo) o a algo más simple.

---

## 4. Impacto Esperado

Los productos entregables de esta consultoría resuelven la necesidad operativa del INER, pero estratégicamente constituyen el analisis exploratorio de los datos que utilizará el proyecto de tesis y más importante: **el cimiento del conjunto de datos etiquetado** indispensable para el aprendizaje del modelo:

* **Insumo para la Arquitectura Neuronal:** El análisis exploratorio de los datos justificará la estructura final de los registros serializados (secuencias de texto y *Early Fusion*) requerida por los modelos (SBERT y DITTO).
* **Fundamento para el Aprendizaje Auto-Supervisado:** El perfilado exhaustivo de datos (EDA) y la tipificación de errores de captura (typos) proveerán el contexto necesario para diseñar la estrategia de *Data Augmentation*, permitiendo entrenar la red neuronal multiplicando sintéticamente los pares de registros vinculados.
* **Institucional:** Se transiciona a un pipeline moderno, entregando al INER un sistema que garantiza la vinculación de pacientes en bases de datos fragmentadas para un seguimiento clínico confiable.


## 5. Actualización de pasos siguientes

* **Confirmación del nivel de limpieza requerido por el INER:** Pendiente de definir el nivel de limpieza requerido por el INER, lo cual impactará directamente en la estructura de la base de datos consolidada y del diccionario final. Esto se definirá en conjunto con el asesor de la estancia, considerando sus necesidades operativas. Por el momento, los módulos de limpieza implementados en el pipeline son robustos, generales y basados en los hallazgos del análisis exploratorio de datos, pero se pueden ajustar para incluir reglas más estrictas o específicas según las indicaciones del INER.

* **Implementación de métricas de similitud para vinculación de registros:**
El análisis exploratorio de datos reveló que los nombres propios y expedientes son los campos más susceptibles a ser considerados como llaves foráneas para la vinculación entre bases. Sin embargo, dada la ausencia de identificadores únicos confiables, se definió que el enfoque de comparación sintáctica y métricas clásicas se centrará en la inspección de los registros vinculados entre bases pendientes a confirmar manualmente. Actualmente hay 1569 pares de registros en esta situación, la idea es que utilizando estas métricas, se puedan reducir el número de registros a revisar manualmente, estableciendo umbrales de similitud para automatizar la vinculación de registros con alta confianza
Pendiente de definir con el asesor de la estancia la interfaz de visualización de estos pares para su revisión manual, se ha propuesto una tabla de excel que muestre los registros lado a lado con las métricas de similitud calculadas para facilitar la inspección y la toma de decisiones. Esto permitirá acelerar el proceso de revisión manual, enfocándose en los casos más difíciles de vincular.