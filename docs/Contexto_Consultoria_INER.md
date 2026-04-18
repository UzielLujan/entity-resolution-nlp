# Plan de Acción de Consultoría: Integración y Homologación de Bases de Datos INER

**Consultor:** Uziel Isaí Luján López (Maestría en Cómputo Estadístico, CIMAT)
**Organización Receptora:** Instituto Nacional de Enfermedades Respiratorias (INER)
**Responsable:** Dra. Mariana Esther Martínez Sánchez
**Marco:** Protocolo B40-25

---

## 1. Definición del Problema
El INER enfrenta un desafío crítico de interoperabilidad y calidad de datos derivado de la emergencia sanitaria por COVID-19. La información de los pacientes se encuentra dispersa en tres fuentes principales. Estos archivos contienen datos relacionados con el **costo económico** de la atención médica, los **diagnósticos y comorbilidades** de los pacientes, y la información de **trabajo social** en un periodo que comprende desde marzo de 2020 hasta mayo de 2023, los cuales se cree que carecen de identificadores únicos (llaves primarias/foráneas) confiables, presentan alta duplicidad y nula estandarización tipográfica.

El objetivo de esta consultoría es resolver el problema de calidad e integración desde una perspectiva de **Análisis e Ingeniería de Datos**, sentando las bases limpias y estructuradas que posteriormente alimentarán el sistema moderno de ligado de registros basado en Procesamiento de Lenguaje Natural y Aprendizaje Profundo (Proyecto de Tesis).

---

## 2. Objetivos Específicos de la Consultoría
Para resolver la problemática institucional, los esfuerzos técnicos se centrarán en:
1. **Análisis exploratorio de Datos:** Analizar la estructura, completitud y anomalías de las tres bases de datos proporcionadas.
2. **Estandarización y Diccionarios:** Identificar columnas susceptibles a ser llaves foráneas y definir reglas estrictas de limpieza y consolidación.
3. **Análisis de duplicados y desarrollo de métricas sintácticas:** Implementar métodos de comparación de cadenas de texto (similitud léxica), con especial enfoque en la resolución de nombres propios y errores de captura.
4. **Consolidación Relacional:** Unificar las tablas dispersas en un esquema relacional estructurado y automatizado.

---

## 3. Entregables de la estancia

El proyecto de consultoría se considerará exitoso tras la entrega de los siguientes productos:

### Producto 1: Reporte de Metodología y EDA
Documento analítico detallando:
* Los hallazgos del Análisis Exploratorio de Datos individual de cada base.
* Las decisiones de consolidación tomadas.
* Los métodos de comparación sintáctica utilizados y la justificación de las métricas aplicadas para nombres propios con el objetivo de detectar duplicados y vinculación entre bases.

### Producto 2: Pipeline de limpieza y procesamiento (Código Fuente)
Scripts modulares desarrollados en Python en un repositorio organizado que:
* Implementen la limpieza y preprocesamiento de los CSV crudos según indicaciones del INER (pendiente de confirmar el nivel) 
* Estén justificados en los hallazgos del análisis exploratorio de datos
* Garanticen la total reproducibilidad del proceso.

### Producto 3: Base de Datos Consolidada
Un esquema relacional limpio, libre de redundancias y estructurado en múltiples tablas normalizadas (ej. `Tabla_Pacientes`, `Tabla_Clinica`, `Tabla_Facturacion`) vinculadas lógicamente mediante llaves primarias y foráneas (PK/FK). Este diseño garantiza la integridad referencial y la disponibilidad de la información para su consulta directa o para sistemas de procesamiento automático.

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

Sin embargo, esto queda pendiente de confirmar por parte del INER, puede ser que este entregable se reduzca solo a un diccionario descriptivo de los datos originales (en cuyo caso estaría listo).

---

## 4. Impacto Esperado: Valor Institucional y Fundamento de Tesis

Los productos entregables de esta consultoría resuelven la necesidad operativa del INER, pero estratégicamente constituyen el analisis exploratorio de los datos que utilizará el proyecto de tesis y más importante: **el cimiento del conjunto de datos etiquetado** indispensable para el aprendizaje del modelo:

* **Insumo para la Arquitectura Neuronal:** El análisis exploratorio de los datos justificará la estructura final de los registros serializados (secuencias de texto y *Early Fusion*) requerida por los modelos (SBERT y DITTO).
* **Fundamento para el Aprendizaje Auto-Supervisado:** El perfilado exhaustivo de datos (EDA) y la tipificación de errores de captura (typos) proveerán el contexto necesario para diseñar la estrategia de *Data Augmentation*, permitiendo entrenar la red neuronal multiplicando sintéticamente los pares de registros vinculados.
* **Institucional:** Se transiciona de archivos planos aislados a un pipeline moderno, entregando al INER un sistema que garantiza la vinculación de pacientes en bases de datos fragmentadas para un seguimiento clínico confiable.
