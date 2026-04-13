# Plan de Acción de Consultoría: Integración y Homologación de Bases de Datos INER

**Consultor:** Uziel Isaí Luján López (Maestría en Cómputo Estadístico, CIMAT)
**Organización Receptora:** Instituto Nacional de Enfermedades Respiratorias (INER)
**Responsable:** Dra. Mariana Esther Martínez Sánchez
**Marco:** Protocolo B40-25

---

## 1. Definición del Problema
El INER enfrenta un desafío crítico de interoperabilidad y calidad de datos derivado de la emergencia sanitaria por COVID-19. La información de los pacientes se encuentra dispersa en tres fuentes principales. Estos archivos contienen datos relacionados con el costo económico de la atención médica, los diagnósticos y comorbilidades de los pacientes, y la información de trabajo social en un periodo que comprende desde marzo de 2020 hasta mayo de 2023, los cuales se cree que carecen de identificadores únicos (llaves primarias/foráneas) confiables, presentan alta duplicidad y nula estandarización tipográfica.

El objetivo de esta consultoría es resolver el problema de calidad e integración desde una perspectiva de **Análisis e Ingeniería de Datos**, sentando las bases limpias y estructuradas que posteriormente alimentarán el sistema moderno de ligado de registros basado en Procesamiento de Lenguaje Natural y Aprendizaje Profundo (Proyecto de Tesis).

---

## 2. Objetivos Específicos de la Consultoría
Para resolver la problemática institucional, los esfuerzos técnicos se centrarán en:
1. **Perfilado de Datos (Data Profiling):** Analizar la estructura, completitud y anomalías de las tres bases de datos proporcionadas.
2. **Estandarización y Diccionarios:** Identificar columnas susceptibles a ser llaves foráneas y definir reglas estrictas de limpieza y consolidación.
3. **Desarrollo de Métricas Sintácticas:** Implementar métodos de comparación de cadenas de texto (similitud léxica), con especial enfoque en la resolución de nombres propios y errores de captura.
4. **Consolidación Relacional:** Unificar las tablas dispersas en un esquema relacional estructurado y automatizado.

---

## 3. Entregables Oficiales (Productos Finales)

El proyecto de consultoría se considerará exitoso tras la entrega de los siguientes 4 productos:

### Producto 1: Diccionarios de Datos (Diagnóstico y Relacional)
Se entregarán dos documentos técnicos (.csv) complementarios para documentar el ciclo de vida del dato, desde la exploración de su estado crudo hasta el mapeo de su consolidación final:
1. **`Diccionario_Origen_INER.csv` (El Diagnóstico):** Bitácora de auditoría que perfila la calidad, anomalías y reglas de limpieza de las bases de datos originales.
   * *Estructura:* `| Archivo_Origen | Columna | Tipo_Dato_Detectado | Total_Registros |Unicos | Nulos | Pct_Nulos | Descripcion | Hallazgos | Transformacion |`

   * *Definiciones clave:*
        * **Hallazgos:** Anomalias, tipos de errores detectados, incluyendo typos, inconsistencias tipográficas, valores faltantes, duplicados, valores fuera de rango o no válidos, formatos incorrectos (ej. orden de nombres, tipo de dato inesperado).

        * **Transformación:** Reglas de limpieza y estandarización aplicadas, como conversión de tipos de datos (ej. casteo de string a int), conversión a minúsculas/mayúsculas, eliminación de espacios, eliminación de columnas redundantes, renombrado para mejorar identificación, limpieza por expresiones regulares, etc.

2. **`Diccionario_Final_INER.csv` (El Entregable):** Mapa técnico que describe la estructura de la nueva base de datos consolidada, sirviendo como guía para el departamento de datos del INER.
   * *Estructura:* `| Tabla_SQL | Columna_Final | Tipo_Dato_Final | Llave | Descripcion_Limpia |`

### Producto 2: Base de Datos Consolidada
Un esquema relacional limpio, libre de redundancias y estructurado en múltiples tablas normalizadas (ej. `Tabla_Pacientes`, `Tabla_Clinica`, `Tabla_Facturacion`) vinculadas lógicamente mediante llaves primarias y foráneas (PK/FK). Este diseño elimina la redundancia masiva de los archivos planos originales, garantizando la integridad referencial y dejando la información lista para su consulta directa o disponible para ser ingerida por sistemas de procesamiento automático.

### Producto 3: Pipeline de Procesamiento (Código Fuente)
Repositorio organizado con los scripts modulares (desarrollados en Python/Polars) que implementan:
* La limpieza y preprocesamiento de los CSV crudos.
* El modelo algorítmico de ligado de registros basado en similitud de cadenas.
* *Objetivo:* Garantizar la total reproducibilidad del proceso (ETL).

### Producto 4: Reporte de Metodología y EDA
Documento analítico detallando:
* Los hallazgos del Análisis Exploratorio de Datos (EDA) individual de cada base.
* Las decisiones de consolidación tomadas.
* Los métodos de comparación sintáctica utilizados (ej. Levenshtein, Jaro-Winkler) y la justificación de las métricas aplicadas para nombres propios.

---

## 4. Impacto Esperado: Valor Institucional y Fundamento de Tesis

Los productos entregables de esta consultoría resuelven la necesidad operativa del INER, pero estratégicamente constituyen el **cimiento de ingeniería de datos (Fase 0)** indispensable para el éxito del proyecto de tesis:

* **Insumo para la Arquitectura Neuronal:** La base de datos relacional consolidada y el diccionario dinámico dictarán la estructura exacta de los bloques de texto que alimentarán la serialización (*Early Fusion*) requerida por los modelos de lenguaje (SBERT y DITTO).
* **Fundamento para el Aprendizaje Auto-Supervisado:** El perfilado exhaustivo de datos (EDA) y la tipificación de errores de captura (typos) proveerán el contexto necesario para diseñar la estrategia de *Data Augmentation*, permitiendo entrenar la red neuronal sin depender de datos previamente etiquetados o multiplicando sintéticamente los registros existentes.
* **Despliegue Institucional:** Se transiciona de archivos planos aislados a un pipeline moderno, entregando al INER un sistema que garantiza la unicidad del expediente del paciente para un seguimiento clínico confiable.
