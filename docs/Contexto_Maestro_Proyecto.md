# Contexto Maestro del Proyecto: Tesis y Consultoría

## 1. Visión General
Este documento contextualiza y unifica dos esfuerzos paralelos y complementarios sobre las bases de datos de pacientes COVID-19 del INER: la solución de Análisis e Ingeniería de Datos (Consultoría) y la investigación del Estado del Arte en Ligado de Registros (Tesis de Maestría). Estos proyectos académicamente producen resultados separados pero ambos se retroalimentan, pasando de archivos sueltos sin vincular, a entregables del curso de Consultoría y a un sistema moderno de resolución de entidades basado en aprendizaje profundo y técnicas de procesamiento de lenguaje natural.

---

## 2. Eje 1. Consultoría: Análisis e Ingeniería de Datos
Esta es la **fase exploratoria y de estandarización**. Su propósito es diagnosticar la calidad de los datos crudos del INER y construir los cimientos relacionales del proyecto.

Los tres objetivos y productos finales esperados son:
1. **Diccionario de Datos:** Creación de un diccionario de los datos proporcionados identificando las columnas susceptibles a ser consideradas llaves foráneas y que permitan la vinculación entre bases.
2. **Base de Datos Consolidada:** Consolidar las distintas tablas dispersas en una sola base de datos relacional para que esté disponible a sistemas de procesamiento automático.
3. **Métodos de Comparación:** Desarrollo de métodos de comparación de cadenas de texto de longitudes similares, centrados en nombres propios.

> Los detalles operativos están en:* `Contexto_Consultoria_INER.md` ([[Contexto_Consultoria_INER.md]])

---

## 3. Eje 2 Tesis: Sistema de Ligado de Registros
Este es el proyecto académico insignia de mi maestría en Cómputo Estadístico. Su propósito final es diseñar y desarrollar un sistema moderno de resolución de entidades (Record Linkage) que pueda ser aplicado a las bases de datos del INER, pero lo suficientemente robusto para tratar otros conjuntos de datos con problemas similares.

Los objetivos y productos que requiere el modelo de tesis son:

1. **Definición de Bloques Semánticos:** Identificar y definir bloques semánticos relevantes para el proceso de ligado, basados en las columnas identificadas en la fase de consultoría.
2. **Serialización de datos tabulares:** Mapear las columnas originales hacia los bloques semánticos y desarrollar un proceso serialización que transforme los datos tabulares en secuencias de texto adecuados para modelos de lenguaje pre-entrenados.
3. **Consolidación del conjunto de datos:** Consolidar el conjunto de datos vectorizado en formato `.parquet` que contenga las representaciones serializadas de los datos textuales, optimizada para la ingesta del modelo.

* Para el fundamento académico y diseño inicial del proyecto: `ElProtocolodeInvestigacion.md` ([[ElProtocolodeInvestigacion.md]]). Los detalles completos del sistema moderno actual se encuentran en: `Metodologia_arquitectura.md` ([[Metodologia_arquitectura.md]]).

---

## 4. Arquitectura del Flujo de Datos (Pipeline Global)
El ciclo de vida del dato se bifurca para satisfacer los entregables de ambos ejes:

```text
[Bases de Datos Crudas del INER]
           │
           ▼
[EDAs Granulares y Análisis de Duplicados] (Individual por base y cruce conjunto)
           │
           ├─────────────────────────────────────────┐
           ▼                                         ▼
   Rama Consultoría (Entregables)            Rama Tesis (Insumo IA)
   ------------------------------            ----------------------
   • Diccionario_Datos_Origen                • Mapeo a Bloques Semánticos
   • Base Consolidada Relacional             • Serialización a Texto
   • Diccionario_Datos_final                 • Base de Datos vectorizada
   • Reporte de Consultoría                  • Modelo Híbrido (Bi-Encoder + Cross-Encoder)
   • Métodos Léxicos (Levenshtein, etc.)
```
---

## 5. Roadmap y Estado Actual

* **Fase 1: Diseño y Planeación `[x]`**
    * `[x]` Definición de Arquitectura SOTA (Bi-Encoder + Cross-Encoder).
    * `[x]` Destilación de requerimientos de Consultoría.

* **Fase 2: Análisis Exploratorio Orientado a Objetivos de Tesis y Consultoría `[x]`**
    * Caracterización de columnas, Calidad para Serialización y Mapeo a Bloques Semánticos, información completa contenida en:
        * `[x]` EDA Diagnósticos y Comorbilidades: `EDA_Comorbilidad.ipynb`
        * `[x]` EDA Trabajo Social: `EDA_TrabajoSocial.ipynb`
        * `[x]` EDA Gastos y Económico: `EDA_Econo.ipynb`
        * `[x]` Reporte de hallazgos extraídos a partir de dichos notebooks. Disponible en: `~/Documents/Maestria/Tesis/Reporte_Consultoría/out/Reporte_INER.pdf`

* **Fase 3: Entregables de Datos (Consultoría) `[parcial]`**
    * `[ parcial ]` Construcción del **Diccionario_Datos**. Ya se tiene el primer diccionario, falta definir el segundo, consultar `Contexto_Consultoria_INER` ([[Contexto_Consultoria_INER.md]]) para ver la estructura propuesta.
    * `[x]` Script del pipeline de limpieza y consolidación de la base de datos final relacional (módulos sueltos en notebooks).
    * `[x]` Reporte de metodología de comparación sintáctica (progreso actual reportado dentro de `Reporte_INER.pdf`).

* **Fase 4: Implementación Neuronal (Tesis) `[ ]`**
    * `[x]` Extracción de Verdad Base mediante análisis de duplicados (cubierto en `Reporte_INER.pdf`).
    * `[siguiente]` Data Augmentation y Entrenamiento SBERT.

## 6. Artefactos y Rutas Externas al Repo

### Manuscrito de Tesis (LaTeX)
- **Fuente:** `~/Documents/Maestria/Tesis/Tesis_Latex/Tesis_UzielLujan.tex`
- **PDF compilado:** `~/Documents/Maestria/Tesis/Tesis_Latex/Tesis_Uziel_EscritoActual.pdf`
- **Estructura:** `Capitulos/`, `Figuras/`, `Resultados/`, `Bibliografia/`, `Preambulo.tex`

### Reporte de Consultoría (LaTeX)
- **Fuente:** `~/Documents/Maestria/Tesis/Reporte_Consultoría/Reporte_INER.tex`
- **PDF compilado:** `~/Documents/Maestria/Tesis/Reporte_Consultoría/out/Reporte_INER.pdf`
- **Estructura:** `Capitulos/`, `Figuras/`, `Resultados/`, `Bibliografia/`, `Preambulo.tex`

### Anexos de Bloques Semánticos y estimación de tokens
- **Fuente:** `~/Documents/Maestria/Tesis/Anexo_BS_T/Reporte_BS_y_T.tex`
- **Archivos:** `3_EDA_Comorbilidad.tex`, `3_EDA_Econo.tex`, `3_EDA_TrabajoSocial.tex`
- **PDF compilado:** `~/Documents/Maestria/Tesis/Anexo_BS_T/Anexo_BloquesSemanticos_Tokens.pdf`
- **Contenido:** Sección 3 de cada EDA: estructura de bloques semánticos, columnas por bloque, y especificaciones para serialización en `dataset.py`