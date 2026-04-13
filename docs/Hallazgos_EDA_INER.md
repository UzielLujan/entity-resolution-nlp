# Hallazgos EDA INER — Fuente de la Verdad

---

# 1. EDA_Comorbilidad

## Sección 1. Caracterización de Columnas

### 1.1 Panorama general — Tipos, nulos y cardinalidad

**Resumen: 4,278 registros y 24 columnas**

| Columna | Tipo de Dato | No Nulos | Nulos | % Nulos | Únicos |
|---|---|---|---|---|---|
| expediente | int64 | 4278 | 0 | 0.00% | 4097 |
| nombre | str | 4278 | 0 | 0.00% | 4150 |
| fechaing | datetime64[us] | 4278 | 0 | 0.00% | 988 |
| fechaegr | datetime64[us] | 4278 | 0 | 0.00% | 982 |
| diagnosticoprincipal | str | 4278 | 0 | 0.00% | 214 |
| cie101 | str | 4278 | 0 | 0.00% | 3 |
| diagnostico2 | str | 2927 | 1351 | 31.58% | 204 |
| cie102 | str | 2915 | 1363 | 31.86% | 114 |
| diagnostico3 | str | 1563 | 2715 | 63.46% | 83 |
| cie103 | str | 1561 | 2717 | 63.51% | 59 |
| diagnostico4 | str | 461 | 3817 | 89.22% | 24 |
| cie104 | str | 461 | 3817 | 89.22% | 14 |
| dx2 | str | 2915 | 1363 | 31.86% | 114 |
| dx3 | str | 1561 | 2717 | 63.51% | 59 |
| dx4 | str | 461 | 3817 | 89.22% | 14 |
| obesidad | float64 | 4278 | 0 | 0.00% | 2 |
| obesidad1 | float64 | 4278 | 0 | 0.00% | 2 |
| cardiopatia | float64 | 4278 | 0 | 0.00% | 2 |
| comorbi | str | 4278 | 0 | 0.00% | 10 |
| diabetes | float64 | 4278 | 0 | 0.00% | 2 |
| nefropatia | float64 | 4278 | 0 | 0.00% | 2 |
| eaperge | float64 | 4278 | 0 | 0.00% | 2 |
| tephap | float64 | 4278 | 0 | 0.00% | 2 |
| comorbicv | str | 4278 | 0 | 0.00% | 5 |

### 1.2 Mapa de calor de valores nulos
![alt text](image.png)
### 1.3 Análisis detallado por tipo de variable
En esta sección, para cada una de las 24 columnas, identificamos su tipo de variable y confirmamos qué información contiene realmente, de acuerdo con las siguientes subsecciones de análisis:

- **1.3.1 Identificadores y fechas (4 columnas):**
  - `expediente`, `nombre`, `fechaing`, `fechaegr`

- **1.3.2 Variables categóricas: Diagnósticos, Texto clínico y códigos CIE-10 (11 columnas):**
  - `diagnosticoprincipal`, `diagnostico2`, `diagnostico3`, `diagnostico4`
  - `cie101`, `cie102`, `cie103`, `cie104`
  - `dx2`, `dx3`, `dx4`

- **1.3.3 Variables categóricas: Comorbilidades agrupadas (2 columnas):**
  - `comorbi`, `comorbicv`

- **1.3.4 Variables binarias: Comorbilidades específicas (7 columnas):**
  - `obesidad`, `obesidad1`, `cardiopatia`, `diabetes`, `nefropatia`, `eaperge`, `tephap`

Esto cubre las 24 columnas del csv.
#### 1.3.1 Identificadores y fechas

**Top 10 `expediente` más frecuentes** | **Top 10 `nombre` más frecuentes**

| expediente | Freq | | nombre | Freq |
|---|---|---|---|---|
| 249200 | 5 | | VICENTE MARTIN VALENCIA CHAVEZ | 5 |
| 238376 | 4 | | JAVIER RODRIGUEZ ROJAS | 4 |
| 238904 | 4 | | ALBERTO GARCIA RAMIREZ | 4 |
| 242473 | 4 | | GABINO LEMUS HERNANDEZ | 3 |
| 191202 | 3 | | RUBEN SECUNDINO AGAPITO | 3 |
| 232632 | 3 | | CRECENCIO HERNANDEZ LOPEZ | 3 |
| 237871 | 3 | | GONZALO BECERRIL GUTIERREZ | 3 |
| 114639 | 3 | | JORGE ARTURO VELAZQUEZ CARRANZA | 3 |
| 238389 | 3 | | FELIPE GONZALEZ GUTIERREZ | 2 |
| 238743 | 3 | | DAVID RAMOS DIMAS | 2 |

**Campos de fecha — rango temporal**

| Campo | Nulos | Fecha mín | Fecha máx | Años cubiertos |
|---|---|---|---|---|
| fechaing | 0 | 2020-02-28 | 2023-12-19 | [2020, 2021, 2022, 2023] |
| fechaegr | 0 | 2020-03-02 | 2023-12-24 | [2020, 2021, 2022, 2023] |

#### 1.3.2 Variables categóricas — Diagnósticos, Texto Libre y Códigos CIE-10

**`diagnosticoprincipal`** (214 cat., 0 nulos) | | **`diagnostico2`** (204 cat., 1351 nulos)

| Categoría | Registros | % | | Categoría | Registros | % |
|---|---|---|---|---|---|---|
| NEUMONIA POR SARS COV2 | 1428 | 33.4% | | ⟨NaN⟩ | 1351 | 31.6% |
| SARS-COV2 | 490 | 11.5% | | SIND. INSUF. RESPIRATORIA AGUDA | 1116 | 26.1% |
| COVID 19 | 264 | 6.2% | | HAS | 348 | 8.1% |
| NEUMONIA POR COVID 19 | 254 | 5.9% | | INSUF. RESPIRATORIA TIPO I | 216 | 5.0% |
| ENF. RESP. AGUDA POR nCOV | 150 | 3.5% | | INSUF. RESPIRATORIA AGUDA | 169 | 4.0% |
| INFECCION POR SARS COV2 | 140 | 3.3% | | DIABETES MELLITUS TIPO II | 129 | 3.0% |
| NEUMONIA VIRAL POR COVID 19 | 125 | 2.9% | | OBESIDAD GRADO I | 82 | 1.9% |
| NEUMONIA VIRAL POR SARS COV2 | 101 | 2.4% | | CHOQUE SEPTICO | 77 | 1.8% |
| ENF. RESP. AGUDA POR nCOV | 94 | 2.2% | | SOBREPESO | 64 | 1.5% |
| … (205 más) | 1232 | 28.8% | | … (195 más) | 678 | 12.2% |

**`diagnostico3`** (83 cat., 2715 nulos) | | **`diagnostico4`** (24 cat., 3817 nulos)

| Categoría | Registros | % | | Categoría | Registros | % |
|---|---|---|---|---|---|---|
| ⟨NaN⟩ | 2715 | 63.5% | | ⟨NaN⟩ | 3817 | 89.2% |
| HAS | 432 | 10.1% | | DIABETES MELLITUS TIPO II | 207 | 4.8% |
| DIABETES MELLITUS TIPO II | 356 | 8.3% | | CHOQUE SEPTICO | 86 | 2.0% |
| CHOQUE SEPTICO | 185 | 4.3% | | OBESIDAD GRADO I | 38 | 0.9% |
| OBESIDAD GRADO I | 119 | 2.8% | | SOBREPESO | 30 | 0.7% |
| SOBREPESO | 115 | 2.7% | | OBESIDAD GRADO III | 23 | 0.5% |
| OBESIDAD GRADO II | 52 | 1.2% | | HAS | 23 | 0.5% |
| SIND. INSUF. RESPIRATORIA AGUDA | 36 | 0.8% | | OBESIDAD GRADO II | 18 | 0.4% |
| INSUF. RESPIRATORIA TIPO I | 22 | 0.5% | | OBESIDAD | 8 | 0.2% |
| … (74 más) | 246 | 5.8% | | … (15 más) | 28 | 0.8% |

---


**`cie101`** (3 cat., 0 nulos) | | **`cie102`** (114 cat., 1363 nulos) — `dx2` = `cie102` idéntico (100%)

| cie101 | Reg. | % | | cie102 | Reg. | % |
|---|---|---|---|---|---|---|
| U07.1 | 4217 | 98.6% | | J96.0 | 1563 | 36.5% |
| U07.2 | 58 | 1.4% | | ⟨NaN⟩ | 1363 | 31.9% |
| U09.9 | 3 | 0.1% | | I10.X | 347 | 8.1% |
| | | | | E66.9 | 157 | 3.7% |
| | | | | E11.9 | 156 | 3.6% |
| | | | | A41.9 | 79 | 1.8% |
| | | | | E66.8 | 65 | 1.5% |
| | | | | J39.8 | 54 | 1.3% |
| | | | | J18.9 | 29 | 0.7% |
| | | | | … (105 más) | 465 | 10.8% |

**`cie103`** (59 cat., 2717 nulos) — `dx3` = `cie103` idéntico (100%) | | **`cie104`** (14 cat., 3817 nulos) — `dx4` = `cie104` idéntico (100%)

| cie103 | Reg. | % | | cie104 | Reg. | % |
|---|---|---|---|---|---|---|
| ⟨NaN⟩ | 2717 | 63.5% | | ⟨NaN⟩ | 3817 | 89.2% |
| I10.X | 432 | 10.1% | | E11.9 | 211 | 4.9% |
| E11.9 | 372 | 8.7% | | A41.9 | 86 | 2.0% |
| E66.9 | 253 | 5.9% | | E66.9 | 77 | 1.8% |
| A41.9 | 187 | 4.4% | | E66.8 | 49 | 1.1% |
| E66.8 | 78 | 1.8% | | I10.X | 23 | 0.5% |
| J96.0 | 73 | 1.7% | | J96.0 | 4 | 0.1% |
| J18.9 | 14 | 0.3% | | M06.9 | 3 | 0.1% |
| M06.9 | 12 | 0.3% | | J44.9 | 2 | 0.0% |
| … (50 más) | 140 | 3.3% | | … (5 más) | 6 | 0.1% |

> **`dx2`, `dx3`, `dx4`** son columnas 100% redundantes con `cie102`, `cie103`, `cie104` respectivamente — candidatas a eliminar antes de serialización.

#### 1.3.3 Variables categóricas — Comorbilidades agrupadas

#### 1.3.4 Variables binarias — Comorbilidades específicas

---

## Sección 2. Calidad para Serialización

### 2.1 `nombre` — Campo crítico para Record Linkage

#### 2.1.1 Estructura de la columna `nombre`

#### 2.1.2 Caracteres problemáticos (incluye `?` = Ñ mal codificada)

#### 2.1.3 Nombres duplicados + conflictos expediente ↔ nombre

### 2.2 `expediente` — Identificador primario

### 2.3 Fechas — Coherencia temporal

### 2.4 Códigos CIE-10 — Validación de formato

### 2.5 Redundancia `dx2`/`dx3`/`dx4` vs `cie102`/`cie103`/`cie104`

### 2.6 Variabilidad textual vs. Código CIE-10 — Diagnósticos D1–D4

#### 2.6.1 Resumen de Cardinalidad — Texto Original vs Limpio vs CIE-10

#### 2.6.2 Mega tabla consolidada

#### 2.6.3 Top 10 variantes textuales por código más representativo

### 2.7 Contraste `obesidad` vs `obesidad1`

### 2.8 Relación entre comorbilidades agrupadas y específicas

---

## Sección 3. Bloques Semánticos y Presupuesto de Tokens

### 3.1 Mapeo de columnas a bloques semánticos

### 3.2 Estimación de tokens por registro serializado

---

## Sección 4. Resumen — Hallazgos clave

### 4.1 Diccionario de Datos — CSV Comorbilidad

### 4.2 Calidad — Resumen

### 4.3 Bloques Semánticos — Completitud

### 4.4 Presupuesto de Tokens

### 4.5 Acciones Pre-Serialización

---
---

# 2. EDA_TrabajoSocial

## Sección 0. Carga de datos y configuración

---

## Sección 1. Caracterización de Columnas

### 1.1 Panorama general — Tipos, nulos y cardinalidad

### 1.2 Mapa de calor de valores nulos

### 1.3 Análisis detallado por tipo de variable

#### 1.3.1 Identificadores y Fechas

#### 1.3.2 Variables Categóricas — Demográficas

#### 1.3.3 Variables Categóricas — Socioeconómicas

#### 1.3.4 Variables Categóricas — Clínicas y Geográficas

---

## Sección 2. Calidad para Serialización

### 2.1 Análisis de los campos separados relacionados con el nombre del paciente

#### 2.1.1 Estructura de las columnas de nombre

#### 2.1.2 Caracteres problemáticos

#### 2.1.3 Nombres duplicados + conflictos expediente ↔ nombre

### 2.2 Análisis de `EXPEDIENTE` y `NO. HISTORIA`

### 2.3 Análisis del campo `EDAD` — Texto libre con formato inconsistente

### 2.4 Fechas

### 2.5 `FILA` — Análisis del contenido real

### 2.6 Inconsistencias en variables categóricas

---

## Sección 3. Bloques Semánticos y Presupuesto de Tokens

### 3.1 Mapeo de columnas a bloques semánticos

### 3.2 Estimación de tokens por registro serializado

---

## Sección 4. Resumen — Hallazgos orientados a Record Linkage

### 4.1 Campos Críticos para Serialización

### 4.2 Bloques Semánticos

### 4.3 Presupuesto de Tokens

### 4.4 Inconsistencias Categóricas

### 4.5 Acciones Recomendadas Pre-Serialización

---
---

# 3. EDA_Econo

## Sección 1. Caracterización de Columnas

### 1.1 Panorama general — Tipos, nulos y cardinalidad

### 1.2 Mapa de calor de valores nulos

### 1.3 Análisis detallado por tipo de variable

#### 1.3.1 Identificadores, Fechas y Variables Geográficas

#### 1.3.2 Variables numéricas

#### 1.3.3 Variables categóricas de baja cardinalidad

#### 1.3.4 Variables categóricas de alta cardinalidad

### 1.4 Resumen de hallazgos de la caracterización

---

## Sección 2. Calidad de los Datos

### 2.1 `NOMBRE_DEL_PACIENTE` — Campo crítico para Record Linkage

#### 2.1.1 Estructura de la columna

#### 2.1.2 Caracteres problemáticos

#### 2.1.3 Nombres duplicados + conflictos EXP ↔ nombre

### 2.2 `EXP` — Identificador primario

### 2.3 Fechas — Coherencia temporal

### 2.4 Anomalías y valores extremos en variables numéricas

### 2.5 Inconsistencias Categóricas para Serialización

---

## Sección 3. Bloques Semánticos y Presupuesto de Tokens

### 3.1 Mapeo de columnas a bloques semánticos

### 3.2 Estimación de tokens por registro serializado

---

## Sección 4. Resumen — Hallazgos clave y próximos pasos

### 4.1 Diccionario de Datos — CSV Económico

### 4.2 Calidad Categórica — Resumen

### 4.3 Bloques Semánticos — Completitud

### 4.4 Presupuesto de Tokens

### 4.5 Acciones Pre-Serialización
