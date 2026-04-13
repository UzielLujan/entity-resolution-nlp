# Bases de Datos del INER

Este documento describe las bases de datos proporcionadas por la Dra. Mariana Esther. Me han sido proporcionados tres archivos CSV con información de pacientes con COVID-19 atendidos en el Instituto Nacional de Enfermedades Respiratorias (INER) en México. Estos archivos contienen datos relacionados con el costo económico de la atención médica, los diagnósticos y comorbilidades de los pacientes, y la información de trabajo social en un periodo que comprende desde marzo de 2020 hasta mayo de 2023. A continuación, se presenta información detallada sobre cada uno de estos archivos necesaria para la creación de un diccionario de datos y la consolidación de la base de datos final que alimentará al modelo de Ligado de Registros.


## 1 `..._CostoPacientes_Econo.csv`

Este CSV, integra información sobre el costo económico de la atención médica de los pacientes con COVID-19 atendidos en el INER, fue unificada por Óscar Uriel Pérez Salazar dentro de su trabajo de tesis. Contiene 4632 filas y 24 columnas. La información de este CSV se desglosa a continuación:

| Idx | Column | Non-Null Count | Dtype | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| 0 | EXP | 4552 non-null | str | |
| 1 | NOMBRE_DEL_PACIENTE | 4632 non-null | str | |
| 2 | SEXO | 4632 non-null | str | |
| 3 | EDAD | 4625 non-null | float64 | |
| 4 | GRUPO_EDAD | 4625 non-null | str | |
| 5 | RESULTADO | 3409 non-null | str | |
| 6 | ETIQUETAS_COVID | 1550 non-null | str | |
| 7 | MOTIVO_DE_EGRESO | 4614 non-null | str | |
| 8 | FECHA_INGRESO_INER | 4632 non-null | str | |
| 9 | FECHA_DE_ALTA_MEJORIA | 4632 non-null | str | |
| 10 | DIAS_ESTANCIA | 4632 non-null | int64 | |
| 11 | GASTO_TOTAL | 4632 non-null | float64 | |
| 12 | GASTO_DIARIO | 4632 non-null | float64 | |
| 13 | TOTAL_DE_INGRESOS | 3904 non-null | float64 | |
| 14 | TOTAL_DE_EGRESOS | 3693 non-null | float64 | |
| 15 | ESCOLARIDAD | 3493 non-null | str | |
| 16 | OCUPACION | 1630 non-null | str | |
| 17 | DERECHOHABIENTE_Y/O_BENEFICIARIO | 4632 non-null | str | |
| 18 | VULNERABILIDAD_SOCIOECONOMICA | 4632 non-null | bool | |
| 19 | NIVEL_SOCIOECONOMICO | 4025 non-null | str | |
| 20 | ESTADO_RESIDENCIA | 4587 non-null | str | |
| 21 | CLAVE_GEOESTADISTICA_ESTATAL | 4576 non-null | float64 | |
| 22 | MUNICIPIO_RESIDENCIA | 4587 non-null | str | |
| 23 | CLAVE_GEOESTADISTICA_MUNICIPAL | 4576 non-null | float64 | |

## 2 `..._Pacientes_DiagnosticoComorbilidad.csv`

Este CSV contiene información sobre los diagnósticos y comorbilidades de los pacientes con 4278 filas y 24 columnas. Sin embargo, es el CSV más problemático pues sus columnas no están claramente definidas, por lo que,  para entender mejor su contenido, es necesario explorar las columnas, sus valores únicos, valores nulos y su relación con otras variables, de esta forma lograr unificar y estandarizar la información para que se pueda integrar en la base de datos final.


| Idx | Column | Non-Null Count | Dtype | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| 0 | expediente | 4278 non-null | int64 | |
| 1 | nombre | 4278 non-null | str | |
| 2 | fechaing | 4278 non-null | str | |
| 3 | fechaegr | 4278 non-null | str | |
| 4 | diagnosticoprincipal | 4278 non-null | str | |
| 5 | cie101 | 4278 non-null | str | |
| 6 | diagnostico2 | 2927 non-null | str | |
| 7 | cie102 | 2915 non-null | str | |
| 8 | diagnostico3 | 1563 non-null | str | |
| 9 | cie103 | 1561 non-null | str | |
| 10 | diagnostico4 | 461 non-null | str | |
| 11 | cie104 | 461 non-null | str | |
| 12 | dx2 | 2915 non-null | str | |
| 13 | dx3 | 1561 non-null | str | |
| 14 | dx4 | 461 non-null | str | |
| 15 | obesidad | 4278 non-null | float64 | |
| 16 | obesidad1 | 4278 non-null | float64 | |
| 17 | cardiopatia | 4278 non-null | float64 | |
| 18 | comorbi | 4278 non-null | str | |
| 19 | diabetes | 4278 non-null | float64 | |
| 20 | nefropatia | 4278 non-null | float64 | |
| 21 | eaperge | 4278 non-null | float64 | |
| 22 | tephap | 4278 non-null | float64 | |
| 23 | comorbicv | 4278 non-null | str | |

### 2.1 Hallazgos de la Exploración y Decisiones de Estandarización

En esta sección se detallan las características de los campos de diagnóstico y comorbilidad, integrando las descripciones generales con los hallazgos del Análisis Exploratorio de Datos (EDA) que justifican las futuras decisiones de limpieza y estandarización para la creación de la base de datos final.

#### A. Campos de Diagnóstico y Códigos CIE-10
Estos campos registran la condición clínica del paciente en orden de importancia o aparición durante la estancia hospitalaria, siguiendo una estructura o jerarquía clara:


| Grupo de Diagnóstico | Campos Relacionados | Descripción General | Hallazgos del EDA / Decisión Técnica |
| :--- | :--- | :--- | :--- |
| **Principal** | `diagnosticoprincipal`, `cie101` | Causa primaria de ingreso al INER (COVID-19). | Distribución en 3 códigos CIE-10 (`U07.1`, `U07.2`, `U09.9`) pero con **variantes de texto en los diagnósticos**. |
| **Secundario** | `diagnostico2`, `cie102`, `dx2` | Complicaciones agudas derivadas de la afección principal. | Predominio de insuficiencia respiratoria (`J96.0`). Se confirmó que `dx2` es **100% redundante** con `cie102`. |
| **Terciario** | `diagnostico3`, `cie103`, `dx3` | Comorbilidades de base o condiciones crónicas preexistentes. | Mayor presencia de enfermedades metabólicas: Diabetes (E11.9) con 354 casos, HTAS (I10.X) con 432 casos. Se confirmó redundancia de `dx3`. |
| **Cuarto** | `diagnostico4`, `cie104`, `dx4` | Comorbilidades adicionales, aunque con menor frecuencia, también relacionadas con enfermedades metabólicas y cardiovasculares. | Destacan Diabetes Mellitus tipo II (E11.9) con 207 casos y Choque Septico (A41.9) con 86 casos. |


#### B. Variabilidad en la Captura Textual de Diagnósticos vs Códigos CIE-10

A pesar de contar con códigos estandarizados (CIE-10), se halló una alta variabilidad en las descripciones textuales de los diagnósticos de las columnas `diagnostico1-4`. Este alto indice de "ruido" es debido a uso de sinónimos, falta de estandarización en la entrada de datos o simplemente a errores tipográficos en la captura manual.


Por ejemplo, en la siguiente tabla se muestra la cantidad de variantes textuales encontradas para algunos códigos CIE-10 específicos, junto con ejemplos de estas variantes:



| Código CIE-10 | Descripción Médica | Variantes Halladas | Ejemplos de Error / Variación |
| :--- | :--- | :--- | :--- |
| **U07.1** | COVID-19 (Confirmado) | 184 | "NEUMONIA POR SARS COV2", "SARS-COV2", "NEUMONIA VIRAL POR COVID 19", "COVID 19 ", "**N**COV" |
| **J96.0** | Enfermedades del Sistema Respiratorio | 24 | "INSUFICIENC**AI**", "INSU**FIO**CIENCIA", "INSUFICIENCIA RESPIRATORIA**S**", "RESPIRATORIA **AGUDA GRAVE**" |
| **E11.9** | Diabetes Mellitus Tipo II | 4+ | "DIABETES **MLEELITUS** TIPO II" (typo) |
| **E66.9** | Obesidad / Sobrepeso | 8 | Mezcla de términos: "SOBREPESO", "OBESIDAD GRADO I", "OBESIDAD" |



Este fenómeno se repite de manera general, lo que sugiere que el texto libre en los campos de diagnóstico no está estandarizado presentando variaciones sintácticas. En contraste, los códigos CIE-10 asociados a cada diagnóstico ya están estandarizados de forma consistente. Ya que el objetivo del proyecto es aprovechar la información semántica presente en los campos de las bases de datos, pese a sus variaciones lingüísticas, se propone utilizar ambas columnas en combinación.

* **Decisión:** Mantener ambos campos (texto libre y código CIE-10) para cada diagnóstico, consolidandolos en un **único campo de diagnóstico estandarizado** que combine la información de ambos, para enriquecer la semántica y capturar matices adicionales pero manteniendo la estandarización a través de los códigos CIE-10 como fuente de verdad primaria.

#### C. Redundancia de Columnas

Se confirmó mediante análisis de cardinalidad que las columnas `dx2`, `dx3` y `dx4` son redundantes pues presentan una relación 1:1 con los campos `cie10` respectivos, conteniendo **exactamente la misma información**, como si fueran duplicados de los códigos CIE-10. Esto se evidenció al comparar los valores únicos y su frecuencia, donde ambos campos presentan la misma distribución de códigos CIE-10 sin variación alguna.

* **Decisión:** Estas columnas serán eliminadas en la limpieza final.


#### D. Renombramiento de Columnas (Estandarización)

Para mantener la consistencia se propone renombrar las columnas a un esquema que refleje claramente la jerarquía y relación entre los campos de diagnóstico, utilizando un formato uniforme:

* diagnosticoprincipal → diagnostico_1
* diagnostico[2-4] → diagnostico_[2-4]
* cie10[1-4] → cie10_[1-4]
* Las columnas `dx2, dx3, dx4` se eliminarán del set final.


### 2.3 Información específica sobre campos de comorbilidad


#### C. Indicadores Binarios y Categóricos de Comorbilidad
Además de los diagnósticos dinámicos, la base cuenta con etiquetas fijas para condiciones de riesgo.

*   **Indicadores Binarios (`0/1`):** Las columnas `obesidad`, `cardiopatia`, `diabetes`, `nefropatia`, `eaperge` y `tephap` funcionan como banderas de presencia/ausencia. Se detectó que `obesidad` y `obesidad1` son redundantes entre sí.
*   **Campos Categóricos (`comorbi`, `comorbicv`):** Son resúmenes de texto que clasifican al paciente.
    *   `comorbi`: Clasificación general (ej. "Ninguna", "Otras", "HTAS").
    *   `comorbicv`: Especializado en riesgo cardiovascular.

**Conclusión del Bloque:** Para el desarrollo de modelos predictivos o estadísticos, la fuente de verdad primaria serán los códigos **CIE-10** y los **indicadores binarios**, descartando las descripciones textuales originales para evitar sesgos por errores de captura.


- `obesidad` y `obesidad1`: Estos campos presentan valores de `0` y `1`, lo que sugiere que podrían ser simplemente indicadores binarios de la presencia o ausencia de obesidad, aunque no es clara la diferencia entre ambos.

* `cardiopatia`, `diabetes`, `nefropatia`, `eaperge`, `tephap`: Al igual que los campos relacionados con la obesidad, estos campos solo tienen valores de `0` y `1`, por lo tanto, son indicadores binarios de la presencia o ausencia de estas condiciones específicas. En particular:
    - `cardiopatia`: Indica la presencia o ausencia de alguna enfermedad cardíaca.
    - `diabetes`: Indica la presencia o ausencia de diabetes.
    - `nefropatia`: Indica la presencia o ausencia de alguna enfermedad renal.
    - `eaperge`: Indica la presencia o ausencia de Enfermedad Ácido Péptica (EAP) o Enfermedad por Reflujo Gastroesofágico (ERGE).
    - `tephap`: Indica la presencia o ausencia de  Tromboembolismo Pulmonar (TEP) o Hipertensión Arterial Pulmonar (HAP).


- `comorbi` y `comorbicv`: Estos campos presentan diferentes valores de tipo texto, son indicadores categóricos de la presencia o ausencia de comorbilidades en general (`comorbi`) y comorbilidad cardiovascular específica (`comorbicv`). En particular, los valores únicos que presentan estos campos son:
    - `comorbi`:
        - `Ninguna`	1363
        - `Otras`	1214
        - `Obesidad/Trastornos alimentación`	675
        - `Diabetes Mellitus`	605
        - `HTAS`	327
        - `Enfermedades del sistema renal y urinario`	42
        - `TEP &/| HAP`	26
        - `Cardiopatía Isquémica`	15
        - `EAP & ERGE`	9
        - `Insuficiencia Cardíaca`	2
    - `comorbicv`:
        - `0` (sin comorbilidad cardiovascular) 3432
        - `HTAS`	789
        - `TEP &/| HAP`	28
        - `Cardiopatía Isquémica`	26
        - `Insuficiencia Cardíaca`	3



**Catálogo de categorías del CIE-10 (Clasificación Internacional de Enfermedades, 10ª revisión):**

| Capítulo | Título del Capítulo | Rango de Categorías |
| :--- | :--- | :--- |
| Cap. I | Ciertas enfermedades infecciosas y parasitarias | (A00-B99) |
| Cap. II | Tumores [neoplasias] | (C00-D48) |
| Cap. III | Enfermedades de la sangre y de los órganos hematopoyéticos, y ciertos trastornos que afectan el mecanismo de la inmunidad | (D50-D89) |
| Cap. IV | Enfermedades endocrinas, nutricionales y metabólicas | (E00-E90) |
| Cap. V | Trastornos mentales y del comportamiento | (F00-F99) |
| Cap. VI | Enfermedades del sistema nervioso | (G00-G99) |
| Cap. VII | Enfermedades del ojo y sus anexos | (H00-H59) |
| Cap. VIII | Enfermedades del oído y de la apófisis mastoides | (H60-H95) |
| Cap. IX | Enfermedades del sistema circulatorio | (I00-I99) |
| Cap. X | Enfermedades del sistema respiratorio | (J00-J99) |
| Cap. XI | Enfermedades del sistema digestivo | (K00-K93) |
| Cap. XII | Enfermedades de la piel y del tejido subcutáneo | (L00-L99) |
| Cap. XIII | Enfermedades del sistema osteomuscular y del tejido conjuntivo | (M00-M99) |
| Cap. XIV | Enfermedades del sistema genitourinario | (N00-N99) |
| Cap. XV | Embarazo, parto y puerperio | (O00-O99) |
| Cap. XVI | Ciertas afecciones originadas en el período perinatal | (P00-P99) |
| Cap. XVII | Malformaciones congénitas, deformidades y anomalías cromosómicas | (Q00-Q99) |
| Cap. XVIII | Síntomas, signos y hallazgos anormales clínicos y de laboratorio, no clasificados en otra parte | (R00-R99) |
| Cap. XIX | Traumatismos, envenenamientos y algunas otras consecuencias de causas externas | (S00-T98) |
| Cap. XX | Causas externas de morbilidad y de mortalidad | (V01-V99) |
| Cap. XXI | Factores que influyen en el estado de salud y contacto con los servicios de salud | (Z00-Z99) |
| Cap. XXII | Códigos para propósitos especiales, es decir, diagnósticos que no se han clasificado en los capítulos anteriores | (U00-U99) |



## 3. `INER_COVID19_TrabajoSocial.csv`
El CSV de trabajo social contiene información sobre la situación socioeconómica de los pacientes atendidos en el INER, con 14796 filas y 20 columnas. La información de este CSV está bien cuidada y se desglosa a continuación:

| Idx | Column | Non-Null Count | Dtype | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| 0 | AÑO | 14796 non-null | int64 | |
| 1 | FILA | 14796 non-null | int64 | |
| 2 | EXPEDIENTE | 14796 non-null | int64 | |
| 3 | NO. HISTORIA | 14796 non-null | str | |
| 4 | FECHA DE ELABORACIÓN | 14796 non-null | str | |
| 5 | APELLIDO PATERNO | 14795 non-null | str | |
| 6 | APELLIDO MATERNO | 14697 non-null | str | |
| 7 | NOMBRE | 14777 non-null | str | |
| 8 | EDAD | 14145 non-null | str | |
| 9 | FECHA DE NACIMIENTO | 14145 non-null | str | |
| 10 | GENERO | 14128 non-null | str | |
| 11 | DIAGNOSTICO | 5726 non-null | str | |
| 12 | ESCOLARIDAD | 14777 non-null | str | |
| 13 | OCUPACIÓN | 14141 non-null | str | |
| 14 | DERECHOHABIENTE Y/O BENEFICIARIO | 14547 non-null | str | |
| 15 | DELEGACIÓN O MUNICIPIO PERMANENTE | 14545 non-null | str | |
| 16 | ESTADO / PAIS PERMANENTE | 14789 non-null | str | |
| 17 | TOTAL DE PUNTOS | 14791 non-null | str | |
| 18 | NIVEL SOCIOECONÓMICO | 14791 non-null | str | |
| 19 | Unnamed: 19 | 258 non-null | str | |

