Este markdown contiene las tablas del Diccionario de Datos para el dataset `INER_COVID19_CostoPacientes_Econo.csv`.
Este CSV, integra información sobre el costo económico de la atención médica de los pacientes con COVID-19 atendidos en el INER, fue unificada por Óscar Uriel Pérez Salazar dentro de su trabajo de tesis, dejando como resultado un dataset que se presume limpio y listo para su análisis, cuya información más relevante fue consolidada en el siguiente diccionario de datos.

**Diccionario de datos de la tabla enfocada en los pacientes ingresados.**

| Columna | Tipo de dato | Tipo de variable | Descripción |
| :--- | :--- | :--- | :--- |
| EXP | Carácter | Texto | Expediente del paciente. |
| NOMBRE_DEL_PACIENTE | Carácter | Texto | Nombre completo del paciente. |
| SEXO | Carácter | Categórica | Sexo del paciente. |
| EDAD | Numérico | Numérico | Edad del paciente. |
| GRUPO_EDAD | Carácter | Categórica | Grupo de edad a la que pertenece el paciente. |
| RESULTADO | Carácter | Texto | Resultado con base a los valores de MUESTRA e HISOPADO. |
| ETIQUETAS_COVID | Carácter | Texto | Etiquetas que indican si el paciente tiene o no COVID, así como vacunas, entre otras cosas. |
| MOTIVO_DE_EGRESO | Carácter | Texto | Motivo por el cual egresó el paciente. |
| FECHA_INGRESO_INER | Numérico | Fecha | Fecha y hora de ingreso del paciente al INER. |
| FECHA_DE_ALTA_MEJORIA | Numérico | Fecha | ExpedienteFecha de alta del paciente. |
| DIAS_ESTANCIA | Numérico | Numérico | Días de estancia del paciente en el INER. |
| GASTO_TOTAL | Numérico | Numérico | Gasto total del paciente durante su estancia en el INER. |
| GASTO_DIARIO | Numérico | Numérico | Gasto por día de estancia del paciente. |
| TOTAL_DE_INGRESOS | Numérico | Numérico | Monto total de ingresos del paciente. |
| TOTAL_DE_EGRESOS | Numérico | Numérico | Monto total de egresos del paciente. |
| ESCOLARIDAD | Carácter | Categórica | Último grado de escolaridad del paciente independientemente de que lo haya acabado o no. |
| OCUPACION | Carácter | Categórica | Ocupación del paciente categorizada. |
| DERECHOHABIENTE_Y/O_BENEFICIARIO | Carácter | Categórica | Seguro o cobertura médica que goza el paciente. |
| VULNERABILIDAD_SOCIOECONOMICA | Booleano | Booleano | Indica si el paciente tiene vulnerabilidad socioeconómica. |
| NIVEL_SOCIOECONOMICO | Carácter | Categórica | Nivel socioeconómico del paciente. |
| ESTADO_RESIDENCIA | Carácter | Categórica | Estado de residencia del paciente. |
| CLAVE_GEOESTADISTICA_ESTATAL | Carácter | Texto | Clave geoestadística estatal asignada por el INEGI. |
| MUNICIPIO_RESIDENCIA | Carácter | Categórica | Municipio de residencia del paciente. |
| CLAVE_GEOESTADISTICA_MUNICIPAL | Carácter | Texto | Clave geoestadística municipal asignada por el INEGI. |