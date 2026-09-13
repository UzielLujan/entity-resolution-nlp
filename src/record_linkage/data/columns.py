"""Metadatos de columnas preservadas en la serialización por fuente."""

# Campos de identidad después del preprocesamiento de cada fuente.
SOURCE_IDENTITY_COLUMNS = {
    "Comorbilidad": ("expediente", "nombre"),
    "Económico": ("EXP", "NOMBRE_DEL_PACIENTE"),
    "Trabajo Social": ("EXPEDIENTE", "NOMBRE_COMPLETO"),
}
