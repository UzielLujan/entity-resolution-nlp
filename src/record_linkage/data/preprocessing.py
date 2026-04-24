"""
Pipeline de limpieza y preprocesamiento de los CSVs crudos del INER.

Módulos M0–M7, organizados en tres perfiles de ejecución:
  Perfil A  — base analítica completa para el INER (M0→M1→M2→M3→M4→M5→M6→M7)
  Perfil B1 — mínima intervención para tesis (M1→M4 si TS)
  Perfil B2 — limpieza de chars + concat + renombrado (M1→M2→M4 si TS→M7)

M7 es siempre opcional y corre al final para no romper referencias de columnas previas.
"""

import re
import unicodedata
import pandas as pd


# =============================================================================
# M0 — Normalización base de texto categórico  |  CSV: los 3  |  Opcional
# =============================================================================
#
# Aplica strip + upper a todas las columnas dtype==object de cualquier CSV.
# Colapsa variantes sintácticas espurias (espacios finales, capitalización mixta)
# que inflan artificialmente la cardinalidad de columnas categóricas.
# Detectado en EDA_Comorbilidad cells 34-36 (diagnósticos) y EDA_Econo cell 32
# (ESCOLARIDAD, OCUPACION) y EDA_TS cell 35 (variables socioeconómicas).
#
# No forma parte de Perfil A ni Perfil B por defecto — se llama explícitamente.
# Corre antes que M1 y M2: redundante en campos de nombre pero no dañino,
# ya que M1 (? → Ñ) y M2 (limpieza específica) operan sobre el resultado.

def m0_normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.upper()
    return df


# =============================================================================
# M1 — Corrección de encoding  |  CSV: Comorbilidad
# =============================================================================
#
# Problema: el sistema fuente de Comorbilidad codificó mal la 'Ñ' como '?'.
# Detectado en EDA_Comorbilidad cell 21 con:
#   patron = re.compile(r'[^A-ZÁÉÍÓÚÜÑ \-]', re.IGNORECASE)
#   El carácter '?' es el más frecuente (223 registros afectados).
# Ejemplos: 'RAMIREZ NI?O' → 'RAMIREZ NIÑO'
#
# Perfil A: '?' → 'Ñ' (salida legible para el INER)
# Perfil B: '?' → 'N' (NFD posterior elimina la tilde de Ñ de todas formas;
#           consistente con normalizar_nombre_v2 del notebook Duplicados_INER)

def m1_fix_encoding(df: pd.DataFrame, perfil: str = 'A', csv: str = 'comorbilidad') -> pd.DataFrame:
    if csv != 'comorbilidad':
        return df.copy()
    df = df.copy()
    reemplazo = 'Ñ' if perfil == 'A' else 'N'
    df['nombre'] = df['nombre'].str.replace('?', reemplazo, regex=False)
    return df


# =============================================================================
# M2 — Limpieza de caracteres en nombres  |  CSV: los 3
# =============================================================================
#
# Función de detección unificada para los 3 CSV (Duplicados_INER cell 18):
#   patron = re.compile(r'[^A-ZÁÉÍÓÚÜÑ \-]', re.IGNORECASE)
#
# Espacios múltiples (los 3 CSV):
#   r'\s{2,}' → colapsar con str.replace(r'\s{2,}', ' ', regex=True)
#
# Comorbilidad — columna `nombre`:
#   - '?' resuelto por M1 primero; quedan '.', '|'
#
# Económico — columna `NOMBRE_DEL_PACIENTE`:
#   - Punto '.' (49 registros)
#   - Paréntesis con anotaciones clínicas, ej. 'CAMACHO ROJAS MARIA (ESAVI)' (28 regs.)
#     → re.sub(r'\(.*?\)', '', texto).strip()
#   - Barra '/' (1 registro)
#
# Trabajo Social — columnas `APELLIDO PATERNO`, `APELLIDO MATERNO`, `NOMBRE`:
#   - NBSP \xa0 → reemplazar primero, antes de cualquier otra limpieza
#   - Punto '.', barra '/', '|'
#
# NOTA: M2 opera sobre los campos de nombre crudos (pre-concatenación en TS).
# La concatenación de los 3 campos de TS se hace en M4, luego M6 normaliza.

def _limpiar_campo_nombre(texto: str, csv: str) -> str:
    if pd.isna(texto):
        return texto
    s = str(texto)
    if csv == 'trabajo_social':
        s = s.replace('\xa0', ' ')
    if csv == 'econo':
        s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[.\|/]', '', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def m2_clean_nombres(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    """
    csv: 'comorbilidad' | 'econo' | 'trabajo_social'
    """
    df = df.copy()
    if csv == 'comorbilidad':
        df['nombre'] = df['nombre'].apply(lambda x: _limpiar_campo_nombre(x, csv))
    elif csv == 'econo':
        df['NOMBRE_DEL_PACIENTE'] = df['NOMBRE_DEL_PACIENTE'].apply(
            lambda x: _limpiar_campo_nombre(x, csv)
        )
    elif csv == 'trabajo_social':
        for col in ['APELLIDO PATERNO', 'APELLIDO MATERNO', 'NOMBRE']:
            df[col] = df[col].apply(lambda x: _limpiar_campo_nombre(x, csv))
    return df


# =============================================================================
# M3 — Corrección de tipos de datos  |  CSV: los 3  |  Solo Perfil A
# =============================================================================
#
# Mapeo de conversiones extraído de EDAs y reportes LaTeX:
#
# Comorbilidad:
#   - `fechaing`, `fechaegr`: string → datetime (rango 2020-2023)
#   - Columnas binarias (7 cols): float64 → int64
#
# Económico:
#   - `FECHA_INGRESO_INER`, `FECHA_DE_ALTA_MEJORIA`: string → datetime (rango 2020-2023)
#   - `EXP`: reemplazar 'S/E' → NaN, luego → Int64 (nullable int)
#   - `VULNERABILIDAD_SOCIOECONOMICA`: bool → int64
#
# Trabajo Social:
#   - `FECHA DE ELABORACIÓN`, `FECHA DE NACIMIENTO`: string → datetime
#     (formato mixto, dayfirst=True)
#   - `EDAD`: texto libre (ej. "69 Años", "61 Años") → extraer entero con _extraer_anios()
#   - `TOTAL DE PUNTOS`: string numéricos → int64
#
# NOTA — Fechas al exportar a CSV:
#   datetime incluye componente de hora (00:00:00) sin valor informativo.
#   Al exportar la base consolidada del Perfil A usar .dt.strftime('%Y-%m-%d')
#   para columnas de fecha antes de to_csv().
#
# NOTA ARQUITECTÓNICA — Columnas binarias (Perfil A vs Perfil B):
#   Para serialización en dataset.py, las columnas binarias tienen dos opciones:
#   - Perfil B: int64 (0/1) — más compacto, directo
#   - Perfil A (opcional): representación semántica en español ("Verdadero"/"Falso")
#     para que modelos de lenguaje capturen la semántica de presencia/ausencia.
#     Implementar en dataset.py durante serialize_record(), no aquí.

def _extraer_anios(texto):
    if pd.isna(texto):
        return None
    m = re.match(r'(\d+)\s*año', str(texto), re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.match(r'^(\d+)$', str(texto).strip())
    if m2:
        return int(m2.group(1))
    return None

def m3_fix_types(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    if csv == 'comorbilidad':
        df['fechaing'] = pd.to_datetime(df['fechaing'], errors='coerce')
        df['fechaegr'] = pd.to_datetime(df['fechaegr'], errors='coerce')
        cols_binarias = ['obesidad', 'obesidad1', 'cardiopatia', 'diabetes', 'nefropatia', 'eaperge', 'tephap']
        for col in cols_binarias:
            df[col] = (df[col]).astype('int64', errors='ignore')
    elif csv == 'econo':
        df['FECHA_INGRESO_INER'] = pd.to_datetime(df['FECHA_INGRESO_INER'], errors='coerce')
        df['FECHA_DE_ALTA_MEJORIA'] = pd.to_datetime(df['FECHA_DE_ALTA_MEJORIA'], errors='coerce')
        df['EXP'] = df['EXP'].replace('S/E', pd.NA)
        df['EXP'] = pd.to_numeric(df['EXP'], errors='coerce').astype('Int64')
        df['VULNERABILIDAD_SOCIOECONOMICA'] = df['VULNERABILIDAD_SOCIOECONOMICA'].astype('int64')
    elif csv == 'trabajo_social':
        df['FECHA DE ELABORACIÓN'] = pd.to_datetime(
            df['FECHA DE ELABORACIÓN'], format='mixed', dayfirst=True, errors='coerce'
        )
        df['FECHA DE NACIMIENTO'] = pd.to_datetime(
            df['FECHA DE NACIMIENTO'], format='mixed', dayfirst=True, errors='coerce'
        )
        df['EDAD'] = df['EDAD'].apply(_extraer_anios)
        df['TOTAL DE PUNTOS'] = pd.to_numeric(df['TOTAL DE PUNTOS'], errors='coerce').astype('Int64')
    return df


# =============================================================================
# M4 — Concatenación de nombre completo  |  CSV: Trabajo Social
# =============================================================================
#
# Trabajo Social — concatenación de los 3 campos de nombre (EDA_TS cells 11, 23
# y Duplicados_INER cell 10). Opera sobre nombres originales (pre-M7).
#
# Comorbilidad — `obesidad` vs `obesidad1` (EDA_Comorbilidad cells 39-40):
#   Difieren en ~13.7% de registros. `obesidad` aplica criterio clínico más amplio.
#   Decisión de cuál conservar: pendiente de criterio clínico del INER.

def m4_concat_nombre_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['NOMBRE_COMPLETO'] = (
        df['APELLIDO PATERNO'].fillna('') + ' ' +
        df['APELLIDO MATERNO'].fillna('') + ' ' +
        df['NOMBRE'].fillna('')
    ).str.strip()
    return df


# =============================================================================
# M5 — Eliminación de columnas redundantes  |  CSV: Comorbilidad y Trabajo Social
# =============================================================================
#
# Comorbilidad (EDA_Comorbilidad cell 31):
#   - `dx2`, `dx3`, `dx4`: duplicados exactos (100% coincidencia registro a registro)
#     de `cie102`, `cie103`, `cie104` incluyendo mismos nulos → eliminar.
#
# Trabajo Social (EDA_TS cells 31-33, Duplicados_INER cell 2):
#   - `Unnamed: 19`: 98.3% nulos, artefacto del sistema
#   - `AÑO`: redundante con `FECHA DE ELABORACIÓN`, traslapes entre años consecutivos
#   - `FILA`: índice heredado de los 4 archivos anuales originales (valores 0–6107,
#     aparece 1-4 veces según cuántos archivos contenían ese renglón)

_COLS_ELIMINAR = {
    'comorbilidad':   ['dx2', 'dx3', 'dx4'],
    'trabajo_social': ['AÑO', 'FILA'],
}

def m5_drop_columns(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    if csv == 'trabajo_social':
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    cols = _COLS_ELIMINAR.get(csv, [])
    return df.drop(columns=[c for c in cols if c in df.columns])


# =============================================================================
# M6 — Normalización de nombres  |  CSV: los 3  |  Solo Perfil A
# =============================================================================
#
# Función definitiva extraída de Duplicados_INER (normalizar_nombre_v2).
# Integra el efecto de M1 + M2 en un paso de normalización para comparación.
# En el pipeline M1 y M2 ya se aplicaron antes, por lo que aquí
# el replace('?', 'N') es redundante pero inocuo.
#
# Aplicación por CSV:
#   - Comorbilidad: sobre `nombre` (ya corregido por M1 y M2)
#   - Económico:    sobre `NOMBRE_DEL_PACIENTE` (ya limpiado por M2)
#   - Trabajo Social: sobre `NOMBRE_COMPLETO` (producido por M4)

def _normalizar_nombre_v2(texto: str) -> str:
    if pd.isna(texto):
        return ''
    s = str(texto).upper().strip()
    s = s.replace('?', 'N')
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^A-Z ]', '', s)
    return ' '.join(sorted(s.split()))

_COL_NOMBRE = {
    'comorbilidad':   'nombre',
    'econo':          'NOMBRE_DEL_PACIENTE',
    'trabajo_social': 'NOMBRE_COMPLETO',
}

def m6_normalizar_nombres(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    col = _COL_NOMBRE[csv]
    df[f'{col}_norm'] = df[col].apply(_normalizar_nombre_v2)
    return df


# =============================================================================
# M7 — Renombrado de columnas  |  CSV: los 3  |  Opcional, siempre al final
# =============================================================================
#
# Desambigua abreviaciones raras y elimina caracteres especiales de los nombres
# de columna (espacios, slashes, acentos) para mejorar legibilidad e interpretación
#
# IMPORTANTE: debe correr siempre al final del pipeline. Cualquier módulo anterior
# que referencie columnas por nombre asume los nombres originales del CSV crudo.
#
# Beneficios principales:
#   - Abreviaciones raras → tokens semánticos (eaperge, tephap, comorbi, etc.)
#   - Caracteres especiales en nombres de columna → snake_case limpio
#   - Mejora la calidad de los tokens generados durante serialización en dataset.py

_RENOMBRAR = {
    'comorbilidad': {
        'diagnosticoprincipal': 'diagnostico_principal',
        'diagnostico2': 'diagnostico_secundario',
        'diagnostico3': 'diagnostico_terciario',
        'diagnostico4': 'diagnostico_cuarto',
        'comorbi': 'comorbilidad_principal',
        'comorbicv': 'comorbilidad_cardiovascular',
        'eaperge': 'enfermedad_acido_peptica_reflujo',
        'tephap': 'tromboembolismo_pulmonar_hap',
    },
    'econo': {
        'EXP': 'EXPEDIENTE',
        'DERECHOHABIENTE_Y/O_BENEFICIARIO': 'DERECHOHABIENTE_O_BENEFICIARIO',
    },
    'trabajo_social': {
        'NO. HISTORIA': 'NUMERO_HISTORIA',
        'FECHA DE ELABORACIÓN': 'FECHA_ELABORACION',
        'FECHA DE NACIMIENTO': 'FECHA_NACIMIENTO',
        'APELLIDO PATERNO': 'APELLIDO_PATERNO',
        'APELLIDO MATERNO': 'APELLIDO_MATERNO',
        'DERECHOHABIENTE Y/O BENEFICIARIO': 'DERECHOHABIENTE_O_BENEFICIARIO',
        'DELEGACIÓN O MUNICIPIO PERMANENTE': 'DELEGACION_MUNICIPIO_PERMANENTE',
        'ESTADO / PAIS PERMANENTE': 'ESTADO_PAIS_PERMANENTE',
        'TOTAL DE PUNTOS': 'TOTAL_PUNTOS',
        'NIVEL SOCIOECONÓMICO': 'NIVEL_SOCIOECONOMICO',
    }
}

def m7_rename_columns(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    df = df.copy()
    renombres = _RENOMBRAR.get(csv, {})
    return df.rename(columns=renombres)


# =============================================================================
# Perfiles de ejecución — Orquestación de módulos M0–M7
# =============================================================================

def run_profile_a(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    """
    Perfil A — Base analítica completa para el INER.
    Ejecución: M0 → M1 → M2 → M3 → M4(si TS) → M5 → M6 → M7

    Limpieza exhaustiva: tipos, normalización de nombres, eliminación de
    redundancias y renombrado semántico al final.
    """
    df = m0_normalize_text(df)
    df = m1_fix_encoding(df, perfil='A', csv=csv)
    df = m2_clean_nombres(df, csv)
    df = m3_fix_types(df, csv)
    if csv == 'trabajo_social':
        df = m4_concat_nombre_ts(df)
    df = m5_drop_columns(df, csv)
    df = m6_normalizar_nombres(df, csv)
    df = m7_rename_columns(df, csv)
    return df


def run_profile_b1(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    """
    Perfil B1 — Mínima intervención.
    Ejecución: M1 → M4(si TS)

    Columnas conservan nombres originales del CSV crudo. Compatible con
    SEMANTIC_BLOCKS de dataset.py. Punto de partida del Experimento 0.
    """
    df = m1_fix_encoding(df, perfil='B', csv=csv)
    if csv == 'trabajo_social':
        df = m4_concat_nombre_ts(df)
    return df


def run_profile_b2(df: pd.DataFrame, csv: str) -> pd.DataFrame:
    """
    Perfil B2 — Limpieza de caracteres + concat + renombrado semántico.
    Ejecución: M1 → M2 → M4(si TS) → M7

    M7 corre al final: no hay dependencias de orden entre módulos.
    SEMANTIC_BLOCKS para B2 (post-M7) pendiente de definición en dataset.py.
    """
    df = m1_fix_encoding(df, perfil='B', csv=csv)
    df = m2_clean_nombres(df, csv)
    if csv == 'trabajo_social':
        df = m4_concat_nombre_ts(df)
    df = m7_rename_columns(df, csv)
    return df
