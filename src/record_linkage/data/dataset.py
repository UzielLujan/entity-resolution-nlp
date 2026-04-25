"""Dataset construction: serialization, entity labeling, and parquet generation."""

import numpy as np
import pandas as pd
from typing import List, Optional
import unicodedata
import re


# Mapeo de columnas a bloques semánticos por CSV
SEMANTIC_BLOCKS = {
    "comorbilidad": {
        "[BLK_ID]": ["nombre"],
        "[BLK_ADMIN]": ["expediente", "fechaing", "fechaegr"],
        "[BLK_CLIN]": [
            "diagnosticoprincipal", "cie101",
            "diagnostico2", "cie102", "dx2",
            "diagnostico3", "cie103", "dx3",
            "diagnostico4", "cie104", "dx4",
            "comorbi", "comorbicv",
            "obesidad", "obesidad1", "cardiopatia", "diabetes", "nefropatia",
            "eaperge", "tephap"
        ],
    },
    "econo": {
        "[BLK_ID]": ["nombre_del_paciente", "sexo", "edad", "grupo_edad"],
        "[BLK_ADMIN]": [
            "exp", "dias_estancia",
            "fecha_ingreso_iner", "fecha_de_alta_mejoria",
            "total_de_ingresos", "total_de_egresos",
            "gasto_total", "gasto_diario"
        ],
        "[BLK_CLIN]": ["resultado", "etiquetas_covid", "motivo_de_egreso"],
        "[BLK_GEO]": [
            "estado_residencia", "clave_geoestadistica_estatal",
            "municipio_residencia", "clave_geoestadistica_municipal"
        ],
        "[BLK_SOCIO]": [
            "escolaridad", "ocupacion",
            "vulnerabilidad_socioeconomica",
            "nivel_socioeconomico",
            "derechohabiente_y/o_beneficiario"
        ],
    },
    "trabajo_social": {
        "[BLK_ID]": ["apellido_paterno", "apellido_materno", "nombre", "edad", "fecha_de_nacimiento", "genero"],
        "[BLK_ADMIN]": ["expediente", "no._historia", "fecha_de_elaboración", "año", "fila"],
        "[BLK_CLIN]": ["diagnostico"],
        "[BLK_GEO]": ["delegación_o_municipio_permanente", "estado_/_pais_permanente"],
        "[BLK_SOCIO]": [
            "escolaridad", "ocupación",
            "derechohabiente_y/o_beneficiario",
            "total_de_puntos",
            "nivel_socioeconómico"
        ],
    },
}

# Orden de serialización de bloques semánticos
SERIALIZATION_ORDER = ["[BLK_ID]", "[BLK_CLIN]", "[BLK_ADMIN]", "[BLK_GEO]", "[BLK_SOCIO]"]


def _normalize_column_name(col: str) -> str:
    """Normaliza nombre de columna para coincidencia con SEMANTIC_BLOCKS."""
    return col.lower().replace(" ", "_").replace("/", "_").replace(".", "_").replace("-", "_")


def _format_value(val) -> str:
    """Formatea un valor para serialización.

    Nulos → "".
    Numéricos con decimal cero → entero ("0.0" → "0", "1.0" → "1").
    Floats reales → string directo ("18339.19"). Strings → strip.
    Aplica a todos los perfiles: es normalización de presentación para el tokenizador,
    no intervención sobre los datos — independiente del nivel de limpieza del CSV.
    """
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float, np.integer, np.floating)):
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    s = str(val).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return s


def _serialize_block(row: pd.Series, block_cols: List[str], block_name: str,
                     use_block_tokens: bool = True) -> str:
    """Serializa un bloque semántico individual.

    Args:
        row: una fila de un DataFrame
        block_cols: lista de columnas del bloque
        block_name: nombre del bloque (ej: "[BLK_ID]")
        use_block_tokens: si False, omite tokens especiales (para zero-shot)

    Returns:
        Fine-tuning:  "[BLK_*] [COL] col1 [VAL] val1 [COL] col2 [VAL] val2 ..."
        Zero-shot:    "col1: val1 col2: val2 ..."
        Bloque vacío: "" en ambos casos — se omite completamente de la secuencia
    """
    block_values = []

    for col in block_cols:
        col_norm = _normalize_column_name(col)
        val = None
        if col_norm in row.index:
            val = _format_value(row[col_norm])
        else:
            for idx_col in row.index:
                if _normalize_column_name(idx_col) == col_norm:
                    val = _format_value(row[idx_col])
                    break

        if val:
            if use_block_tokens:
                block_values.append(f"[COL] {col_norm} [VAL] {val}")
            else:
                block_values.append(f"{col_norm}: {val}")

    if not block_values:
        return ""

    content = " ".join(block_values)
    return f"{block_name} {content}" if use_block_tokens else content


def serialize_record(row: pd.Series, csv_name: str,
                     use_block_tokens: bool = True) -> str:
    """Serializa un registro tabular a secuencia de texto.

    Args:
        row: Una fila de pandas (pd.Series con nombres de columnas)
        csv_name: Nombre del CSV — debe ser exactamente 'comorbilidad', 'econo' o 'trabajo_social'
        use_block_tokens: True → incluye tokens [BLK_*] (entrenamiento fine-tuned)
                          False → texto limpio sin tokens (zero-shot / baseline)

    Returns:
        Fine-tuning: "[BLK_ID] [COL] nombre [VAL] Juan García [BLK_ADMIN] [COL] expediente [VAL] 12345 ..."
        Zero-shot:   "nombre: Juan García expediente: 12345 ..."
        Bloques sin datos se omiten completamente de la secuencia.
    """
    blocks_def = SEMANTIC_BLOCKS.get(csv_name)
    if blocks_def is None:
        raise ValueError(f"csv_name '{csv_name}' no reconocido. Usar: 'comorbilidad', 'econo' o 'trabajo_social'.")

    serialized_blocks = []

    for block_name in SERIALIZATION_ORDER:
        if block_name in blocks_def:
            block_cols = blocks_def[block_name]
            block_text = _serialize_block(row, block_cols, block_name, use_block_tokens)
            if block_text:
                serialized_blocks.append(block_text)

    return " ".join(serialized_blocks)


def assign_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna entity_id a registros basado en llave determinista (source_db, expediente, nombre).

    Args:
        df: DataFrame con columnas: source_db, expediente, nombre (ya preprocesado)
            - source_db: uno de {'Económico', 'Comorbilidad', 'Trabajo Social'}
            - expediente: int o str convertible a int
            - nombre: string del nombre del paciente (Perfil A o B; se normaliza igual)

    Returns:
        DataFrame original con nueva columna 'entity_id' (int64).
        Registros con la misma tupla (expediente, nombre_norm) reciben el mismo entity_id,
        independientemente del source_db — esto habilita pares positivos cross-database para MNRL.

    **Lógica:**
    1. Normaliza nombres con normalizar_nombre_v2() (desambigua encoding, ordena tokens)
    2. Crea tupla (expediente, nombre_norm) para cada registro — sin source_db
    3. Agrupa registros con tupla idéntica → entity_id único por grupo
    4. Conserva order original de filas (`.reindex()`)

    **Ground Truth:**
    Basado en análisis Duplicados_INER.ipynb:
    - 9,855 pares confirmados (EXP + nombre coinciden)
    - 4,341 entidades vinculables entre 3 CSVs
    - Estrategia en Duplicados_INER sección 4.3 (normalizar_nombre_v2)
    """
    df_result = df.copy()

    # Normalización robusta: ?→N, NFD desacentuación, tokens ordenados
    def normalizar_nombre_v2(texto):
        if pd.isna(texto):
            return ""
        s = str(texto).upper().strip()
        s = s.replace("?", "N")
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        s = re.sub(r"[^A-Z ]", "", s)
        return " ".join(sorted(s.split()))

    # Aplicar normalización
    df_result["nombre_norm"] = df_result["nombre"].apply(normalizar_nombre_v2)

    # Convertir expediente a int (maneja strings y NaN)
    df_result["expediente_int"] = pd.to_numeric(df_result["expediente"], errors="coerce").astype("Int64")

    # Crear tupla (expediente, nombre_norm) — sin source_db para vincular entre CSVs
    df_result["llave_entity"] = df_result.apply(
        lambda row: (row["expediente_int"], row["nombre_norm"]),
        axis=1
    )

    # Agrupar y asignar entity_id
    llave_a_entity_id = {llave: entity_id for entity_id, llave in enumerate(df_result["llave_entity"].unique())}
    df_result["entity_id"] = df_result["llave_entity"].map(llave_a_entity_id).astype("int64")

    # Limpiar columnas auxiliares
    df_result = df_result.drop(columns=["nombre_norm", "expediente_int", "llave_entity"])

    return df_result


def build_dataset(
    csv_paths: List,
    output_path: str,
    source_db_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Construye dataset completo: lectura → serialización → entity assignment → .parquet.

    Args:
        csv_paths: Lista de rutas a CSVs limpios (str o Path, después de preprocessing.py)
            Ejemplo: ['path/to/comorbilidad_clean.csv', 'path/to/econo_clean.csv', 'path/to/ts_clean.csv']

        output_path: Ruta donde guardar dataset.parquet (str o Path)
            Ejemplo: 'path/to/dataset.parquet'

        source_db_names: Lista de nombres para source_db (en mismo orden que csv_paths)
            Ejemplo: ['Comorbilidad', 'Económico', 'Trabajo Social']
            Si es None, se infieren del nombre del CSV

    Returns:
        DataFrame con columnas: record_id, source_db, expediente, nombre, text, entity_id

    Pipeline:
        1. Lee CSVs y concatena
        2. Asigna record_id (índice global) y source_db (nombre del CSV)
        3. Serializa cada registro con serialize_record()
        4. Asigna entity_ids basado en llave determinista (source_db, expediente, nombre)
        5. Guarda como .parquet
        6. Retorna DataFrame para inspección/debugging

    **Ground Truth:**
    Basado en Duplicados_INER.ipynb:
    - Llave: (source_db, expediente, nombre_v2_normalizado)
    - Registros con misma llave → mismo entity_id
    - Automático; no requiere archivo de pares externo
    """
    from pathlib import Path

    csv_paths_obj = [Path(str(p)) for p in csv_paths]
    output_path_obj = Path(str(output_path))

    # Leer CSVs
    dfs = []
    for csv_path in csv_paths_obj:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

        df = pd.read_csv(csv_path)
        dfs.append(df)

    # Inferir source_db si no se proporciona
    if source_db_names is None:
        source_db_names = []
        for csv_path in csv_paths_obj:
            name_lower = csv_path.stem.lower()
            if "comor" in name_lower:
                source_db_names.append("Comorbilidad")
            elif "econ" in name_lower:
                source_db_names.append("Económico")
            elif "trabajo" in name_lower or "social" in name_lower:
                source_db_names.append("Trabajo Social")
            else:
                source_db_names.append(csv_path.stem)

    # Concatenar con source_db
    dfs_with_source = []
    for df, source_db in zip(dfs, source_db_names):
        df["source_db"] = source_db
        dfs_with_source.append(df)

    df_combined = pd.concat(dfs_with_source, ignore_index=True)

    # Asignar record_id global
    df_combined.insert(0, "record_id", range(len(df_combined)))

    # Mapeo de columnas de expediente y nombre por source_db
    # Necesario para assign_entity_ids — no se guardan en el parquet final
    _COL_MAP = {
        "comorbilidad": ("expediente", "nombre"),
        "econo":        ("EXP",        "NOMBRE_DEL_PACIENTE"),
        "trabajo_social": ("EXPEDIENTE", "NOMBRE_COMPLETO"),
    }

    # Serializar cada registro
    def get_csv_type(row):
        source = row["source_db"]
        if "comor" in source.lower():
            return "comorbilidad"
        elif "econ" in source.lower():
            return "econo"
        elif "trabajo" in source.lower():
            return "trabajo_social"
        raise ValueError(f"source_db '{source}' no reconocido.")

    df_combined["text"] = df_combined.apply(
        lambda row: serialize_record(row, csv_name=get_csv_type(row)),
        axis=1
    )

    # Construir DataFrame auxiliar para assign_entity_ids con columnas normalizadas
    df_for_entity = df_combined[["record_id", "source_db"]].copy()
    df_for_entity["expediente"] = df_combined.apply(
        lambda row: row.get(_COL_MAP[get_csv_type(row)][0]), axis=1
    )
    df_for_entity["nombre"] = df_combined.apply(
        lambda row: row.get(_COL_MAP[get_csv_type(row)][1]), axis=1
    )

    df_for_entity = assign_entity_ids(df_for_entity)

    # Merge entity_id de vuelta al DataFrame original
    df_combined = df_combined.merge(
        df_for_entity[["record_id", "entity_id"]],
        on="record_id",
        how="left"
    )

    # Columnas finales según esquema parquet de la metodología (sección 2.2)
    df_output = df_combined[["record_id", "source_db", "text", "entity_id"]]

    # Guardar como parquet (con pyarrow engine para columnas comprimidas)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(
        output_path_obj,
        engine="pyarrow",
        index=False,
        compression="snappy"
    )

    print(f"✓ Dataset guardado: {output_path_obj}")
    print(f"  Registros: {len(df_output):,}")
    print(f"  Entidades: {df_output['entity_id'].nunique():,}")
    print(f"  Pares positivos potenciales (in-batch): {sum(df_output['entity_id'].value_counts() ** 2) // 2:,}")

    return df_output
