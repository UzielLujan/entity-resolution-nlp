"""Dataset construction: serialization, entity labeling, and parquet generation."""

import pandas as pd
from typing import List, Optional


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
            "expediente", "motivo_de_egreso",
            "fecha_ingreso_iner", "fecha_de_alta_mejoria",
            "total_de_ingresos", "total_de_egresos",
            "gasto_total", "gasto_diario"
        ],
        "[BLK_CLIN]": ["resultado", "etiquetas_covid", "dias_estancia"],
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

# Orden de serialización: primero ID y ADMIN, luego CLIN, luego GEO y SOCIO
SERIALIZATION_ORDER = ["[BLK_ID]", "[BLK_ADMIN]", "[BLK_CLIN]", "[BLK_GEO]", "[BLK_SOCIO]"]


def _detect_csv_type(row: pd.Series) -> str:
    """Detecta qué CSV es basado en columnas presentes.

    Returns: "comorbilidad", "econo", o "trabajo_social"
    """
    cols_lower = {col.lower().replace(" ", "_").replace("/", "_").replace(".", "_")
                  for col in row.index}

    # Heurística: detecta por columnas únicas
    if "comorbi" in cols_lower or "comorbicv" in cols_lower:
        return "comorbilidad"
    elif "gasto_total" in cols_lower or "dias_estancia" in cols_lower:
        return "econo"
    elif "ocupación" in cols_lower or "apellido_paterno" in cols_lower:
        return "trabajo_social"
    else:
        # Default: intenta deducir por cantidad de columnas
        if len(row) < 20:
            return "trabajo_social"
        elif len(row) < 30:
            return "comorbilidad"
        else:
            return "econo"


def _normalize_column_name(col: str) -> str:
    """Normaliza nombre de columna para coincidencia con SEMANTIC_BLOCKS."""
    return col.lower().replace(" ", "_").replace("/", "_").replace(".", "_").replace("-", "_")


def _format_value(val) -> str:
    """Formatea un valor para serialización. Maneja nulos y tipos."""
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float)):
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)
    return str(val).strip()


def _serialize_block(row: pd.Series, block_cols: List[str], block_name: str) -> str:
    """Serializa un bloque semántico individual.

    Args:
        row: una fila de un DataFrame
        block_cols: lista de columnas del bloque
        block_name: nombre del bloque (ej: "[BLK_ID]")

    Returns:
        string con formato "[BLK_*] col1: val1 col2: val2 ..."
        o "[BLK_*] (no data)" si todas las columnas son nulas
    """
    block_values = []

    for col in block_cols:
        col_norm = _normalize_column_name(col)
        if col_norm in row.index:
            val = _format_value(row[col_norm])
            if val:
                block_values.append(f"{col_norm}: {val}")
        # Si la columna no existe en la fila, intenta búsqueda case-insensitive
        else:
            for idx_col in row.index:
                if _normalize_column_name(idx_col) == col_norm:
                    val = _format_value(row[idx_col])
                    if val:
                        block_values.append(f"{col_norm}: {val}")
                    break

    if block_values:
        return f"{block_name} {' '.join(block_values)}"
    else:
        return f"{block_name} (no data)"


def serialize_record(row: pd.Series, csv_name: Optional[str] = None) -> str:
    """Serializa un registro tabular a secuencia de texto con tokens [BLK_*].

    Args:
        row: Una fila de pandas (pd.Series con nombres de columnas)
        csv_name: Nombre del CSV (opcional; se detecta automáticamente si no se proporciona)

    Returns:
        string con tokens semánticos, separados por espacios.
        Ej: "[BLK_ID] nombre: Juan García [BLK_ADMIN] expediente: 12345 ..."
    """
    if csv_name is None:
        csv_type = _detect_csv_type(row)
    else:
        csv_name_norm = csv_name.lower().replace("_", "").replace("-", "").replace(".", "")
        if "comorbi" in csv_name_norm:
            csv_type = "comorbilidad"
        elif "econ" in csv_name_norm or "econo" in csv_name_norm:
            csv_type = "econo"
        elif "trabajo" in csv_name_norm or "social" in csv_name_norm:
            csv_type = "trabajo_social"
        else:
            csv_type = _detect_csv_type(row)

    blocks_def = SEMANTIC_BLOCKS.get(csv_type)
    if blocks_def is None:
        raise ValueError(f"CSV type '{csv_type}' not recognized. Use 'comorbilidad', 'econo', or 'trabajo_social'.")

    serialized_blocks = []

    for block_name in SERIALIZATION_ORDER:
        if block_name in blocks_def:
            block_cols = blocks_def[block_name]
            block_text = _serialize_block(row, block_cols, block_name)
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
        Registros con la misma tupla (source_db, expediente, nombre_norm) reciben el mismo entity_id.

    **Lógica:**
    1. Normaliza nombres con normalizar_nombre_v2() (desambigua encoding, ordena tokens)
    2. Crea tupla (source_db, expediente, nombre_norm) para cada registro
    3. Agrupa registros con tupla idéntica → entity_id único por grupo
    4. Conserva order original de filas (`.reindex()`)

    **Ground Truth:**
    Basado en análisis Duplicados_INER.ipynb:
    - 9,855 pares confirmados (EXP + nombre coinciden)
    - 4,341 entidades vinculables entre 3 CSVs
    - Estrategia en Duplicados_INER sección 4.3 (normalizar_nombre_v2)
    """
    import unicodedata
    import re

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

    # Crear tupla (source_db, expediente, nombre_norm)
    df_result["llave_entity"] = df_result.apply(
        lambda row: (row["source_db"], row["expediente_int"], row["nombre_norm"]),
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

    # Serializar cada registro
    def get_csv_type(row):
        source = row["source_db"]
        if "comor" in source.lower():
            return "comorbilidad"
        elif "econ" in source.lower():
            return "econo"
        elif "trabajo" in source.lower():
            return "trabajo_social"
        return "comorbilidad"

    df_combined["text"] = df_combined.apply(
        lambda row: serialize_record(row, csv_name=get_csv_type(row)),
        axis=1
    )

    # Asignar entity_ids
    # Primero, normalizar nombres de columnas para assign_entity_ids
    df_for_entity = df_combined[["source_db", "expediente", "nombre", "record_id", "text"]].copy()

    # Rename a lower_case para consistency
    if "nombre_del_paciente" in df_combined.columns:
        df_for_entity["nombre"] = df_combined["nombre_del_paciente"]

    df_for_entity = assign_entity_ids(df_for_entity)

    # Merge entity_id de vuelta al DataFrame original
    df_combined = df_combined.merge(
        df_for_entity[["record_id", "entity_id"]],
        on="record_id",
        how="left"
    )

    # Seleccionar columnas finales
    df_output = df_combined[["record_id", "source_db", "expediente", "nombre", "text", "entity_id"]]

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
