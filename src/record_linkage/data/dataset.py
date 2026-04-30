"""Dataset construction: serialization, entity labeling, and parquet generation."""

import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


# Mapeo de columnas a bloques semánticos por CSV. Nombres exactos tal como aparecen en los CSVs
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
        "[BLK_ID]": ["NOMBRE_DEL_PACIENTE", "SEXO", "EDAD", "GRUPO_EDAD"],
        "[BLK_ADMIN]": [
            "EXP", "DIAS_ESTANCIA",
            "FECHA_INGRESO_INER", "FECHA_DE_ALTA_MEJORIA",
            "TOTAL_DE_INGRESOS", "TOTAL_DE_EGRESOS",
            "GASTO_TOTAL", "GASTO_DIARIO"
        ],
        "[BLK_CLIN]": ["RESULTADO", "ETIQUETAS_COVID", "MOTIVO_DE_EGRESO"],
        "[BLK_GEO]": [
            "ESTADO_RESIDENCIA", "CLAVE_GEOESTADISTICA_ESTATAL",
            "MUNICIPIO_RESIDENCIA", "CLAVE_GEOESTADISTICA_MUNICIPAL"
        ],
        "[BLK_SOCIO]": [
            "ESCOLARIDAD", "OCUPACION",
            "VULNERABILIDAD_SOCIOECONOMICA",
            "NIVEL_SOCIOECONOMICO",
            "DERECHOHABIENTE_Y/O_BENEFICIARIO"
        ],
    },
    "trabajo_social": {
        "[BLK_ID]": ["APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRE",
                     "EDAD", "FECHA DE NACIMIENTO", "GENERO"],
        "[BLK_ADMIN]": ["EXPEDIENTE", "NO. HISTORIA", "FECHA DE ELABORACIÓN", "AÑO", "FILA"],
        "[BLK_CLIN]": ["DIAGNOSTICO"],
        "[BLK_GEO]": ["DELEGACIÓN O MUNICIPIO PERMANENTE", "ESTADO / PAIS PERMANENTE"],
        "[BLK_SOCIO]": [
            "ESCOLARIDAD", "OCUPACIÓN",
            "DERECHOHABIENTE Y/O BENEFICIARIO",
            "TOTAL DE PUNTOS",
            "NIVEL SOCIOECONÓMICO"
        ],
    },
}

# Orden de serialización de bloques semánticos
SERIALIZATION_ORDER = ["[BLK_ID]", "[BLK_CLIN]", "[BLK_ADMIN]", "[BLK_GEO]", "[BLK_SOCIO]"]


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
        block_cols: lista de columnas del bloque (nombres exactos del CSV)
        block_name: nombre del bloque (ej: "[BLK_ID]")
        use_block_tokens: si False, omite tokens especiales (para zero-shot)

    Returns:
        Fine-tuning:  "[BLK_*] [COL] col1 [VAL] val1 [COL] col2 [VAL] val2 ..."
        Zero-shot:    "col1: val1 col2: val2 ..."
        Bloque vacío: "" en ambos casos — se omite completamente de la secuencia
    """
    block_values = []
    has_real_value = False

    for col in block_cols:
        if col not in row.index:
            continue

        val = _format_value(row[col])

        if val:
            has_real_value = True
            if use_block_tokens:
                block_values.append(f"[COL] {col} [VAL] {val}")
            else:
                block_values.append(f"{col}: {val}")
        elif use_block_tokens:
            block_values.append(f"[COL] {col} [VAL] NULL")

    if not has_real_value:
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


_ZEROSHOT_EXCLUDE = {"record_id", "source_db"}


def serialize_record_zeroshot(row: pd.Series) -> str:
    """Serializa un registro para evaluación zero-shot.

    Itera las columnas en su orden natural (orden del CSV), omitiendo columnas
    auxiliares añadidas por build_dataset. Campos nulos se omiten por completo.

    Returns:
        "col1: val1 col2: val2 ..." — texto plano sin tokens especiales.
    """
    parts = []
    for col in row.index:
        if col not in _ZEROSHOT_EXCLUDE:
            val = _format_value(row[col])
            if val:
                parts.append(f"{col}: {val}")
    return " ".join(parts)


def assign_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna entity_id a registros basado en llave determinista (expediente, nombre_norm).

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
    2. Crea tupla (expediente, nombre_norm) por registro — sin source_db, para vincular entre CSVs
    3. Agrupa registros con tupla idéntica → entity_id único por grupo
    4. Conserva order original de filas (`.reindex()`)

    **Ground Truth:**
    Basado en análisis Duplicados_INER.ipynb:
    - 9,855 pares confirmados (expediente + nombre coinciden entre CSVs)
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
    output_path: Union[str, Path],
    source_db_names: Optional[List[str]] = None,
    use_block_tokens: bool = True,
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
        DataFrame con columnas: record_id, source_db, text, entity_id

    Pipeline:
        1. Lee CSVs y concatena
        2. Asigna record_id (índice global) y source_db (nombre del CSV)
        3. Serializa cada registro con serialize_record()
        4. Asigna entity_ids basado en llave determinista (expediente, nombre_norm)
        5. Guarda como .parquet
        6. Retorna DataFrame para inspección/debugging

    **Ground Truth:**
    Basado en Duplicados_INER.ipynb:
    - Llave: (expediente, nombre_v2_normalizado) — sin source_db para vincular entre CSVs
    - Registros con misma llave → mismo entity_id
    - Automático; no requiere archivo de pares externo
    """
    csv_paths_obj = [Path(str(p)) for p in csv_paths]
    output_path_obj = Path(str(output_path))

    # Leer CSVs
    dfs = []
    for csv_path in csv_paths_obj:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

        df = pd.read_csv(csv_path)
        # Zero-shot: serializar antes del concat para preservar orden original de columnas
        if not use_block_tokens:
            df["text"] = df.apply(serialize_record_zeroshot, axis=1)
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

    # Mapeo de columnas de expediente y nombre por source_db (perfiles con NOMBRE_COMPLETO)
    _COL_MAP = {
        "comorbilidad": ("expediente", "nombre"),
        "econo":        ("EXP",        "NOMBRE_DEL_PACIENTE"),
        "trabajo_social": ("EXPEDIENTE", "NOMBRE_COMPLETO"),
    }

    # Serializar cada registro (solo B1/B2 — zero-shot ya fue serializado antes del concat)
    def get_csv_type(row):
        source = row["source_db"]
        if "comor" in source.lower():
            return "comorbilidad"
        elif "econ" in source.lower():
            return "econo"
        elif "trabajo" in source.lower():
            return "trabajo_social"
        raise ValueError(f"source_db '{source}' no reconocido.")

    if use_block_tokens:
        df_combined["text"] = df_combined.apply(
            lambda row: serialize_record(row, csv_name=get_csv_type(row), use_block_tokens=True),
            axis=1
        )

    # Construir DataFrame auxiliar para assign_entity_ids con columnas normalizadas
    df_for_entity = df_combined[["record_id", "source_db"]].copy()

    def _get_expediente(row):
        csv_type = get_csv_type(row)
        col = "EXP" if csv_type == "econo" else "expediente" if csv_type == "comorbilidad" else "EXPEDIENTE"
        return row.get(col)

    def _get_nombre(row):
        csv_type = get_csv_type(row)
        if csv_type == "trabajo_social":
            if "NOMBRE_COMPLETO" in row.index:
                return row.get("NOMBRE_COMPLETO")
            # Perfil ZS: concatenar campos separados on-the-fly
            parts = ["" if pd.isna(row.get(c)) else str(row.get(c)).strip()
                     for c in ["APELLIDO PATERNO", "APELLIDO MATERNO", "NOMBRE"]]
            return " ".join(p for p in parts if p)
        return row.get(_COL_MAP[csv_type][1])

    df_for_entity["expediente"] = df_combined.apply(_get_expediente, axis=1)
    df_for_entity["nombre"] = df_combined.apply(_get_nombre, axis=1)

    df_for_entity = assign_entity_ids(df_for_entity)

    # Pares residuales a nivel de entidad: mismo expediente, distinta source_db, entity_id diferente.
    # Se deduplica por (expediente, source_db, entity_id) para contar vínculos únicos, no registros.
    _df_exp = df_for_entity[["source_db", "entity_id"]].copy()
    _df_exp["expediente_int"] = pd.to_numeric(df_for_entity["expediente"], errors="coerce")
    _df_exp = _df_exp[_df_exp["expediente_int"].notna()]
    _df_exp = _df_exp.drop_duplicates(subset=["expediente_int", "source_db", "entity_id"]).reset_index(drop=True)
    _df_exp["_idx"] = _df_exp.index
    _merged_exp = _df_exp.merge(_df_exp, on="expediente_int", suffixes=("_a", "_b"))
    _cross_exp_mask = (
        (_merged_exp["_idx_a"] < _merged_exp["_idx_b"]) &
        (_merged_exp["source_db_a"] != _merged_exp["source_db_b"])
    )
    total_mismo_exp = int(_cross_exp_mask.sum())
    residual_pairs = int((_cross_exp_mask & (_merged_exp["entity_id_a"] != _merged_exp["entity_id_b"])).sum())

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

    # Pares confirmados cross-database a nivel de entidad:
    # por cada entidad, C(|source_dbs únicas|, 2) — equivale a contar vínculos únicos entre CSVs.
    _entity_sources = df_output.groupby("entity_id")["source_db"].apply(set)
    cross_db_pairs = int(_entity_sources.apply(lambda s: len(s) * (len(s) - 1) // 2).sum())

    print(f"✓ Dataset guardado: {output_path_obj}")
    print(f"  Registros:                   {len(df_output):,}")
    print(f"  Entidades únicas:            {df_output['entity_id'].nunique():,}")
    vc = df_output['entity_id'].value_counts()
    pares = (vc * (vc - 1) // 2).sum()
    print(f"  Pares positivos (in-batch):  {pares:,}")
    print(f"  Pares confirmados cross-db:  {cross_db_pairs:,}  ← debe ser 9,855")
    print(f"  Pares residuales:            {residual_pairs:,}  ← debe ser 1,569")
    print(f"  Total pares mismo exp.:      {total_mismo_exp:,}  ← debe ser 11,424")

    return df_output
