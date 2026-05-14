"""Construction and classification of candidate record pairs for entity resolution."""

from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.distance import Levenshtein

from record_linkage.utils.normalization import normalizar_nombre_v2


# Columnas de expediente y nombre por CSV (nombres exactos después de preprocessing)
_COL_MAP = {
    "comorbilidad":   ("expediente",          "nombre"),
    "econo":          ("EXP",                 "NOMBRE_DEL_PACIENTE"),
    "trabajo_social": ("EXPEDIENTE",           "NOMBRE_COMPLETO"),
}


def _detect_csv_type(df: pd.DataFrame) -> str:
    """Detecta el tipo de CSV a partir de sus columnas."""
    cols = set(df.columns)
    if "NOMBRE_DEL_PACIENTE" in cols:
        return "econo"
    if "NOMBRE_COMPLETO" in cols or "APELLIDO PATERNO" in cols:
        return "trabajo_social"
    return "comorbilidad"


def build_pairs_df(
    df_econo: pd.DataFrame,
    df_comor: pd.DataFrame,
    df_ts: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el DataFrame completo de pares candidatos cross-CSV.

    Genera todos los pares de registros que comparten expediente entre CSVs distintos,
    más los pares con expediente NaN que coinciden en nombre_norm (casos Eco-TS).
    Replica la lógica de Duplicados_INER.ipynb sección 4.3 y el bloque diagnóstico
    de build_dataset(), consolidada como fuente única de verdad.

    Args:
        df_econo: DataFrame del CSV Económico (ya preprocesado)
        df_comor: DataFrame del CSV Comorbilidad (ya preprocesado)
        df_ts:    DataFrame del CSV Trabajo Social (ya preprocesado)

    Returns:
        DataFrame con una fila por par candidato y columnas:
            record_id_a, source_a, nombre_a, exp_a,
            record_id_b, source_b, nombre_b, exp_b,
            exp_shared (int64 o pd.NA), nombre_norm_a, nombre_norm_b, nan_exp (bool)

    Notas:
        - Los pares NaN-TS (exp NaN en Económico, mismo nombre_norm en TS) se marcan
          con nan_exp=True y exp_shared=pd.NA. Hay ~31 de estos casos.
        - Los record_id son índices globales asignados aquí; dataset_v2.py los unifica
          al construir el parquet final.
        - entity_id NO se asigna aquí — depende de classify_pairs() y assign_entity_ids().
    """
    frames = {
        "Económico":     (df_econo, _detect_csv_type(df_econo)),
        "Comorbilidad":  (df_comor, _detect_csv_type(df_comor)),
        "Trabajo Social": (df_ts,   _detect_csv_type(df_ts)),
    }

    # Construir tabla auxiliar con columnas normalizadas para cada CSV
    unified_rows = []
    record_id = 0
    for source_name, (df, csv_type) in frames.items():
        exp_col, nombre_col = _COL_MAP[csv_type]
        for _, row in df.iterrows():
            exp_raw = row.get(exp_col)
            nombre_raw = row.get(nombre_col, "")
            unified_rows.append({
                "record_id": record_id,
                "source":    source_name,
                "exp_raw":   exp_raw,
                "exp_int":   pd.to_numeric(exp_raw, errors="coerce"),
                "nombre_norm": normalizar_nombre_v2(nombre_raw),
            })
            record_id += 1

    aux = pd.DataFrame(unified_rows)
    aux["exp_int"] = aux["exp_int"].astype("Int64")

    # --- Pares con expediente compartido (no NaN) ---
    aux_valid = aux[aux["exp_int"].notna()].copy()

    # Deduplicar por (exp_int, source, nombre_norm) antes del merge — igual que v1.
    # Si un CSV tiene registros duplicados con mismo expediente y mismo nombre normalizado,
    # se toma el primero como representante. Los demás se reconectan en dataset_v2._step_finalize.
    aux_dedup = aux_valid.drop_duplicates(subset=["exp_int", "source", "nombre_norm"]).reset_index(drop=True)

    merged = aux_dedup.merge(aux_dedup, on="exp_int", suffixes=("_a", "_b"))
    cross_mask = (
        (merged["record_id_a"] < merged["record_id_b"]) &
        (merged["source_a"] != merged["source_b"])
    )
    pairs_exp = merged[cross_mask].copy()
    pairs_exp["nan_exp"] = False
    pairs_exp["exp_shared"] = pairs_exp["exp_int"].astype("Int64")
    # exp_int es la clave del merge — pandas no la sufija, hay que duplicarla manualmente
    pairs_exp["exp_int_a"] = pairs_exp["exp_int"].astype("Int64")
    pairs_exp["exp_int_b"] = pairs_exp["exp_int"].astype("Int64")

    # --- Pares NaN-TS: expediente NaN en Económico, mismo nombre_norm en Trabajo Social ---
    econo_nan = aux[(aux["source"] == "Económico") & (aux["exp_int"].isna())].copy()
    ts_all    = aux[aux["source"] == "Trabajo Social"].copy()

    # Deduplicar por nombre_norm: si dos registros TS tienen el mismo nombre normalizado,
    # se toma el primero como representante — evita duplicar el par NaN-TS.
    ts_dedup_for_nan = ts_all.drop_duplicates(subset=["nombre_norm"]).copy()

    nan_merged = econo_nan.merge(ts_dedup_for_nan, on="nombre_norm", suffixes=("_a", "_b"))
    nan_merged = nan_merged[nan_merged["nombre_norm"] != ""]  # excluir nombres vacíos
    nan_merged["nan_exp"] = True
    nan_merged["exp_shared"] = pd.NA
    # nombre_norm es la clave del merge — pandas no la sufija, hay que duplicarla manualmente
    nan_merged["nombre_norm_a"] = nan_merged["nombre_norm"]
    nan_merged["nombre_norm_b"] = nan_merged["nombre_norm"]

    # Renombrar para unificar esquema
    cols_keep = [
        "record_id_a", "source_a", "nombre_norm_a", "exp_int_a",
        "record_id_b", "source_b", "nombre_norm_b", "exp_int_b",
        "exp_shared", "nan_exp",
    ]

    pairs_exp = pairs_exp[cols_keep].copy()

    nan_cols = {
        "record_id_a": "record_id_a",
        "source_a":    "source_a",
        "nombre_norm_a": "nombre_norm_a",
        "exp_int_a":   "exp_int_a",
        "record_id_b": "record_id_b",
        "source_b":    "source_b",
        "nombre_norm_b": "nombre_norm_b",
        "exp_int_b":   "exp_int_b",
        "exp_shared":  "exp_shared",
        "nan_exp":     "nan_exp",
    }
    pairs_nan = nan_merged[[c for c in nan_cols]].copy()

    pairs_all = pd.concat([pairs_exp, pairs_nan], ignore_index=True)
    pairs_all = pairs_all.rename(columns={
        "exp_int_a": "exp_a",
        "exp_int_b": "exp_b",
    })

    return pairs_all.reset_index(drop=True)


def classify_pairs(
    pairs_df: pd.DataFrame,
    umbral_jw: float = 0.92,
    umbral_lev: float = 0.85,
    umbral_sem: float = 0.80,
    encoder=None,
    overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Agrega columna 'criterio' con la etiqueta de clasificación de cada par.

    Aplica una cascada de cuatro criterios sobre los pares candidatos:
        1. llave_exacta    — nombre_norm_a == nombre_norm_b y nan_exp=False
        2. metrica_clasica — JW(a,b) >= umbral_jw  OR  Lev_ratio(a,b) >= umbral_lev
        3. metrica_semantica — coseno(BETO(a), BETO(b)) >= umbral_sem
        4. no_confirmado   — ningún criterio resuelve el par, o nan_exp=True

    Los pares nan_exp=True van directamente a no_confirmado (no se aplica cascada).

    IMPORTANTE: Los umbrales por defecto son tentativos. Deben calibrarse empíricamente
    sobre los 9,855 pares llave_exacta antes de usar en producción (ver Paso 6.5 del plan).

    Args:
        pairs_df:      DataFrame producido por build_pairs_df()
        umbral_jw:     Umbral mínimo de Jaro-Winkler (0–1). Default tentativo: 0.92
        umbral_lev:    Umbral mínimo de ratio Levenshtein (0–1). Default tentativo: 0.85
        umbral_sem:    Umbral mínimo de similitud coseno semántica (0–1). Default tentativo: 0.80
        encoder:       Instancia de BiEncoder (build_biencoder()) para metrica_semantica.
                       Si es None, se omite la capa semántica y esos pares van a no_confirmado.
        overrides_path: Ruta a overrides.csv con columnas [record_id_a, record_id_b, criterio].
                       Si se pasa, sobreescribe la clasificación automática para esos pares.
                       Permite incorporar decisiones manuales del notebook de revisión.

    Returns:
        pairs_df con columnas adicionales:
            jw_score (float), lev_score (float), sem_score (float o NaN), criterio (str)
    """
    result = pairs_df.copy()

    # --- Métricas clásicas sobre todos los pares ---
    result["jw_score"] = result.apply(
        lambda r: JaroWinkler.normalized_similarity(r["nombre_norm_a"], r["nombre_norm_b"]),
        axis=1,
    )
    result["lev_score"] = result.apply(
        lambda r: Levenshtein.normalized_similarity(r["nombre_norm_a"], r["nombre_norm_b"]),
        axis=1,
    )
    result["sem_score"] = float("nan")

    # --- Cascada de clasificación ---
    criterio = pd.Series([""] * len(result), index=result.index)

    # Capa 1: llave_exacta
    mask_exacta = (~result["nan_exp"]) & (result["nombre_norm_a"] == result["nombre_norm_b"])
    criterio[mask_exacta] = "llave_exacta"

    # Capa 2: metrica_clasica (solo pares no resueltos con exp válido)
    pending = criterio == ""
    no_nan = ~result["nan_exp"]
    mask_clasica = (
        pending & no_nan &
        ((result["jw_score"] >= umbral_jw) | (result["lev_score"] >= umbral_lev))
    )
    criterio[mask_clasica] = "metrica_clasica"

    # Capa 3: metrica_semantica (solo si hay encoder disponible)
    pending = criterio == ""
    no_nan = ~result["nan_exp"]
    sem_candidates = result[pending & no_nan].index

    if encoder is not None and len(sem_candidates) > 0:
        from record_linkage.models.biencoder import encode_texts
        nombres_a = result.loc[sem_candidates, "nombre_norm_a"].tolist()
        nombres_b = result.loc[sem_candidates, "nombre_norm_b"].tolist()
        emb_a = encode_texts(encoder, nombres_a)
        emb_b = encode_texts(encoder, nombres_b)
        import torch
        cos = torch.nn.functional.cosine_similarity(
            torch.tensor(emb_a), torch.tensor(emb_b), dim=1
        ).numpy()
        result.loc[sem_candidates, "sem_score"] = cos
        mask_sem = pd.Series(False, index=result.index)
        mask_sem[sem_candidates] = cos >= umbral_sem
        criterio[mask_sem] = "metrica_semantica"

    # Capa 4: no_confirmado — todo lo que queda sin clasificar
    criterio[criterio == ""] = "no_confirmado"

    result["criterio"] = criterio

    # --- Overrides manuales ---
    if overrides_path is not None:
        overrides_path = Path(overrides_path)
        if overrides_path.exists():
            ov = pd.read_csv(overrides_path)
            ov_index = result.set_index(["record_id_a", "record_id_b"]).index
            for _, ov_row in ov.iterrows():
                key = (ov_row["record_id_a"], ov_row["record_id_b"])
                matches = (result["record_id_a"] == key[0]) & (result["record_id_b"] == key[1])
                result.loc[matches, "criterio"] = ov_row["criterio"]

    return result
