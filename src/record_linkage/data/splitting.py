"""
Lógica de partición train/val/test del dataset a nivel de entidad.

Garantiza que todos los registros de una misma entidad caigan en el mismo split
(sin data leakage). Estratifica por tipo de entidad:
   - Vinculable → aparece en más de una base de datos y permite formar pares positivos
   - De una sola base de datos → no forma pares positivos entre bases de datos

También reporta singletons estrictos: entidades con exactamente un registro.

Val y test se evalúan con pares positivos entre bases de datos a nivel registro,
sin augmentación.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def split_dataset(
    parquet_path: Union[str, Path],
    output_path: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Asigna columna 'split' (train/val/test) a cada registro del parquet.

    Ocurre a nivel de entity_id: todos los registros de una misma entidad
    caen en el mismo split. Se estratifica por tipo (vinculable vs de una sola base)
    para
    permitir pares positivos entre bases de datos a nivel registro en val y test.

    Args:
        parquet_path: Ruta al dataset.parquet generado por el pipeline de datos
        output_path:  Ruta de salida para el parquet con columna 'split'
        train_ratio:  Fracción de entidades para train (default 0.70)
        val_ratio:    Fracción de entidades para val  (default 0.15)
        seed:         Semilla para reproducibilidad

    Returns:
        dict con estadísticas del split (entidades, registros y pares por partición)
    """
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)
    if test_ratio < 0:
        raise ValueError(f"train_ratio + val_ratio debe ser <= 1.0")

    df = pd.read_parquet(parquet_path)

    # Clasificar entidades: vinculables (2+ bases de datos) vs de una sola base
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    entity_record_counts = df.groupby("entity_id").size()
    linkable_ids = entity_sources[entity_sources > 1].index.tolist()
    single_source_ids = entity_sources[entity_sources == 1].index.tolist()

    # Shuffle determinista con semilla
    rng = np.random.default_rng(seed)
    linkable_arr = np.array(linkable_ids)
    single_source_arr = np.array(single_source_ids)
    rng.shuffle(linkable_arr)
    rng.shuffle(single_source_arr)

    def _split(ids: np.ndarray):
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    train_link, val_link, test_link = _split(linkable_arr)
    train_single_source, val_single_source, test_single_source = _split(single_source_arr)

    split_map = {}
    for ids, label in [
        (train_link, "train"), (train_single_source, "train"),
        (val_link,   "val"),   (val_single_source,   "val"),
        (test_link,  "test"),  (test_single_source,  "test"),
    ]:
        split_map.update({eid: label for eid in ids})

    df["split"] = df["entity_id"].map(split_map)

    n_unassigned = int(df["split"].isna().sum())
    if n_unassigned:
        raise ValueError(f"{n_unassigned} registros no recibieron split")

    splits_per_entity = df.groupby("entity_id", dropna=False)["split"].nunique()
    if not splits_per_entity.eq(1).all():
        raise ValueError("Se detectaron entity_id asignados a más de un split")

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    # Estadísticas
    stats = {}
    for split_name in ("train", "val", "test"):
        sdf = df[df["split"] == split_name]
        n_linkable = int((entity_sources[sdf["entity_id"].unique()] > 1).sum())
        n_single_source = int((entity_sources[sdf["entity_id"].unique()] == 1).sum())
        n_singleton = int((entity_record_counts[sdf["entity_id"].unique()] == 1).sum())

        src_sets = sdf.groupby("entity_id")["source_db"].apply(set)
        n_entity_level_pairs = int(
            src_sets.apply(lambda s: len(s) * (len(s) - 1) // 2).sum()
        )
        records_per_source = sdf.groupby(["entity_id", "source_db"]).size()
        n_record_level_pairs = int(
            records_per_source.groupby(level="entity_id").apply(
                lambda counts: (counts.sum() ** 2 - (counts ** 2).sum()) // 2
            ).sum()
        )

        stats[split_name] = {
            "entities_linkable":  n_linkable,
            "entities_single_source": n_single_source,
            "entities_singleton": n_singleton,
            "entities_total":        n_linkable + n_single_source,
            "records":            len(sdf),
            "cross_db_entity_positive_pairs": n_entity_level_pairs,
            "cross_db_record_positive_pairs": n_record_level_pairs,
        }

    _print_stats(stats, train_ratio, val_ratio, test_ratio)
    return stats


def _print_stats(stats: dict, train_ratio: float, val_ratio: float, test_ratio: float):
    print(f"\n{'='*71}")
    print(f"  Split del dataset  ({train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%})")
    print(f"{'='*71}")
    header = f"  {'':42} {'train':>8} {'val':>8} {'test':>8}"
    print(header)
    print(f"  {'-'*69}")
    rows = [
        ("Entidades vinculables",  "entities_linkable"),
        ("Entidades en una sola base", "entities_single_source"),
        ("Entidades singleton",   "entities_singleton"),
        ("Entidades total",        "entities_total"),
        ("Registros",              "records"),
        ("Pares positivos entre bases (nivel entidad)", "cross_db_entity_positive_pairs"),
        ("Pares positivos entre bases (nivel registro)", "cross_db_record_positive_pairs"),
    ]
    for label, key in rows:
        vals = [stats[s][key] for s in ("train", "val", "test")]
        print(f"  {label:42} {vals[0]:>8,} {vals[1]:>8,} {vals[2]:>8,}")
    print(f"{'='*71}\n")
