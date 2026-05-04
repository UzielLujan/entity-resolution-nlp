"""
Partición train/val/test del dataset a nivel de entidad.

Garantiza que todos los registros de una misma entidad caigan en el mismo split
(sin data leakage). Estratifica por tipo de entidad:
  - Vinculable  (2+ source_dbs) → pares positivos naturales cross-db
  - Singleton   (1 source_db)   → solo contribuyen vía augmentación en train

Val y test usan únicamente pares naturales (sin augmentación).
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

    La partición ocurre a nivel de entity_id — todos los registros de una entidad
    caen en el mismo split. Se estratifica por tipo (vinculable vs singleton) para
    garantizar pares positivos naturales en val y test.

    Args:
        parquet_path: Ruta al dataset.parquet generado por build_dataset()
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

    # Clasificar entidades: vinculables (2+ CSVs) vs singletons (1 CSV)
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    linkable_ids = entity_sources[entity_sources > 1].index.tolist()
    singleton_ids = entity_sources[entity_sources == 1].index.tolist()

    # Shuffle determinista con semilla
    rng = np.random.default_rng(seed)
    linkable_arr = np.array(linkable_ids)
    singleton_arr = np.array(singleton_ids)
    rng.shuffle(linkable_arr)
    rng.shuffle(singleton_arr)

    def _split(ids: np.ndarray):
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]

    train_link, val_link, test_link = _split(linkable_arr)
    train_sing, val_sing, test_sing = _split(singleton_arr)

    split_map = {}
    for ids, label in [
        (train_link, "train"), (train_sing, "train"),
        (val_link,   "val"),   (val_sing,   "val"),
        (test_link,  "test"),  (test_sing,  "test"),
    ]:
        split_map.update({eid: label for eid in ids})

    df["split"] = df["entity_id"].map(split_map)

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False, compression="snappy")

    # Estadísticas
    stats = {}
    for split_name in ("train", "val", "test"):
        sdf = df[df["split"] == split_name]
        n_linkable = int((entity_sources[sdf["entity_id"].unique()] > 1).sum())
        n_singleton = int((entity_sources[sdf["entity_id"].unique()] == 1).sum())

        src_sets = sdf.groupby("entity_id")["source_db"].apply(set)
        n_pairs = int(src_sets.apply(lambda s: len(s) * (len(s) - 1) // 2).sum())

        stats[split_name] = {
            "entities_linkable":  n_linkable,
            "entities_singleton": n_singleton,
            "entities_total":     n_linkable + n_singleton,
            "records":            len(sdf),
            "natural_pairs":      n_pairs,
        }

    _print_stats(stats, train_ratio, val_ratio, test_ratio)
    return stats


def _print_stats(stats: dict, train_ratio: float, val_ratio: float, test_ratio: float):
    print(f"\n{'='*60}")
    print(f"  Split del dataset  ({train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%})")
    print(f"{'='*60}")
    header = f"  {'':25} {'train':>8} {'val':>8} {'test':>8}"
    print(header)
    print(f"  {'-'*49}")
    rows = [
        ("Entidades vinculables",  "entities_linkable"),
        ("Entidades singleton",    "entities_singleton"),
        ("Entidades total",        "entities_total"),
        ("Registros",              "records"),
        ("Pares naturales cross-db","natural_pairs"),
    ]
    for label, key in rows:
        vals = [stats[s][key] for s in ("train", "val", "test")]
        print(f"  {label:25} {vals[0]:>8,} {vals[1]:>8,} {vals[2]:>8,}")
    print(f"{'='*60}\n")
