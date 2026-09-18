"""Mineria de pares para el pipeline clasico a partir de similitud TF-IDF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def mine_tfidf_pairs(
    df_linkable: pd.DataFrame,
    vectors: csr_matrix,
    top_k: int = 20,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Genera positivos y negativos Top-K cross-source usando TF-IDF.

    Las comparaciones intra-fuente se invalidan antes del Top-K para que cada
    candidato corresponda a una posible vinculacion entre bases de datos.
    """
    required = {"record_id", "entity_id", "source_db"}
    missing = required - set(df_linkable.columns)
    if missing:
        raise ValueError(f"df_linkable sin columnas requeridas: {missing}")
    if len(df_linkable) != vectors.shape[0]:
        raise ValueError("df_linkable y vectors deben tener el mismo numero de filas")
    if len(df_linkable) < 2:
        raise ValueError("Se requieren al menos dos registros para minar pares")

    n_records = len(df_linkable)
    effective_k = min(top_k, n_records - 1)
    record_ids = df_linkable["record_id"].astype(np.int64).to_numpy()
    entity_ids = df_linkable["entity_id"].astype(np.int64).to_numpy()
    sources = df_linkable["source_db"].to_numpy()
    top_indices = np.empty((n_records, effective_k), dtype=np.int32)
    top_scores = np.empty((n_records, effective_k), dtype=np.float32)

    for start in range(0, n_records, batch_size):
        stop = min(start + batch_size, n_records)
        similarities = (vectors[start:stop] @ vectors.T).toarray()
        same_source = sources[start:stop, None] == sources[None, :]
        similarities[same_source] = -np.inf
        local_rows = np.arange(stop - start)
        similarities[local_rows, np.arange(start, stop)] = -np.inf
        indices = np.argpartition(similarities, -effective_k, axis=1)[:, -effective_k:]
        scores = np.take_along_axis(similarities, indices, axis=1)
        top_indices[start:stop] = indices
        top_scores[start:stop] = scores

    top_sets = [set(row.tolist()) for row in top_indices]
    rows = []

    entity_to_indices: dict[int, list[int]] = {}
    for index, entity_id in enumerate(entity_ids):
        entity_to_indices.setdefault(int(entity_id), []).append(index)

    for indices in entity_to_indices.values():
        for left_position in range(len(indices)):
            for right_position in range(left_position + 1, len(indices)):
                left = indices[left_position]
                right = indices[right_position]
                if sources[left] == sources[right]:
                    continue
                retrieved = right in top_sets[left] or left in top_sets[right]
                rows.append(
                    {
                        "record_id_a": int(record_ids[left]),
                        "record_id_b": int(record_ids[right]),
                        "label": 1,
                        "similarity": _cosine(vectors, left, right),
                        "type": "positive" if retrieved else "hard_positive",
                    }
                )

    seen_negative_pairs = set()
    for left in range(n_records):
        for position, right in enumerate(top_indices[left]):
            right = int(right)
            if entity_ids[left] == entity_ids[right] or sources[left] == sources[right]:
                continue
            record_left = int(record_ids[left])
            record_right = int(record_ids[right])
            key = tuple(sorted((record_left, record_right)))
            if key in seen_negative_pairs:
                continue
            seen_negative_pairs.add(key)
            rows.append(
                {
                    "record_id_a": key[0],
                    "record_id_b": key[1],
                    "label": 0,
                    "similarity": float(top_scores[left, position]),
                    "type": "hard_negative",
                }
            )

    pairs = pd.DataFrame(rows)
    return pairs.sort_values(
        ["label", "record_id_a", "record_id_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _cosine(vectors: csr_matrix, left: int, right: int) -> float:
    return float(vectors[left].multiply(vectors[right]).sum())
