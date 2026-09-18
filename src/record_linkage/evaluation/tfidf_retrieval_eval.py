"""Evaluacion de recuperacion para representaciones TF-IDF dispersas."""

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from record_linkage.evaluation.metrics import candidate_pool_stats


K_VALUES_DEFAULT = [1, 5, 10, 20, 50]


def find_linkable_records(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva entidades presentes en al menos dos fuentes."""
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    linkable_ids = entity_sources[entity_sources > 1].index
    return df[df["entity_id"].isin(linkable_ids)].copy().reset_index(drop=True)


def evaluate_sparse_retrieval(
    df_linkable: pd.DataFrame,
    vectors: csr_matrix,
    k_values: list[int] | None = None,
) -> dict:
    """Evalua las seis direcciones entre fuentes mediante similitud coseno."""
    k_values = k_values or K_VALUES_DEFAULT
    if len(df_linkable) != vectors.shape[0]:
        raise ValueError("df_linkable y vectors deben tener el mismo numero de filas")

    entity_ids = df_linkable["entity_id"].to_numpy()
    source_dbs = df_linkable["source_db"].to_numpy()
    sources = sorted(df_linkable["source_db"].unique())
    results = {}

    for source_a, source_b in permutations(sources, 2):
        mask_a = source_dbs == source_a
        mask_b = source_dbs == source_b
        ids_a = entity_ids[mask_a]
        ids_b = entity_ids[mask_b]
        shared_entities = np.intersect1d(ids_a, ids_b)
        query_mask = np.isin(ids_a, shared_entities)
        query_ids = ids_a[query_mask]
        if len(query_ids) == 0:
            continue

        query_vectors = vectors[mask_a][query_mask]
        candidate_vectors = vectors[mask_b]
        similarity = (query_vectors @ candidate_vectors.T).toarray()
        pair_metrics = _metrics_from_similarity(
            similarity,
            query_ids,
            ids_b,
            k_values,
        )
        pair_metrics["candidate_pool_stats"] = candidate_pool_stats(ids_b)
        results[f"{source_a} -> {source_b}"] = pair_metrics

    return {
        "results": results,
        "macro_average": _macro_average(results, k_values),
    }


def _metrics_from_similarity(
    similarity: np.ndarray,
    query_entity_ids: np.ndarray,
    candidate_entity_ids: np.ndarray,
    k_values: list[int],
) -> dict:
    max_k = min(max(k_values), similarity.shape[1])
    hits = {k: 0 for k in k_values}
    recall = {k: 0.0 for k in k_values}
    recall_norm = {k: 0.0 for k in k_values}
    precision = {k: 0.0 for k in k_values}
    reciprocal_ranks = []
    positive_sims = []
    negative_sims = []

    for row, true_entity in zip(similarity, query_entity_ids):
        positive_mask = candidate_entity_ids == true_entity
        n_positives = int(positive_mask.sum())
        positive_sims.extend(row[positive_mask].tolist())
        negative_sims.extend(row[~positive_mask].tolist())

        top_indices = np.argpartition(row, -max_k)[-max_k:]
        top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
        sorted_entities = candidate_entity_ids[np.argsort(row)[::-1]]
        first_positive = np.flatnonzero(sorted_entities == true_entity)[0]
        reciprocal_ranks.append(1.0 / (first_positive + 1))

        for k in k_values:
            effective_k = min(k, len(top_indices))
            top_entities = candidate_entity_ids[top_indices[:effective_k]]
            n_found = int((top_entities == true_entity).sum())
            hits[k] += int(n_found > 0)
            recall[k] += n_found / n_positives
            recall_norm[k] += n_found / min(k, n_positives)
            precision[k] += n_found / k

    n_queries = len(query_entity_ids)
    result = {"n_queries": n_queries}
    for k in k_values:
        result[f"Hit@{k}"] = round(hits[k] / n_queries, 4)
        result[f"Recall@{k}"] = round(recall[k] / n_queries, 4)
        result[f"RecallNorm@{k}"] = round(recall_norm[k] / n_queries, 4)
        result[f"Precision@{k}"] = round(precision[k] / n_queries, 4)
    result["MRR"] = round(float(np.mean(reciprocal_ranks)), 4)
    result["space_metrics"] = _space_metrics(positive_sims, negative_sims)
    return result


def _space_metrics(positive_sims: list[float], negative_sims: list[float]) -> dict:
    positives = np.asarray(positive_sims)
    negatives = np.asarray(negative_sims)
    mu_pos = float(positives.mean())
    mu_neg = float(negatives.mean())
    sigma_pos = float(positives.std())
    sigma_neg = float(negatives.std())
    pooled_std = float(np.sqrt((sigma_pos**2 + sigma_neg**2) / 2))
    delta = (mu_pos - mu_neg) / pooled_std if pooled_std else 0.0
    return {
        "mu_pos": round(mu_pos, 4),
        "mu_neg": round(mu_neg, 4),
        "sigma_pos": round(sigma_pos, 4),
        "sigma_neg": round(sigma_neg, 4),
        "delta_separability": round(delta, 4),
    }


def _macro_average(results: dict, k_values: list[int]) -> dict:
    keys = ["MRR"]
    for k in k_values:
        keys.extend((f"Hit@{k}", f"Recall@{k}", f"RecallNorm@{k}", f"Precision@{k}"))
    averages = {
        key: round(float(np.mean([metrics[key] for metrics in results.values()])), 4)
        for key in keys
    }
    averages["delta_separability"] = round(
        float(
            np.mean(
                [metrics["space_metrics"]["delta_separability"] for metrics in results.values()]
            )
        ),
        4,
    )
    return averages
