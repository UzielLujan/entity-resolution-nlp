"""Funciones puras de evaluación.

Cubre métricas para:
  - Ranking/retrieval (Bi-Encoder):  Hit@K, Recall@K, RecallNorm@K, Precision@K, MRR
  - Clasificación binaria (Cross-Encoder): F1, Precision, Recall, PR-AUC (vendrán cuando se
    implemente la Etapa 2).
  - Análisis del espacio métrico: Δseparabilidad y momentos de la distribución de similitudes.

Definiciones de las métricas de ranking (sobre una sola query con `n_pos` positivos en el pool):

  Hit@K        = 1 si al menos 1 positivo cae en top-K, 0 si no.            (binario)
  Recall@K     = |positivos en top-K| / n_pos                                (cobertura)
  RecallNorm@K = |positivos en top-K| / min(K, n_pos)                        (eficiencia)
  Precision@K  = |positivos en top-K| / K                                    (pureza)
  MRR          = 1 / posición del primer positivo en el ranking completo

Las métricas se reportan promediadas sobre todas las queries del split.
"""

import numpy as np


def candidate_pool_stats(entity_ids: np.ndarray) -> dict:
    """Frecuencia de cada entity_id en el pool de candidatos.

    Útil para interpretar Recall@K estándar: cuando hay entidades con múltiples
    registros en el pool, K=1 no puede recuperarlos todos aunque el modelo sea perfecto.
    """
    unique, counts = np.unique(entity_ids, return_counts=True)
    return {
        "n_entities": int(len(unique)),
        "max_positives_per_entity": int(counts.max()),
        "mean_positives_per_entity": round(float(counts.mean()), 2),
        "entities_with_multiple_records": int((counts > 1).sum()),
    }


def compute_metrics_at_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_entity_ids: np.ndarray,
    candidate_entity_ids: np.ndarray,
    k_values: list,
    negative_sample_size: int = 10,
) -> dict:
    """Hit@K, Recall@K, RecallNorm@K, Precision@K, MRR y métricas del espacio métrico.

    Args:
        query_embeddings:      (n_q, dim) embeddings normalizados de las queries.
        candidate_embeddings:  (n_c, dim) embeddings normalizados del pool candidato.
        query_entity_ids:      (n_q,) entity_id de cada query.
        candidate_entity_ids:  (n_c,) entity_id de cada candidato.
        k_values:              lista de K a reportar (ej. [1, 5, 10, 20, 50]).
        negative_sample_size:  cuántas similitudes negativas muestrear por query (para Δsep).

    Returns:
        dict con keys: Hit@K, Recall@K, RecallNorm@K, Precision@K (uno por K), MRR,
        n_queries, y opcionalmente space_metrics (μ_pos, μ_neg, σ_pos, σ_neg, Δsep).
    """
    sim_matrix = query_embeddings @ candidate_embeddings.T
    max_k = max(k_values)

    hits          = {k: 0   for k in k_values}
    recall_std    = {k: 0.0 for k in k_values}
    recall_norm   = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    reciprocal_ranks = []
    positive_sims = []
    negative_sims_sample = []

    n_queries = len(query_entity_ids)

    for i in range(n_queries):
        sims = sim_matrix[i]
        true_entity = query_entity_ids[i]

        positive_mask = candidate_entity_ids == true_entity
        n_positives = int(positive_mask.sum())
        if n_positives == 0:
            continue

        positive_sims.extend(sims[positive_mask].tolist())
        neg_sims = sims[~positive_mask]
        if len(neg_sims) > 0:
            # TODO(revisión post-separación de repos): este submuestreo uniforme sin seed
            # puede inflar Δsep al diluir hard negatives y hace la métrica no determinista.
            # Decidir entre usar todos los negativos o fijar seed+K. Ver docs/Anexos/metricas_evaluacion.md §4.2.1.
            sample_idx = np.random.choice(
                len(neg_sims), min(negative_sample_size, len(neg_sims)), replace=False
            )
            negative_sims_sample.extend(neg_sims[sample_idx].tolist())

        # Top-K por similitud descendente
        top_k_indices = np.argpartition(sims, -max_k)[-max_k:]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]

        # MRR — posición del primer positivo en el ranking completo
        sorted_eids = candidate_entity_ids[np.argsort(sims)[::-1]]
        first_pos = np.where(sorted_eids == true_entity)[0]
        if len(first_pos) > 0:
            reciprocal_ranks.append(1.0 / (first_pos[0] + 1))

        for k in k_values:
            top_k_eids = candidate_entity_ids[top_k_indices[:k]]
            n_pos_in_topk = int((top_k_eids == true_entity).sum())
            hits[k]          += 1 if n_pos_in_topk > 0 else 0
            recall_std[k]    += n_pos_in_topk / n_positives
            recall_norm[k]   += n_pos_in_topk / min(k, n_positives)
            precision_sum[k] += n_pos_in_topk / k

    results = {}
    for k in k_values:
        results[f"Hit@{k}"]        = round(hits[k]          / n_queries, 4)
        results[f"Recall@{k}"]     = round(recall_std[k]    / n_queries, 4)
        results[f"RecallNorm@{k}"] = round(recall_norm[k]   / n_queries, 4)
        results[f"Precision@{k}"]  = round(precision_sum[k] / n_queries, 4)

    results["MRR"]       = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0
    results["n_queries"] = n_queries

    if positive_sims and negative_sims_sample:
        results["space_metrics"] = _space_metrics(positive_sims, negative_sims_sample)

    return results


def compute_binary_classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """F1, Precision, Recall, PR-AUC, ROC-AUC, accuracy + confusion matrix.

    Métricas estándar de clasificación binaria — útil para evaluar el Cross-Encoder
    y la salida final del pipeline E2E.

    Args:
        scores:    (n,) probabilidades en [0, 1] del modelo (post-sigmoid).
        labels:    (n,) ground truth en {0, 1}.
        threshold: umbral para convertir scores → decisiones binarias.

    Returns:
        dict con: f1, precision, recall, accuracy, pr_auc, roc_auc, tp, fp, fn, tn.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    preds = (scores >= threshold).astype(int)
    labels = labels.astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "f1":        round(float(f1_score(labels, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds, zero_division=0)), 4),
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "pr_auc":    round(float(average_precision_score(labels, scores)), 4),
        "roc_auc":   round(float(roc_auc_score(labels, scores)), 4) if len(set(labels)) > 1 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(labels)),
    }


def _space_metrics(positive_sims: list, negative_sims: list) -> dict:
    """μ_pos, μ_neg, σ_pos, σ_neg y Δseparabilidad (Cohen's d).

    Δseparabilidad = (μ_pos - μ_neg) / σ_pooled
    Valores > 2 ya indican espacios bien separados; valores > 10 muy estructurados.
    """
    mu_pos     = float(np.mean(positive_sims))
    mu_neg     = float(np.mean(negative_sims))
    sigma_pos  = float(np.std(positive_sims))
    sigma_neg  = float(np.std(negative_sims))
    pooled_std = float(np.sqrt((sigma_pos**2 + sigma_neg**2) / 2))
    delta = (mu_pos - mu_neg) / pooled_std if pooled_std > 0 else 0.0
    return {
        "mu_pos":             round(mu_pos, 4),
        "mu_neg":             round(mu_neg, 4),
        "sigma_pos":          round(sigma_pos, 4),
        "sigma_neg":          round(sigma_neg, 4),
        "delta_separability": round(delta, 4),
    }
