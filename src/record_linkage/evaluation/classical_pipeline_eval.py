"""Evaluacion condicional y end-to-end del pipeline clasico."""

from __future__ import annotations

import numpy as np
import pandas as pd

from record_linkage.baselines.classical_reranker import (
    ClassicalReranker,
    select_f1_threshold,
)
from record_linkage.evaluation.metrics import compute_binary_classification_metrics


def evaluate_classical_pipeline(
    model: ClassicalReranker,
    records_df: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    n_thresholds: int = 51,
) -> tuple[dict, pd.DataFrame]:
    """Selecciona umbral end-to-end en validacion y evalua el flujo completo.

    Los ``hard_positive`` no llegan al reranker. En la evaluacion end-to-end se
    les asigna score cero y cuentan como falsos negativos de recuperacion.
    """
    val_retrieved = val_pairs[val_pairs["type"] != "hard_positive"].copy()
    test_retrieved = test_pairs[test_pairs["type"] != "hard_positive"].copy()
    if set(val_retrieved["label"].unique()) != {0, 1}:
        raise ValueError("Validacion recuperada debe contener positivos y negativos")

    val_scores_retrieved, _ = model.score_pairs(records_df, val_retrieved)
    val_scores = np.zeros(len(val_pairs), dtype=float)
    val_retrieved_mask = val_pairs["type"] != "hard_positive"
    val_scores[val_retrieved_mask.to_numpy()] = val_scores_retrieved
    val_labels = val_pairs["label"].astype(int).to_numpy()
    threshold_selection = select_f1_threshold(
        val_scores,
        val_labels,
        n_thresholds=n_thresholds,
    )
    threshold = threshold_selection["best_threshold"]
    val_conditional_metrics = compute_binary_classification_metrics(
        val_scores_retrieved,
        val_retrieved["label"].astype(int).to_numpy(),
        threshold=threshold,
    )
    val_end_to_end_metrics = _end_to_end_metrics(val_scores, val_labels, threshold)

    test_scores, diagnostics = model.score_pairs(records_df, test_retrieved)
    test_labels = test_retrieved["label"].astype(int).to_numpy()
    conditional_metrics = compute_binary_classification_metrics(
        test_scores,
        test_labels,
        threshold=threshold,
    )

    predictions = test_pairs[
        ["record_id_a", "record_id_b", "label", "similarity", "type"]
    ].copy()
    predictions["score"] = 0.0
    predictions["word_cosine"] = 0.0
    predictions["char_cosine"] = 0.0
    retrieved_index = predictions["type"] != "hard_positive"
    predictions.loc[retrieved_index, "score"] = test_scores
    predictions.loc[retrieved_index, "word_cosine"] = diagnostics["word_cosine"]
    predictions.loc[retrieved_index, "char_cosine"] = diagnostics["char_cosine"]
    predictions["prediction"] = (predictions["score"] >= threshold).astype(np.int8)
    predictions.loc[~retrieved_index, "prediction"] = 0
    predictions["error_type"] = np.select(
        (
            (predictions["label"] == 0) & (predictions["prediction"] == 1),
            (predictions["label"] == 1) & (predictions["prediction"] == 0),
        ),
        ("false_positive", "false_negative"),
        default="correct",
    )

    if threshold <= 0:
        raise ValueError("El umbral end-to-end debe ser mayor que cero")
    end_to_end_metrics = _end_to_end_metrics(
        predictions["score"].to_numpy(),
        predictions["label"].astype(int).to_numpy(),
        threshold,
    )
    hard_positive_count = int((test_pairs["type"] == "hard_positive").sum())
    hard_positive_errors = int(
        (
            (predictions["type"] == "hard_positive")
            & (predictions["error_type"] == "false_negative")
        ).sum()
    )

    return {
        "threshold_selection": threshold_selection,
        "validation_retrieved_pairs": val_conditional_metrics,
        "validation_end_to_end": val_end_to_end_metrics,
        "test_retrieved_pairs": conditional_metrics,
        "test_end_to_end": end_to_end_metrics,
        "test_hard_positives": hard_positive_count,
        "test_false_negatives_from_retrieval": hard_positive_errors,
    }, predictions


def _end_to_end_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Métricas válidas sin enumerar los negativos fuera del conjunto candidato."""
    metrics = compute_binary_classification_metrics(scores, labels, threshold=threshold)
    return {
        key: metrics[key]
        for key in ("threshold", "f1", "precision", "recall", "tp", "fp", "fn")
    }


def summarize_mined_pairs(pairs: pd.DataFrame) -> dict:
    counts = pairs["type"].value_counts().to_dict()
    return {
        "n_pairs": int(len(pairs)),
        "positive": int(counts.get("positive", 0)),
        "hard_positive": int(counts.get("hard_positive", 0)),
        "hard_negative": int(counts.get("hard_negative", 0)),
    }
