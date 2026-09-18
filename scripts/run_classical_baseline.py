"""Entrena y evalua el baseline clasico del reranker sobre pares HNM.

El vocabulario y el clasificador se ajustan con train. El umbral se selecciona
en validacion y se congela antes de evaluar test.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from record_linkage.baselines import ClassicalReranker, select_f1_threshold
from record_linkage.config import (
    DEFAULT_VARIANT,
    EVALUATION_DIR,
    SUPPORTED_VARIANTS,
    baseline_run_dir,
    pairs_path,
    prepared_dataset_path,
)
from record_linkage.evaluation.metrics import compute_binary_classification_metrics


def _score_stats(scores: np.ndarray, labels: np.ndarray) -> dict:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    return {
        "mean_positives": float(positives.mean()),
        "std_positives": float(positives.std()),
        "mean_negatives": float(negatives.mean()),
        "std_negatives": float(negatives.std()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pairs(path: Path, split: str) -> pd.DataFrame:
    pairs = pd.read_parquet(path)
    required = {"record_id_a", "record_id_b", "label"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"{split} no contiene columnas requeridas: {missing}")
    print(
        f"  {split}: {len(pairs):,} pares | "
        f"{int((pairs['label'] == 1).sum()):,} positivos | "
        f"{int((pairs['label'] == 0).sum()):,} negativos"
    )
    return pairs


def _prediction_frame(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    predictions = (scores >= threshold).astype(np.int8)
    result = pairs[["record_id_a", "record_id_b", "label"]].copy()
    result["score"] = scores.astype(np.float32)
    result["prediction"] = predictions
    result["word_cosine"] = diagnostics["word_cosine"]
    result["char_cosine"] = diagnostics["char_cosine"]
    result["error_type"] = np.select(
        (
            (result["label"] == 0) & (result["prediction"] == 1),
            (result["label"] == 1) & (result["prediction"] == 0),
        ),
        ("false_positive", "false_negative"),
        default="correct",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baseline TF-IDF + regresion logistica para el reranker"
    )
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--pairs-train", type=Path, default=None)
    parser.add_argument("--pairs-val", type=Path, default=None)
    parser.add_argument("--pairs-test", type=Path, default=None)
    parser.add_argument("--run-name", default="tfidf_logreg_hnm")
    parser.add_argument("--word-max-features", type=int, default=50_000)
    parser.add_argument("--char-max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--n-thresholds", type=int, default=51)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = args.dataset or prepared_dataset_path(args.variant)
    train_path = args.pairs_train or pairs_path("train", args.variant)
    val_path = args.pairs_val or pairs_path("val", args.variant)
    test_path = args.pairs_test or pairs_path("test", args.variant)
    for path in (dataset, train_path, val_path, test_path):
        if not path.is_file():
            print(f"ERROR: archivo no encontrado: {path}")
            return 1

    print(f"\nCargando registros: {dataset}")
    records = pd.read_parquet(dataset, columns=["record_id", "text"])
    print(f"  {len(records):,} registros")
    print("Cargando pares HNM:")
    train_pairs = _load_pairs(train_path, "train")
    val_pairs = _load_pairs(val_path, "val")
    test_pairs = _load_pairs(test_path, "test")

    baseline = ClassicalReranker(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        min_df=args.min_df,
        c=args.c,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    print("\nAjustando TF-IDF y regresion logistica...")
    started = time.perf_counter()
    fit_summary = baseline.fit(records, train_pairs)
    fit_seconds = time.perf_counter() - started
    print(
        f"  vocabulario: {fit_summary['word_vocabulary_size']:,} word | "
        f"{fit_summary['char_vocabulary_size']:,} char"
    )
    print(f"  features train: {fit_summary['pair_feature_shape']} | {fit_seconds:.1f}s")

    print("\nSeleccionando umbral sobre validacion...")
    started = time.perf_counter()
    val_scores, val_diagnostics = baseline.score_pairs(records, val_pairs)
    val_seconds = time.perf_counter() - started
    val_labels = val_pairs["label"].astype(int).to_numpy()
    threshold_result = select_f1_threshold(
        val_scores,
        val_labels,
        n_thresholds=args.n_thresholds,
    )
    threshold = threshold_result["best_threshold"]
    val_metrics = compute_binary_classification_metrics(
        val_scores,
        val_labels,
        threshold=threshold,
    )
    print(f"  threshold={threshold:.4f} | F1={val_metrics['f1']:.4f} | {val_seconds:.1f}s")

    print("\nEvaluando test con umbral congelado...")
    started = time.perf_counter()
    test_scores, test_diagnostics = baseline.score_pairs(records, test_pairs)
    test_seconds = time.perf_counter() - started
    test_labels = test_pairs["label"].astype(int).to_numpy()
    test_metrics = compute_binary_classification_metrics(
        test_scores,
        test_labels,
        threshold=threshold,
    )
    print(
        f"  F1={test_metrics['f1']:.4f} | precision={test_metrics['precision']:.4f} | "
        f"recall={test_metrics['recall']:.4f}"
    )
    print(
        f"  TP={test_metrics['tp']} FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} TN={test_metrics['tn']} | {test_seconds:.1f}s"
    )

    lexical_diagnostics = {}
    for name in ("word_cosine", "char_cosine"):
        lexical_threshold = select_f1_threshold(
            val_diagnostics[name],
            val_labels,
            n_thresholds=args.n_thresholds,
        )
        lexical_test_metrics = compute_binary_classification_metrics(
            test_diagnostics[name],
            test_labels,
            threshold=lexical_threshold["best_threshold"],
        )
        lexical_diagnostics[name] = {
            "threshold_selection_on_validation": lexical_threshold,
            "test_metrics": lexical_test_metrics,
        }
        print(
            f"  diagnostico {name}: threshold={lexical_threshold['best_threshold']:.4f} | "
            f"test F1={lexical_test_metrics['f1']:.4f}"
        )

    model_dir = baseline_run_dir(args.run_name, args.variant)
    evaluation_dir = EVALUATION_DIR / "baselines" / "reranker" / args.variant
    model_path = model_dir / "model.pkl"
    results_path = evaluation_dir / f"{args.run_name}_results.json"
    predictions_path = evaluation_dir / f"{args.run_name}_test_predictions.parquet"
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(model_path)
    _prediction_frame(
        test_pairs,
        test_scores,
        test_diagnostics,
        threshold,
    ).to_parquet(predictions_path, index=False)

    results = {
        "run_name": args.run_name,
        "task": "classical_reranker_on_hnm_candidates",
        "variant": args.variant,
        "method": {
            "record_representation": "TF-IDF word (1,2) and char_wb (3,5)",
            "pair_features": "abs_diff(word), product(word), cosine(word), cosine(char)",
            "classifier": "logistic_regression",
            "class_weight": "balanced",
            "uses_biencoder_similarity_feature": False,
        },
        "parameters": {
            "word_max_features": args.word_max_features,
            "char_max_features": args.char_max_features,
            "min_df": args.min_df,
            "c": args.c,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
        "paths": {
            "dataset": str(dataset),
            "pairs_train": str(train_path),
            "pairs_val": str(val_path),
            "pairs_test": str(test_path),
            "model": str(model_path),
            "test_predictions": str(predictions_path),
        },
        "input_sha256": {
            "dataset": _sha256(dataset),
            "pairs_train": _sha256(train_path),
            "pairs_val": _sha256(val_path),
            "pairs_test": _sha256(test_path),
        },
        "fit_summary": fit_summary,
        "threshold_selection": threshold_result,
        "validation": {
            "metrics": val_metrics,
            "score_stats": _score_stats(val_scores, val_labels),
            "mean_word_cosine": float(val_diagnostics["word_cosine"].mean()),
            "mean_char_cosine": float(val_diagnostics["char_cosine"].mean()),
        },
        "test": {
            "metrics": test_metrics,
            "score_stats": _score_stats(test_scores, test_labels),
            "mean_word_cosine": float(test_diagnostics["word_cosine"].mean()),
            "mean_char_cosine": float(test_diagnostics["char_cosine"].mean()),
        },
        "lexical_diagnostics": lexical_diagnostics,
        "timing_seconds": {
            "fit": round(fit_seconds, 3),
            "validation_scoring": round(val_seconds, 3),
            "test_scoring": round(test_seconds, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f"\nModelo: {model_path}")
    print(f"Resultados: {results_path}")
    print(f"Predicciones: {predictions_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
