"""Ejecuta el baseline clasico completo: TF-IDF retrieve and rerank."""

import argparse
import gc
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

from record_linkage.baselines import ClassicalReranker, TfidfEncoder
from record_linkage.baselines.hard_pair_mining import mine_tfidf_pairs
from record_linkage.config import (
    DEFAULT_VARIANT,
    EVALUATION_DIR,
    SUPPORTED_VARIANTS,
    baseline_run_dir,
    classical_pairs_dir,
    split_path,
)
from record_linkage.evaluation.classical_pipeline_eval import (
    evaluate_classical_pipeline,
    summarize_mined_pairs,
)
from record_linkage.evaluation.tfidf_retrieval_eval import (
    K_VALUES_DEFAULT,
    evaluate_sparse_retrieval,
    find_linkable_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representation_matrices(encoder: TfidfEncoder, texts: list[str]) -> dict:
    word, char = encoder.transform_components(texts)
    return {
        "word": word,
        "char": char,
        "combined": encoder.combine_components(word, char),
    }


def _transform_selected(
    encoder: TfidfEncoder,
    texts: list[str],
    representation: str,
):
    if representation == "word":
        return encoder.transform_word(texts)
    if representation == "char":
        return encoder.transform_char(texts)
    return encoder.transform(texts)


def _validate_entity_splits(records: pd.DataFrame) -> None:
    if records["record_id"].duplicated().any():
        raise ValueError("record_id debe ser unico en el split")
    split_counts = records.groupby("entity_id")["split"].nunique()
    leaking_entities = split_counts[split_counts > 1]
    if len(leaking_entities):
        raise ValueError(
            f"Se detectaron {len(leaking_entities)} entidades presentes en varios splits"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline clasico end-to-end TF-IDF + regresion logistica"
    )
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--run-name", default="tfidf_logreg_e2e")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=50_000)
    parser.add_argument("--char-max-features", type=int, default=50_000)
    parser.add_argument("--word-weight", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1_000)
    parser.add_argument("--n-thresholds", type=int, default=51)
    parser.add_argument("--mining-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = args.dataset or split_path(args.variant)
    if not dataset_path.is_file():
        print(f"ERROR: split no encontrado: {dataset_path}")
        return 1

    print(f"\nCargando split: {dataset_path}")
    records = pd.read_parquet(dataset_path)
    required = {"record_id", "source_db", "entity_id", "text", "split"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Split sin columnas requeridas: {missing}")
    _validate_entity_splits(records)
    print(f"  {len(records):,} registros | {records['entity_id'].nunique():,} entidades")

    train_records = records[records["split"] == "train"]
    encoder = TfidfEncoder(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        min_df=args.min_df,
        word_weight=args.word_weight,
    )
    print(f"\nAjustando TF-IDF con {len(train_records):,} registros de train...")
    started = time.perf_counter()
    encoder.fit(train_records["text"].astype(str).tolist())
    encoder_fit_seconds = time.perf_counter() - started
    vocabularies = encoder.vocabulary_sizes
    print(
        f"  vocabulario word={vocabularies['word']:,} | "
        f"char={vocabularies['char']:,} | {encoder_fit_seconds:.1f}s"
    )

    print("\nSeleccionando representacion de recuperacion sobre validacion...")
    val_linkable = find_linkable_records(records[records["split"] == "val"])
    started = time.perf_counter()
    val_matrices = _representation_matrices(
        encoder,
        val_linkable["text"].astype(str).tolist(),
    )
    retrieval_validation = {
        name: evaluate_sparse_retrieval(val_linkable, matrix, K_VALUES_DEFAULT)
        for name, matrix in val_matrices.items()
    }
    validation_candidate_coverage = {}
    for name, matrix in val_matrices.items():
        validation_pairs = mine_tfidf_pairs(
            val_linkable,
            matrix,
            top_k=args.top_k,
            batch_size=args.mining_batch_size,
        )
        summary = summarize_mined_pairs(validation_pairs)
        n_positives = summary["positive"] + summary["hard_positive"]
        validation_candidate_coverage[name] = {
            **summary,
            "pair_recall_at_k_symmetric": round(summary["positive"] / n_positives, 6),
        }
    selected_representation = max(
        retrieval_validation,
        key=lambda name: (
            validation_candidate_coverage[name]["pair_recall_at_k_symmetric"],
            retrieval_validation[name]["macro_average"]["MRR"],
            retrieval_validation[name]["macro_average"]["delta_separability"],
        ),
    )
    retrieval_validation_seconds = time.perf_counter() - started
    print(
        f"  seleccion={selected_representation} por Recall@K simetrico="
        f"{validation_candidate_coverage[selected_representation]['pair_recall_at_k_symmetric']:.4f} "
        f"y MRR macro={retrieval_validation[selected_representation]['macro_average']['MRR']:.4f} | "
        f"{retrieval_validation_seconds:.1f}s"
    )
    del val_matrices
    gc.collect()

    print("\nEvaluando recuperacion TF-IDF sobre test...")
    test_linkable = find_linkable_records(records[records["split"] == "test"])
    test_texts = test_linkable["text"].astype(str).tolist()
    started = time.perf_counter()
    test_matrices = _representation_matrices(encoder, test_texts)
    retrieval_results = {
        name: evaluate_sparse_retrieval(test_linkable, matrix, K_VALUES_DEFAULT)
        for name, matrix in test_matrices.items()
    }
    retrieval_seconds = time.perf_counter() - started
    selected_macro = retrieval_results[selected_representation]["macro_average"]
    print(
        f"  {selected_representation}: Recall@1={selected_macro['Recall@1']:.4f} | "
        f"Recall@5={selected_macro['Recall@5']:.4f} | "
        f"MRR={selected_macro['MRR']:.4f} | {retrieval_seconds:.1f}s"
    )
    del test_matrices
    gc.collect()

    pairs_dir = classical_pairs_dir(args.variant)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    mined_pairs = {}
    mining_summaries = {}
    mining_timings = {}
    print("\nGenerando candidatos propios de TF-IDF...")
    for split_name in ("train", "val", "test"):
        split_records = find_linkable_records(records[records["split"] == split_name])
        vectors = _transform_selected(
            encoder,
            split_records["text"].astype(str).tolist(),
            selected_representation,
        )
        started = time.perf_counter()
        pairs = mine_tfidf_pairs(
            split_records,
            vectors,
            top_k=args.top_k,
            batch_size=args.mining_batch_size,
        )
        mining_timings[split_name] = round(time.perf_counter() - started, 3)
        output_path = pairs_dir / f"pairs_{split_name}.parquet"
        pairs.to_parquet(output_path, index=False)
        mined_pairs[split_name] = pairs
        mining_summaries[split_name] = summarize_mined_pairs(pairs)
        summary = mining_summaries[split_name]
        print(
            f"  {split_name}: {summary['positive']:,} positive | "
            f"{summary['hard_positive']:,} hard_positive | "
            f"{summary['hard_negative']:,} hard_negative"
        )
        del vectors
        gc.collect()

    train_retrieved = mined_pairs["train"][
        mined_pairs["train"]["type"] != "hard_positive"
    ].copy()
    if not (train_retrieved["label"] == 1).any():
        raise RuntimeError("TF-IDF no recupero positivos en train; no se puede ajustar el reranker")

    print("\nEntrenando reranker clasico con candidatos TF-IDF recuperados...")
    reranker = ClassicalReranker(
        c=args.c,
        max_iter=args.max_iter,
        seed=args.seed,
        encoder=encoder,
    )
    started = time.perf_counter()
    fit_summary = reranker.fit(
        records[["record_id", "text"]],
        train_retrieved,
        fit_encoder=False,
    )
    reranker_fit_seconds = time.perf_counter() - started
    print(
        f"  {fit_summary['n_train_pairs']:,} pares recuperados | "
        f"{reranker_fit_seconds:.1f}s"
    )

    print("\nEvaluando reranker condicional y pipeline end-to-end...")
    started = time.perf_counter()
    pipeline_evaluation, predictions = evaluate_classical_pipeline(
        reranker,
        records[["record_id", "text"]],
        mined_pairs["val"],
        mined_pairs["test"],
        n_thresholds=args.n_thresholds,
    )
    pipeline_eval_seconds = time.perf_counter() - started
    conditional = pipeline_evaluation["test_retrieved_pairs"]
    end_to_end = pipeline_evaluation["test_end_to_end"]
    print(
        f"  condicional: F1={conditional['f1']:.4f} | "
        f"precision={conditional['precision']:.4f} | recall={conditional['recall']:.4f}"
    )
    print(
        f"  end-to-end: F1={end_to_end['f1']:.4f} | "
        f"precision={end_to_end['precision']:.4f} | recall={end_to_end['recall']:.4f}"
    )

    model_dir = baseline_run_dir(args.run_name, args.variant)
    evaluation_dir = EVALUATION_DIR / "baselines" / "end_to_end" / args.variant
    model_path = model_dir / "model.pkl"
    predictions_path = evaluation_dir / f"{args.run_name}_test_predictions.parquet"
    results_path = evaluation_dir / f"{args.run_name}_results.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    reranker.save(model_path)
    predictions.to_parquet(predictions_path, index=False)

    pair_paths = {
        split_name: pairs_dir / f"pairs_{split_name}.parquet"
        for split_name in ("train", "val", "test")
    }
    results = {
        "run_name": args.run_name,
        "task": "classical_tfidf_retrieve_and_rerank",
        "variant": args.variant,
        "protocol": {
            "encoder_fit_split": "train",
            "retrieval_top_k": args.top_k,
            "retrieval_representation_selection": "highest validation symmetric pair Recall@K, then macro MRR",
            "retrieval_representation": selected_representation,
            "candidate_rule": "global top-k after filtering same-source candidates",
            "reranker_train_pairs": "retrieved positives and hard negatives only",
            "threshold_split": "all val positives plus retrieved val negatives",
            "unretrieved_test_positives": "score zero and false negative end-to-end",
            "uses_neural_embeddings": False,
            "uses_neural_similarity": False,
            "uses_neural_hnm_pairs": False,
        },
        "parameters": {
            "word_max_features": args.word_max_features,
            "char_max_features": args.char_max_features,
            "min_df": args.min_df,
            "word_weight": args.word_weight,
            "c": args.c,
            "max_iter": args.max_iter,
            "n_thresholds": args.n_thresholds,
            "seed": args.seed,
        },
        "paths": {
            "dataset": str(dataset_path),
            "model": str(model_path),
            "predictions": str(predictions_path),
            "pairs": {key: str(value) for key, value in pair_paths.items()},
        },
        "input_sha256": {"dataset": _sha256(dataset_path)},
        "generated_pairs_sha256": {
            key: _sha256(value) for key, value in pair_paths.items()
        },
        "output_sha256": {
            "model": _sha256(model_path),
            "predictions": _sha256(predictions_path),
        },
        "vocabulary_sizes": vocabularies,
        "retrieval_validation": retrieval_validation,
        "candidate_coverage_validation": validation_candidate_coverage,
        "retrieval_test": retrieval_results,
        "mining": mining_summaries,
        "reranker_fit": fit_summary,
        "pipeline_evaluation": pipeline_evaluation,
        "timing_seconds": {
            "encoder_fit": round(encoder_fit_seconds, 3),
            "retrieval_validation_all_representations": round(
                retrieval_validation_seconds,
                3,
            ),
            "retrieval_test_all_representations": round(retrieval_seconds, 3),
            "mining": mining_timings,
            "reranker_fit": round(reranker_fit_seconds, 3),
            "pipeline_evaluation": round(pipeline_eval_seconds, 3),
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
