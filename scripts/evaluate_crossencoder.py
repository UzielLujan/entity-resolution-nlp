"""Evaluación del Cross-Encoder sobre un parquet de pares etiquetados.

Métricas: F1, Precision, Recall, Accuracy, PR-AUC, ROC-AUC + matriz de confusión.

Uso:
    python scripts/evaluate_crossencoder.py \\
        --checkpoint ~/Data/INER/models/checkpoints/<variante>_ce/best \\
        --pairs ~/Data/INER/processed/tesis/output/<variante>/pairs_test.parquet

    # Calibrar threshold sobre validación:
    python scripts/evaluate_crossencoder.py \\
        --checkpoint ... --pairs <val>.parquet --find-threshold
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import EVALUATION_DIR
from record_linkage.evaluation.crossencoder_eval import (
    evaluate_crossencoder_checkpoint,
    find_optimal_threshold,
)


CE_EVAL_DIR = EVALUATION_DIR / "crossencoder"


def main():
    parser = argparse.ArgumentParser(description="Evaluación del Cross-Encoder")
    parser.add_argument("--checkpoint", required=True,
                        help="Ruta al checkpoint del CE (con config.json + pytorch_model.bin)")
    parser.add_argument("--dataset", required=True,
                        help="Parquet de registros con cols record_id y text")
    parser.add_argument("--pairs", required=True,
                        help="Parquet de pares (cols: record_id_a, record_id_b, label)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Umbral para binarizar scores (default: 0.5)")
    parser.add_argument("--find-threshold", action="store_true",
                        help="Barre umbrales [0,1] y devuelve el F1-óptimo (sin evaluar)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--output-name", type=str, default=None,
                        help="Nombre del JSON de salida (default: derivado del checkpoint)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    pairs_path = Path(args.pairs)
    dataset_path = Path(args.dataset)

    for p in (checkpoint_path, pairs_path, dataset_path):
        if not p.exists():
            print(f"ERROR: no encontrado: {p}")
            return 1

    if args.find_threshold:
        result = find_optimal_threshold(
            checkpoint_path, pairs_path, dataset_path,
            batch_size=args.batch_size, max_length=args.max_seq_length,
        )
        print(f"\nThreshold óptimo: {result['best_threshold']} (F1={result['best_f1']:.4f})")
        return 0

    output = evaluate_crossencoder_checkpoint(
        checkpoint_path=checkpoint_path,
        pairs_path=pairs_path,
        dataset_path=dataset_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
    )

    CE_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    name = args.output_name or f"ce_results_{checkpoint_path.parent.name}.json"
    out_path = CE_EVAL_DIR / name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Resultados guardados en {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
