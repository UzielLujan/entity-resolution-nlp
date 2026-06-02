"""Evaluación zero-shot del Bi-Encoder sobre pares cross-database del INER.

Mismas 4 métricas + MRR + Δseparabilidad que evaluate_finetuned.py, pero sobre
modelos preentrenados (sin fine-tuning) y sobre todo el dataset (sin splits).

Uso:
    python scripts/evaluate_zeroshot.py --model BETO
    python scripts/evaluate_zeroshot.py --model RoBERTa-biomedical
    python scripts/evaluate_zeroshot.py --model paraphrase-multilingual
    python scripts/evaluate_zeroshot.py --model BETO --model RoBERTa-biomedical
    python scripts/evaluate_zeroshot.py --all

Dataset por defecto: ~/Data/INER/processed/tesis0_sin_tokens/dataset.parquet
Salida:              ~/Data/INER/outputs/evaluation/zeroshot_results_<perfil>.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import EVALUATION_DIR, PROCESSED_DIR
from record_linkage.evaluation.biencoder_eval import (
    K_VALUES_DEFAULT,
    evaluate_zeroshot_model,
)


KNOWN_MODELS = [
    "BETO",
    "RoBERTa-biomedical",
    "paraphrase-multilingual",
    # RoBERTa-bne-sts pendiente: pre-entrenado para clasificación, no similitud.
    # Discutir con asesor antes de incluir.
    "RoBERTa-bne-sts",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluación zero-shot del Bi-Encoder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", action="append", dest="models",
                       help="Nombre del modelo en models/pretrained/. Repetir para varios.")
    group.add_argument("--all", action="store_true",
                       help="Evalúa todos los modelos conocidos")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Ruta al dataset.parquet (default: tesis0_sin_tokens/dataset.parquet)")
    args = parser.parse_args()

    dataset_path = (
        Path(args.dataset) if args.dataset
        else PROCESSED_DIR / "tesis0_sin_tokens" / "dataset.parquet"
    )
    if not dataset_path.exists():
        print(f"ERROR: dataset no encontrado en {dataset_path}")
        print("Genera primero: python scripts/run_dataset.py --perfil tesis0 --no-special-tokens")
        return 1

    models_to_eval = KNOWN_MODELS if args.all else args.models

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    all_model_results = {}
    for model_name in models_to_eval:
        results = evaluate_zeroshot_model(
            model_name=model_name,
            dataset_path=dataset_path,
            k_values=K_VALUES_DEFAULT,
        )
        if results:
            all_model_results[model_name] = results

    if not all_model_results:
        print("\nNingún modelo se pudo evaluar.")
        return 1

    suffix = Path(dataset_path).parent.name
    output_path = EVALUATION_DIR / f"zeroshot_results_{suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Resultados guardados en {output_path}")

    # Resumen compacto
    print("\n" + "=" * 60)
    print("RESUMEN — Hit@1, Recall@1, MRR (zero-shot)")
    print("=" * 60)
    for model_name, model_results in all_model_results.items():
        print(f"\n{model_name}")
        for pair_key, pair_results in model_results.items():
            print(f"  {pair_key:<40}"
                  f"  Hit@1={pair_results.get('Hit@1', 0):.4f}"
                  f"  Rec@1={pair_results.get('Recall@1', 0):.4f}"
                  f"  MRR={pair_results.get('MRR', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
