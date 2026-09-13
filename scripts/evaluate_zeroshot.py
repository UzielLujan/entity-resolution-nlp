"""Evaluación zero-shot del Bi-Encoder sobre pares cross-database del INER.

Mismas 4 métricas + MRR + Δseparabilidad que evaluate_finetuned.py, pero sobre
modelos preentrenados sin fine-tuning.

Uso:
    .venv/bin/python scripts/evaluate_zeroshot.py --model BETO
    .venv/bin/python scripts/evaluate_zeroshot.py --all --split test

Dataset por defecto: $INER_DATA_ROOT/modeling/data/notok_skipnull/split.parquet
Salida: $INER_DATA_ROOT/modeling/outputs/evaluation/biencoder/zeroshot/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import EVALUATION_DIR, split_path
from record_linkage.evaluation.biencoder_eval import (
    K_VALUES_DEFAULT,
    evaluate_zeroshot_model,
)


KNOWN_MODELS = [
    "BETO",
    "RoBERTa-biomedical",
    "paraphrase-multilingual",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluación zero-shot del Bi-Encoder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", action="append", dest="models",
                       help="Nombre del modelo en models/pretrained/. Repetir para varios.")
    group.add_argument("--all", action="store_true",
                       help="Evalúa todos los modelos conocidos")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Ruta al split.parquet (default: notok_skipnull/split.parquet)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test",
                        help="Split a evaluar (default: test; all = dataset completo)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size para encoding (default: 32)")
    args = parser.parse_args()

    dataset_path = (
        Path(args.dataset) if args.dataset
        else split_path("notok_skipnull")
    )
    if not dataset_path.exists():
        print(f"ERROR: dataset no encontrado en {dataset_path}")
        print("  El dataset lo produce la consultoría (repo consultoria-iner).")
        return 1

    models_to_eval = KNOWN_MODELS if args.all else args.models
    split = None if args.split == "all" else args.split

    output_dir = EVALUATION_DIR / "biencoder" / "zeroshot"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_model_results = {}
    for model_name in models_to_eval:
        results = evaluate_zeroshot_model(
            model_name=model_name,
            dataset_path=dataset_path,
            split=split,
            k_values=K_VALUES_DEFAULT,
            batch_size=args.batch_size,
        )
        if results:
            all_model_results[model_name] = results

    if not all_model_results:
        print("\nNingún modelo se pudo evaluar.")
        return 1

    suffix = f"{Path(dataset_path).parent.name}_{args.split}"
    output_path = output_dir / f"zeroshot_results_{suffix}.json"
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
