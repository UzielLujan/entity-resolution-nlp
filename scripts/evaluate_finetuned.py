"""Evaluación del Bi-Encoder fine-tuneado sobre el split de test.

Métricas por par direccional de bases (A→B y B→A — permutations):
  - Hit@K, Recall@K, RecallNorm@K, Precision@K, MRR
  - Δseparabilidad (μ_pos, μ_neg, σ_pos, σ_neg)
  - candidate_pool_stats: max/mean positives por entidad en el pool

Uso:
    python scripts/evaluate_finetuned.py --checkpoint beto_mnrl_hpc_run_e
    python scripts/evaluate_finetuned.py --checkpoint beto_mnrl_hpc_run_e --epoch 15
    python scripts/evaluate_finetuned.py --checkpoint beto_mnrl_hpc_run_e --split val
    python scripts/evaluate_finetuned.py --checkpoint A B C   (varios runs)
    python scripts/evaluate_finetuned.py --all

Salida: ~/Data/INER/outputs/evaluation/finetuned/finetuned_results_<run>.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import EVALUATION_DIR, PROCESSED_DIR
from record_linkage.evaluation.biencoder_eval import (
    K_VALUES_DEFAULT,
    evaluate_finetuned_checkpoint,
    list_available_checkpoints,
)


FINETUNED_DIR = EVALUATION_DIR / "finetuned"


def _print_summary(checkpoints: list[str], output_dir: Path) -> None:
    """Tabla resumen Hit@1 / Recall@1 / RecallNorm@1 / Precision@1 / MRR por par."""
    print("\n" + "=" * 60)
    print("RESUMEN — Hit@1, Recall@1, RecallNorm@1, Precision@1, MRR")
    print("=" * 60)
    for ckpt_name in checkpoints:
        out_path = output_dir / f"finetuned_results_{ckpt_name}.json"
        if not out_path.exists():
            continue
        data = json.load(open(out_path))
        print(f"\n{ckpt_name}  (val_loss={data.get('best_val_loss', '?')})")
        for pair_key, pair_results in data["results"].items():
            print(f"  {pair_key:<40}"
                  f"  Hit@1={pair_results.get('Hit@1', 0):.4f}"
                  f"  Rec@1={pair_results.get('Recall@1', 0):.4f}"
                  f"  RecN@1={pair_results.get('RecallNorm@1', 0):.4f}"
                  f"  Prec@1={pair_results.get('Precision@1', 0):.4f}"
                  f"  MRR={pair_results.get('MRR', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluación del Bi-Encoder fine-tuneado")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", nargs="+", dest="checkpoints", metavar="RUN_NAME",
                       help="Nombre(s) del run en checkpoints/. Acepta uno o varios.")
    group.add_argument("--all", action="store_true",
                       help="Evalúa todos los checkpoints que tengan best/")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Evaluar epoch_XX específica en lugar de best/")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Split a evaluar (default: test)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Ruta al dataset_split.parquet (default: tesis1/dataset_split.parquet)")
    args = parser.parse_args()

    dataset_path = (
        Path(args.dataset) if args.dataset
        else PROCESSED_DIR / "tesis1" / "dataset_split.parquet"
    )
    if not dataset_path.exists():
        print(f"ERROR: dataset no encontrado en {dataset_path}")
        return 1

    checkpoints = list_available_checkpoints() if args.all else args.checkpoints
    if not checkpoints:
        print(f"ERROR: no se encontraron checkpoints con best/")
        return 1
    if args.all:
        print(f"Checkpoints encontrados: {checkpoints}")

    FINETUNED_DIR.mkdir(parents=True, exist_ok=True)

    for ckpt_name in checkpoints:
        output = evaluate_finetuned_checkpoint(
            checkpoint_name=ckpt_name,
            dataset_path=dataset_path,
            split=args.split,
            epoch=args.epoch,
            k_values=K_VALUES_DEFAULT,
        )
        out_path = FINETUNED_DIR / f"finetuned_results_{ckpt_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Resultados guardados en {out_path}")

    if len(checkpoints) > 1:
        _print_summary(checkpoints, FINETUNED_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
