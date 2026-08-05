"""
Script de partición train/val/test del dataset a nivel de entidad.

Uso:
    python scripts/run_splitting.py
    python scripts/run_splitting.py --variant notok_skipnull
    python scripts/run_splitting.py --train 0.70 --val 0.15 --seed 42

El dataset de entrada es el que produce la consultoría:
    <DATA_ROOT>/processed/default/output/<variant>/dataset.parquet
La salida es un artefacto de la tesis:
    <DATA_ROOT>/tesis/splits/<variant>_split.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import CONSULTORIA_OUTPUT_DIR, DEFAULT_VARIANT, SPLITS_DIR
from record_linkage.data.splitting import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Partición train/val/test del dataset")
    parser.add_argument(
        "--variant",
        choices=["tok_skipnull", "tok_keepnull", "notok_skipnull", "notok_keepnull"],
        default=DEFAULT_VARIANT,
        help=f"Variante de consultoría a particionar (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Ruta absoluta al dataset.parquet (default: <CONSULTORIA_OUTPUT_DIR>/<variant>/dataset.parquet)",
    )
    parser.add_argument("--train", type=float, default=0.70, dest="train_ratio")
    parser.add_argument("--val",   type=float, default=0.15, dest="val_ratio")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    parquet_path = (
        Path(args.dataset).expanduser()
        if args.dataset
        else CONSULTORIA_OUTPUT_DIR / args.variant / "dataset.parquet"
    )
    output_path = SPLITS_DIR / f"{args.variant}_split.parquet"

    if not parquet_path.exists():
        print(f"Error: dataset no encontrado en {parquet_path}")
        print("  El dataset lo produce la consultoría (repo consultoria-iner).")
        return 1

    print(f"\nParticionando dataset (variante {args.variant})...")
    print(f"  Entrada: {parquet_path}")
    print(f"  Salida:  {output_path}")

    split_dataset(
        parquet_path=parquet_path,
        output_path=output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"✓ {output_path.name} listo en {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
