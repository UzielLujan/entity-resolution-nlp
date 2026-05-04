"""
Script de partición train/val/test del dataset a nivel de entidad.

Uso:
    python scripts/run_splitting.py --perfil tesis1
    python scripts/run_splitting.py --perfil tesis1 --train 0.70 --val 0.15 --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import PROCESSED_DIR
from record_linkage.data.splitting import split_dataset


def main():
    parser = argparse.ArgumentParser(description="Partición train/val/test del dataset")
    parser.add_argument(
        "--perfil",
        choices=["tesis0", "tesis0_sin_tokens", "tesis1", "tesis2", "iner"],
        default="tesis1",
    )
    parser.add_argument("--train", type=float, default=0.70, dest="train_ratio")
    parser.add_argument("--val",   type=float, default=0.15, dest="val_ratio")
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    parquet_path = PROCESSED_DIR / args.perfil / "dataset.parquet"
    output_path  = PROCESSED_DIR / args.perfil / "dataset_split.parquet"

    if not parquet_path.exists():
        print(f"Error: dataset no encontrado en {parquet_path}")
        print(f"  Ejecuta primero: python scripts/run_dataset.py --perfil {args.perfil}")
        return 1

    print(f"\nParticionando dataset (Perfil {args.perfil})...")
    print(f"  Entrada: {parquet_path}")
    print(f"  Salida:  {output_path}")

    split_dataset(
        parquet_path=parquet_path,
        output_path=output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"✓ dataset_split.parquet listo en {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
