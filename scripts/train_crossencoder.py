"""CLI delgado para entrenar el Cross-Encoder.

La lógica vive en src/record_linkage/training/train_crossencoder.py.
Este script solo hace import y delega a main(), siguiendo el patrón de
scripts/run_train_biencoder.py.

Uso:
    python scripts/train_crossencoder.py --model BETO \\
        --pairs-train ~/Data/INER/processed/tesis/output/<variante>/pairs_train.parquet \\
        --pairs-val   ~/Data/INER/processed/tesis/output/<variante>/pairs_val.parquet \\
        --output      ~/Data/INER/models/checkpoints/<variante>_ce \\
        --epochs 3 --batch-size 16 --lr 2e-5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.training.train_crossencoder import main

if __name__ == "__main__":
    sys.exit(main())
