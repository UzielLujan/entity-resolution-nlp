"""
CLI para entrenar el Bi-Encoder con MNRL.

Uso:
    python scripts/run_train_biencoder.py --model BETO --epochs 1 --batch-size 8 --n-aug 0 --viz
    python scripts/run_train_biencoder.py --model BETO --epochs 10 --batch-size 64 --n-aug 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.training.train_biencoder import main

if __name__ == "__main__":
    sys.exit(main())
