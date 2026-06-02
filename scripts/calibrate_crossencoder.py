"""Calibración del Cross-Encoder e incertidumbre por vínculo (Vía A).

Procedimiento POST-HOC sobre el CE ya entrenado (NO re-entrena):
  - Ajusta la temperatura T sobre validación (temperature scaling).
  - Reporta ECE antes/después + reliability curves.
  - Calcula incertidumbre por par sobre test (entropía calibrada + margen).

Uso:
    python scripts/calibrate_crossencoder.py \\
        --checkpoint ~/Data/INER/models/checkpoints/beto_bce_hpc_v2_tok_skipnull/best \\
        --dataset    ~/Data/INER/processed/tesis/output/tok_skipnull/dataset.parquet \\
        --val-pairs  ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_val.parquet \\
        --test-pairs ~/Data/INER/processed/tesis/output/tok_skipnull/pairs_test.parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import EVALUATION_DIR
from record_linkage.evaluation.crossencoder_eval import calibrate_crossencoder


def main():
    parser = argparse.ArgumentParser(description="Calibración del Cross-Encoder (Vía A)")
    parser.add_argument("--checkpoint", required=True,
                        help="Ruta al checkpoint del CE (con config.json + pesos)")
    parser.add_argument("--dataset", required=True,
                        help="Parquet de registros con cols record_id y text")
    parser.add_argument("--val-pairs", required=True,
                        help="Parquet de pares de VALIDACIÓN (ajuste de T)")
    parser.add_argument("--test-pairs", required=True,
                        help="Parquet de pares de TEST (incertidumbre por vínculo)")
    parser.add_argument("--tau-dec", type=float, default=0.12,
                        help="Umbral de decisión para el margen (default: 0.12)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Bins para ECE y reliability curve (default: 10)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    dataset_path    = Path(args.dataset).expanduser()
    val_pairs_path  = Path(args.val_pairs).expanduser()
    test_pairs_path = Path(args.test_pairs).expanduser()

    for p in (checkpoint_path, dataset_path, val_pairs_path, test_pairs_path):
        if not p.exists():
            print(f"ERROR: no encontrado: {p}")
            return 1

    calibrate_crossencoder(
        checkpoint_path=checkpoint_path,
        val_pairs_path=val_pairs_path,
        test_pairs_path=test_pairs_path,
        dataset_path=dataset_path,
        output_dir=EVALUATION_DIR / "calibration",
        tau_dec=args.tau_dec,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        n_bins=args.n_bins,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
