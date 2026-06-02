"""Mide la distribución de tokens por registro en cada variante del 2×2.

Tokeniza con el modelo indicado y reporta min, max, mean, mediana, p95, p99,
y el % de registros que caben en cortes típicos (128, 256, 384, 512 tokens).

Útil para decidir un `max_seq_length` que no trunque señal significativa, lo
que a su vez determina cuánto batch_size cabe en memoria de la GPU.

Uso:
    python scripts/measure_token_distribution.py --model BETO
    python scripts/measure_token_distribution.py --model BETO --output-csv tokens.csv
    python scripts/measure_token_distribution.py --model RoBERTa-biomedical \\
        --dataset ~/Data/INER/processed/tesis/output/tok_skipnull/dataset.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import MODELS_DIR, PROCESSED_DIR


VARIANTS = ["tok_keepnull", "tok_skipnull", "notok_keepnull", "notok_skipnull"]


def stats_for_variant(tokenizer, parquet_path: Path, variant_label: str) -> dict:
    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()

    # Tokenizar todos los registros sin truncar para medir la longitud real.
    # add_special_tokens=True simula el comportamiento real al alimentar al modelo
    # (incluye [CLS] y [SEP]). truncation=False para no perder el límite real.
    encodings = tokenizer(
        texts, add_special_tokens=True, truncation=False, padding=False,
    )
    lengths = np.array([len(ids) for ids in encodings["input_ids"]])

    stats = {
        "variant":      variant_label,
        "n_records":    len(lengths),
        "min":          int(lengths.min()),
        "mean":         float(lengths.mean()),
        "median":       float(np.median(lengths)),
        "p95":          float(np.percentile(lengths, 95)),
        "p99":          float(np.percentile(lengths, 99)),
        "max":          int(lengths.max()),
        "pct_le_128":   float((lengths <= 128).mean() * 100),
        "pct_le_256":   float((lengths <= 256).mean() * 100),
        "pct_le_384":   float((lengths <= 384).mean() * 100),
        "pct_le_512":   float((lengths <= 512).mean() * 100),
    }
    return stats


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(_fmt(r[h])) for r in rows)) for h in headers}

    line_top = " │ ".join(h.ljust(widths[h]) for h in headers)
    print(line_top)
    print("─" * len(line_top))
    for r in rows:
        print(" │ ".join(_fmt(r[h]).ljust(widths[h]) for h in headers))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Distribución de tokens por variante del 2×2")
    parser.add_argument("--model", default="BETO",
                        help="Modelo a usar como tokenizador (default: BETO)")
    parser.add_argument("--dataset", default=None,
                        help="Si se pasa, mide solo este parquet (col 'text'). "
                             "Si no, mide las 4 variantes en tesis/output/*/dataset.parquet")
    parser.add_argument("--output-csv", default=None,
                        help="Guarda tabla en CSV para reporte de tesis")
    args = parser.parse_args()

    model_path = MODELS_DIR / "pretrained" / args.model
    print(f"Tokenizador: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    rows = []
    if args.dataset:
        rows.append(stats_for_variant(tokenizer, Path(args.dataset), Path(args.dataset).parent.name))
    else:
        tesis_output = PROCESSED_DIR / "tesis" / "output"
        for v in VARIANTS:
            parquet_path = tesis_output / v / "dataset.parquet"
            if not parquet_path.exists():
                print(f"  AVISO: no se encontró {parquet_path}, omitido")
                continue
            print(f"\n  Tokenizando {v}...")
            rows.append(stats_for_variant(tokenizer, parquet_path, v))

    if not rows:
        print("\nERROR: no se midió ninguna variante.")
        return 1

    print(f"\n{'='*60}\nDistribución de tokens (modelo: {args.model})\n{'='*60}")
    print_table(rows)

    print(f"\nLectura:")
    print(f"  pct_le_X = % de registros con ≤X tokens (incluyendo [CLS] y [SEP])")
    print(f"  max_seq_length debe ser ≥ p99 para no truncar el 99% de los registros.")

    if args.output_csv:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(args.output_csv, index=False)
        print(f"\n✓ CSV guardado: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
