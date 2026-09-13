"""Exporta los embeddings del Bi-Encoder fine-tuneado a un parquet keyed por record_id.

Codifica la columna `text` de un dataset.parquet completo (los 23,706 registros, sin
splits) con un checkpoint del BE y escribe `embeddings.parquet [record_id, embedding]`.
Los embeddings salen L2-normalizados (producto punto == similitud coseno).

Es el artefacto de la dirección inversa tesis → consultoría: `consultoria-iner` lo
consume vía el método `cos_biencoder` del REGISTRY (`build_consolidated_json.py --cosine`).
Contrato: el `record_id` es la posición global econo → comorbilidad → trabajo_social,
idéntico en ambos repos (datasets validados bit-a-bit).

La procedencia (checkpoint, dataset, dimensión) queda registrada en la metadata del
parquet para auditoría.

Uso:
    python scripts/export_embeddings.py \\
        --checkpoint beto_mnrl --variant tok_skipnull \\
        --dataset ~/Data/INER/processed/default/output/tok_skipnull/dataset.parquet

    # El output default es $INER_DATA_ROOT/modeling/embeddings/<variant>/embeddings.parquet
    python scripts/export_embeddings.py --checkpoint <run> --dataset <parquet> \\
        --output /ruta/explicita/embeddings.parquet
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import embeddings_path
from record_linkage.evaluation.biencoder_eval import resolve_checkpoint_path
from record_linkage.models.biencoder import build_biencoder, encode_texts


def parse_args():
    p = argparse.ArgumentParser(
        description="Exporta embeddings del Bi-Encoder a embeddings.parquet [record_id, embedding]."
    )
    p.add_argument("--checkpoint", required=True,
                   help="Nombre del run en checkpoints/ (usa best/ por defecto).")
    p.add_argument("--dataset", required=True, type=Path,
                   help="Ruta al dataset.parquet completo (record_id, source_db, text, entity_id).")
    p.add_argument("--output", type=Path, default=None,
                   help="Ruta de salida (default: modeling/embeddings/<variant>/embeddings.parquet).")
    p.add_argument("--variant", default=None,
                   help="Variante del dataset (default: directorio que contiene el parquet).")
    p.add_argument("--epoch", type=int, default=None,
                   help="Época específica del checkpoint (default: best/).")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    dataset_path = args.dataset.expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
    variant = args.variant or dataset_path.parent.name
    output_path = (
        args.output.expanduser()
        if args.output
        else embeddings_path(variant)
    )

    ckpt_path = resolve_checkpoint_path(
        args.checkpoint, epoch=args.epoch, variant=variant
    )
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset:    {dataset_path}")
    print(f"Output:     {output_path}")

    df = pd.read_parquet(dataset_path)
    for col in ("record_id", "text"):
        if col not in df.columns:
            raise ValueError(f"El dataset no tiene columna '{col}' — ¿es un dataset.parquet de finalize?")
    print(f"Registros:  {len(df):,}")

    model = build_biencoder(ckpt_path)

    t0 = time.time()
    embeddings = encode_texts(model, df["text"].tolist(), batch_size=args.batch_size)
    print(f"Encoding completado en {time.time() - t0:.1f}s — shape {embeddings.shape}")

    out_df = pd.DataFrame({
        "record_id": df["record_id"].astype("int64").values,
        "embedding": list(embeddings.astype("float32")),
    })

    provenance = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "dataset": str(dataset_path),
        "variant": variant,
        "n_records": len(out_df),
        "dim": int(embeddings.shape[1]),
        "normalized": True,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    table = pa.Table.from_pandas(out_df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"biencoder_provenance"] = json.dumps(provenance).encode("utf-8")
    table = table.replace_schema_metadata(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")

    size_mb = output_path.stat().st_size / 1e6
    print(f"✓ {output_path}  ({len(out_df):,} registros, {embeddings.shape[1]}D, {size_mb:.1f} MB)")
    print(f"  Procedencia en metadata del parquet: {json.dumps(provenance, indent=2)}")


if __name__ == "__main__":
    main()
