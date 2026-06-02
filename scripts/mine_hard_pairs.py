"""Hard Negative Mining para el Cross-Encoder.

Toma un Bi-Encoder fine-tuneado y un dataset_split.parquet, encoda todos los registros
linkable, y para cada registro identifica:

  • POSITIVE (label=1):      par cross-DB con mismo entity_id, BE lo puso en top-K. "Match obvio".
  • HARD_POSITIVE (label=1): par cross-DB con mismo entity_id, BE NO lo puso en top-K. BE falló.
  • HARD_NEGATIVE (label=0): par cross-DB con entity_id distinto, BE lo puso en top-K. BE confundió.

Los easy negatives (mismo entity_id distinto, BE los filtra correctamente) NO se incluyen —
nunca llegan al CE en producción.

Salida: 3 parquets (train, val, test) en el directorio del dataset, conteniendo SOLO los
índices del grafo (no el texto). El texto vive en `dataset.parquet` y se hace lookup vía
`record_id` en `train_crossencoder.py` y `evaluate_crossencoder.py`.

    pairs_{train,val,test}.parquet
    ├── record_id_a, record_id_b
    ├── label (0/1)
    ├── similarity (cos del BE — para auditoría)
    └── type ("positive" | "hard_positive" | "hard_negative")

Uso:
    python scripts/mine_hard_pairs.py \\
        --checkpoint beto_mnrl_hpc_v2_<variante> \\
        --dataset    ~/Data/INER/processed/tesis/output/<variante>/dataset_split.parquet \\
        --top-k      20

    # Default: --top-k 20, escribe pairs_{train,val,test}.parquet en el dir del dataset.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.evaluation.biencoder_eval import (
    find_linkable_records,
    load_dataset_split,
    resolve_checkpoint_path,
)
from record_linkage.models.biencoder import build_biencoder, encode_texts


def mine_pairs_for_split(
    df_split: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int,
    max_easy_positives: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Construye el parquet de pares (positivos + hard negatives) para un split.

    Args:
        df_split:           dataframe del split (cols: record_id, source_db, text, entity_id).
                            Solo los registros vinculables (entity_id en ≥2 source_db).
        embeddings:         (n, dim) embeddings normalizados, en el mismo orden que df_split.
        top_k:              cuántos vecinos por query considerar como "hard" negativos.
        max_easy_positives: si > 0, submuestrea los "positive" (no los hard_positive) a este
                            tope. Útil cuando dominan el dataset y queremos balancearlos contra
                            los hard cases. 0 = mantener todos.

    Returns:
        DataFrame con columnas:
            record_id_a, record_id_b, label, similarity, type
        Solo índices del grafo — el texto se hace lookup vía dataset.parquet al cargar.
    """
    record_ids = df_split["record_id"].values
    entity_ids = df_split["entity_id"].values
    source_dbs = df_split["source_db"].values

    n = len(df_split)
    print(f"  Split con {n:,} registros vinculables")

    # Matriz de similitud completa (n × n). Para 23k registros cabe en RAM
    # (23k² × 4 bytes ≈ 2 GB) — usar GPU si está disponible.
    print(f"  Calculando matriz de similitud {n}×{n}...")
    t0 = time.time()
    emb_t = torch.from_numpy(embeddings).to("cuda" if torch.cuda.is_available() else "cpu")
    sim_matrix = (emb_t @ emb_t.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, -np.inf)  # excluir auto-similitud
    print(f"  Matriz calculada en {time.time()-t0:.1f}s")

    # Precalcular top-K por cada query para reusar en hard_neg y hard_pos detection
    top_k_per_query = np.argpartition(sim_matrix, -top_k, axis=1)[:, -top_k:]
    top_k_set_per_query = [set(row.tolist()) for row in top_k_per_query]

    rows = []

    # === Positivos cross-DB: separar en "positive" (BE acertó) vs "hard_positive" (BE falló) ===
    print(f"  Clasificando positivos cross-DB...")
    entity_to_indices: dict = {}
    for i, eid in enumerate(entity_ids):
        entity_to_indices.setdefault(eid, []).append(i)

    n_easy_pos = 0
    n_hard_pos = 0
    for idxs in entity_to_indices.values():
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = idxs[i], idxs[j]
                if source_dbs[a] == source_dbs[b]:
                    continue
                # Hard positive si b NO está en top-K de a Y a NO está en top-K de b
                # (criterio simétrico: el BE falló en ambas direcciones)
                in_top_k = (b in top_k_set_per_query[a]) or (a in top_k_set_per_query[b])
                pair_type = "positive" if in_top_k else "hard_positive"
                if pair_type == "positive":
                    n_easy_pos += 1
                else:
                    n_hard_pos += 1
                rows.append({
                    "record_id_a": int(record_ids[a]),
                    "record_id_b": int(record_ids[b]),
                    "label":       1,
                    "similarity":  float(sim_matrix[a, b]),
                    "type":        pair_type,
                })
    print(f"    {n_easy_pos:,} positive (BE acertó) + {n_hard_pos:,} hard_positive (BE falló)")

    # Submuestrear easy positives si se pidió
    if max_easy_positives > 0 and n_easy_pos > max_easy_positives:
        rng = np.random.default_rng(seed)
        easy_idxs = [i for i, r in enumerate(rows) if r["type"] == "positive"]
        keep = set(rng.choice(easy_idxs, size=max_easy_positives, replace=False).tolist())
        rows = [r for i, r in enumerate(rows) if r["type"] != "positive" or i in keep]
        print(f"    Submuestreados easy positives: {n_easy_pos:,} → {max_easy_positives:,}")

    # === Hard negatives: top-K por query, filtrar mismo entity_id ===
    print(f"  Buscando hard negatives (top-{top_k} por query)...")
    n_hard_neg = 0
    seen_neg_pairs = set()  # evita duplicar (a, b) y (b, a)
    for i in range(n):
        for j in top_k_per_query[i]:
            if entity_ids[i] == entity_ids[j]:
                continue  # es positivo
            if source_dbs[i] == source_dbs[j]:
                continue  # mismo source_db, no es par cross-DB
            a, b = (i, j) if record_ids[i] < record_ids[j] else (j, i)
            key = (a, b)
            if key in seen_neg_pairs:
                continue
            seen_neg_pairs.add(key)
            rows.append({
                "record_id_a": int(record_ids[a]),
                "record_id_b": int(record_ids[b]),
                "label":       0,
                "similarity":  float(sim_matrix[a, b]),
                "type":        "hard_negative",
            })
            n_hard_neg += 1
    print(f"    {n_hard_neg:,} hard_negative únicos cross-DB")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Hard Negative Mining sobre el Bi-Encoder")
    parser.add_argument("--checkpoint", required=True,
                        help="Nombre del run en checkpoints/ (usa best/ por defecto)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Evaluar epoch_XX específica en lugar de best/")
    parser.add_argument("--dataset", required=True,
                        help="Ruta al dataset_split.parquet")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K vecinos por query para definir hard negatives (default: 20)")
    parser.add_argument("--max-easy-positives", type=int, default=0,
                        help="Si >0, submuestrea los easy positives a este tope por split. "
                             "0 = mantener todos. Útil cuando dominan vs hard cases.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directorio de salida (default: mismo dir del dataset)")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset no encontrado: {dataset_path}")
        return 1
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar BE
    ckpt_path = resolve_checkpoint_path(args.checkpoint, args.epoch)
    print(f"\nCargando BE: {ckpt_path}")
    t0 = time.time()
    model = build_biencoder(ckpt_path)
    model.max_seq_length = args.max_seq_length
    print(f"BE cargado en {time.time()-t0:.1f}s")

    # Cargar dataset completo
    df = pd.read_parquet(dataset_path)
    if "split" not in df.columns:
        print(f"ERROR: dataset sin columna 'split': {dataset_path}")
        return 1
    print(f"\nDataset: {len(df):,} registros | splits: {sorted(df['split'].unique())}")

    # Minar pares por split
    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}\nMinando split = {split}\n{'='*60}")
        df_split = df[df["split"] == split].reset_index(drop=True)
        df_linkable = find_linkable_records(df_split)

        if len(df_linkable) == 0:
            print(f"  Sin registros vinculables — omitiendo.")
            continue

        print(f"  Codificando {len(df_linkable):,} registros...")
        t0 = time.time()
        embeddings = encode_texts(
            model, df_linkable["text"].tolist(), batch_size=args.batch_size
        )
        print(f"  Embeddings: {time.time()-t0:.1f}s — shape={embeddings.shape}")

        df_pairs = mine_pairs_for_split(
            df_linkable, embeddings,
            top_k=args.top_k,
            max_easy_positives=args.max_easy_positives,
            seed=args.seed,
        )

        out_path = output_dir / f"pairs_{split}.parquet"
        df_pairs.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")

        type_counts = df_pairs["type"].value_counts().to_dict()
        print(f"\n  ✓ {out_path}")
        print(f"    {len(df_pairs):,} pares totales | "
              f"positive={type_counts.get('positive', 0):,} | "
              f"hard_positive={type_counts.get('hard_positive', 0):,} | "
              f"hard_negative={type_counts.get('hard_negative', 0):,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
