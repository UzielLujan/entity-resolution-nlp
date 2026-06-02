"""Orquestación de evaluación del Bi-Encoder — finetuned y zero-shot.

Flujo común:
  1. Carga modelo (checkpoint fine-tuneado o pretrained)
  2. Carga dataset (con o sin split)
  3. Filtra entidades vinculables (presentes en ≥2 bases)
  4. Encoda todos los registros vinculables
  5. Para cada par direccional (A→B y B→A): calcula métricas con compute_metrics_at_k
  6. Devuelve dict con resultados por par + metadata del run

Las funciones públicas:
  - evaluate_finetuned_checkpoint(): para checkpoints fine-tuneados con MNRL
  - evaluate_zeroshot_model():       para modelos pretrained sin fine-tuning
"""

import json
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from record_linkage.config import MODELS_DIR
from record_linkage.evaluation.metrics import candidate_pool_stats, compute_metrics_at_k
from record_linkage.models.biencoder import build_biencoder, encode_texts

K_VALUES_DEFAULT = [1, 5, 10, 20, 50]


# =============================================================================
# Helpers de filesystem / metadata
# =============================================================================

def resolve_checkpoint_path(checkpoint_name: str, epoch: Optional[int] = None) -> Path:
    """Resuelve la ruta al checkpoint: best/ por defecto, epoch_XX si se especifica."""
    run_dir = MODELS_DIR / "checkpoints" / checkpoint_name
    if not run_dir.exists():
        raise FileNotFoundError(f"checkpoint no encontrado: {run_dir}")

    if epoch is not None:
        ckpt_path = run_dir / f"epoch_{epoch:02d}"
        if not ckpt_path.exists():
            available = sorted(p.name for p in run_dir.glob('epoch_*'))
            raise FileNotFoundError(
                f"época {epoch} no encontrada en {run_dir}. Disponibles: {available}"
            )
        return ckpt_path

    best_path = run_dir / "best"
    if best_path.exists():
        return best_path

    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    if epoch_dirs:
        print(f"  Advertencia: no hay best/ — usando {epoch_dirs[-1].name}")
        return epoch_dirs[-1]

    raise FileNotFoundError(f"no hay best/ ni epoch_XX/ en {run_dir}")


def load_run_metadata(checkpoint_name: str) -> dict:
    """Lee training_history.json del run si existe (devuelve {} si no)."""
    hist_path = MODELS_DIR / "checkpoints" / checkpoint_name / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            return json.load(f)
    return {}


def list_available_checkpoints() -> list[str]:
    """Lista de runs en checkpoints/ que tienen best/ disponible."""
    ckpt_root = MODELS_DIR / "checkpoints"
    if not ckpt_root.exists():
        return []
    return sorted(
        d.name for d in ckpt_root.iterdir()
        if d.is_dir() and (d / "best").exists()
    )


# =============================================================================
# Helpers de dataset
# =============================================================================

def load_dataset_split(dataset_path: Path, split: Optional[str] = None) -> pd.DataFrame:
    """Lee parquet y opcionalmente filtra por split.

    Si split=None, devuelve todo el dataframe (modo zero-shot que no usa splits).
    """
    df = pd.read_parquet(dataset_path)
    if split is not None:
        if "split" not in df.columns:
            raise ValueError(
                f"el parquet no tiene columna 'split': {dataset_path}\n"
                "Usa dataset_split.parquet, no dataset.parquet"
            )
        df = df[df["split"] == split].reset_index(drop=True)
        entity_sources = df.groupby("entity_id")["source_db"].nunique()
        linkable = (entity_sources > 1).sum()
        print(f"  Split '{split}': {len(df):,} registros | "
              f"{df['entity_id'].nunique():,} entidades | {linkable:,} vinculables")
    else:
        print(f"  Dataset: {len(df):,} registros | {df['entity_id'].nunique():,} entidades")
    return df


def find_linkable_records(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra registros cuya entity_id aparece en 2+ source_db (vinculables cross-database)."""
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    linkable_ids = entity_sources[entity_sources > 1].index
    df_linkable = df[df["entity_id"].isin(linkable_ids)].copy()
    print(f"  Registros vinculables: {len(df_linkable):,} "
          f"({len(linkable_ids):,} entidades en 2+ bases)")
    return df_linkable


# =============================================================================
# Núcleo: evaluación bidireccional por par de bases
# =============================================================================

def _evaluate_bidirectional_pairs(
    df_linkable: pd.DataFrame,
    embeddings: np.ndarray,
    k_values: list,
) -> dict:
    """Para cada par (A, B) y (B, A) calcula métricas con compute_metrics_at_k.

    Args:
        df_linkable: dataframe con columnas record_id, source_db, entity_id (filtrado a vinculables).
        embeddings:  (n_records, dim) en el mismo orden que df_linkable.
        k_values:    lista de K para Hit/Recall/Precision.

    Returns:
        dict {"src_a → src_b": metric_dict, ...} con candidate_pool_stats inline.
    """
    entity_ids_array = df_linkable["entity_id"].values
    source_db_array  = df_linkable["source_db"].values
    source_list      = sorted(df_linkable["source_db"].unique())

    all_results = {}
    for src_a, src_b in permutations(source_list, 2):
        pair_key = f"{src_a} → {src_b}"
        print(f"\n  Evaluando: {pair_key}")

        mask_a = source_db_array == src_a
        mask_b = source_db_array == src_b
        emb_a = embeddings[mask_a]
        emb_b = embeddings[mask_b]
        ids_a = entity_ids_array[mask_a]
        ids_b = entity_ids_array[mask_b]

        linkable_ids = set(ids_a) & set(ids_b)
        query_mask = np.isin(ids_a, list(linkable_ids))
        if query_mask.sum() == 0:
            print(f"    Sin pares positivos cross-database — omitiendo")
            continue

        pool_stats = candidate_pool_stats(ids_b)
        print(f"    Pool candidatos: {pool_stats['n_entities']} entidades | "
              f"max_pos={pool_stats['max_positives_per_entity']} | "
              f"mean_pos={pool_stats['mean_positives_per_entity']}")

        metrics = compute_metrics_at_k(
            query_embeddings=emb_a[query_mask],
            candidate_embeddings=emb_b,
            query_entity_ids=ids_a[query_mask],
            candidate_entity_ids=ids_b,
            k_values=k_values,
        )
        metrics["candidate_pool_stats"] = pool_stats
        all_results[pair_key] = metrics

        print(f"    n_queries={metrics['n_queries']}  MRR={metrics['MRR']:.4f}")
        for k in k_values:
            print(f"    K={k:2d}  Hit={metrics[f'Hit@{k}']:.4f}  "
                  f"Rec={metrics[f'Recall@{k}']:.4f}  "
                  f"RecN={metrics[f'RecallNorm@{k}']:.4f}  "
                  f"Prec={metrics[f'Precision@{k}']:.4f}")
        if "space_metrics" in metrics:
            sm = metrics["space_metrics"]
            print(f"    μ_pos={sm['mu_pos']:.3f}  μ_neg={sm['mu_neg']:.3f}  Δ={sm['delta_separability']:.3f}")

    return all_results


# =============================================================================
# API pública: evaluación finetuned y zero-shot
# =============================================================================

def evaluate_finetuned_checkpoint(
    checkpoint_name: str,
    dataset_path: Path,
    split: str = "test",
    epoch: Optional[int] = None,
    k_values: list = K_VALUES_DEFAULT,
) -> dict:
    """Evalúa un checkpoint fine-tuneado con MNRL sobre el split indicado."""
    print(f"\n{'='*60}\nEvaluando: {checkpoint_name}\n{'='*60}")

    ckpt_path = resolve_checkpoint_path(checkpoint_name, epoch)
    print(f"  Checkpoint: {ckpt_path}")

    metadata = load_run_metadata(checkpoint_name)
    args_meta = metadata.get("args", {})
    best_epoch = metadata.get("best_epoch")
    epoch_label = f"epoch_{epoch:02d}" if epoch is not None else f"best (ep{best_epoch})"
    print(f"  Modelo base: {args_meta.get('model', '?')} | "
          f"lr={args_meta.get('base_lr', '?')} | "
          f"temp={args_meta.get('temperature', '?')} | "
          f"evaluando: {epoch_label}")

    print(f"\n  Cargando modelo...")
    t0 = time.time()
    model = build_biencoder(ckpt_path)
    max_seq = args_meta.get("max_seq_length", 512)
    model.max_seq_length = max_seq
    print(f"  Modelo cargado en {time.time()-t0:.1f}s — max_seq_length={max_seq}")
    print(f"  Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"\n  Cargando split '{split}'...")
    df_split = load_dataset_split(dataset_path, split=split)
    df_linkable = find_linkable_records(df_split)

    print(f"\n  Codificando registros vinculables...")
    t0 = time.time()
    embeddings = encode_texts(model, df_linkable["text"].tolist(), batch_size=64)
    print(f"  Embeddings: {time.time()-t0:.1f}s — shape={embeddings.shape}")

    results = _evaluate_bidirectional_pairs(df_linkable, embeddings, k_values)

    return {
        "checkpoint":      checkpoint_name,
        "checkpoint_path": str(ckpt_path),
        "epoch_evaluated": epoch_label,
        "split":           split,
        "model":           args_meta.get("model", "?"),
        "base_lr":         args_meta.get("base_lr"),
        "temperature":     args_meta.get("temperature"),
        "best_val_loss":   min(
            (r["val_loss"] for r in metadata.get("history", [])), default=None
        ),
        "results":         results,
    }


def evaluate_zeroshot_model(
    model_name: str,
    dataset_path: Path,
    k_values: list = K_VALUES_DEFAULT,
    max_seq_length: int = 512,
    batch_size: int = 64,
) -> dict:
    """Evalúa un modelo pretrained (sin fine-tuning) sobre todo el dataset.

    A diferencia de evaluate_finetuned_checkpoint, no usa splits — todo el dataset
    sirve como pool. Devuelve {} si el modelo no se encuentra localmente.
    """
    print(f"\n{'='*60}\nEvaluando zero-shot: {model_name}\n{'='*60}")

    model_path = MODELS_DIR / "pretrained" / model_name
    if not model_path.exists():
        print(f"  ERROR: modelo no encontrado en {model_path}")
        print(f"  Ejecuta primero: python scripts/download_model.py --name {model_name} --model <hub-id>")
        return {}

    print(f"  Cargando modelo desde {model_path}...")
    t0 = time.time()
    model = build_biencoder(model_path)
    model.max_seq_length = max_seq_length
    print(f"  Modelo cargado en {time.time()-t0:.1f}s — max_seq_length={max_seq_length}")
    print(f"  Dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print(f"\n  Cargando dataset...")
    df = load_dataset_split(dataset_path, split=None)
    df_linkable = find_linkable_records(df)

    print(f"\n  Codificando registros vinculables...")
    t0 = time.time()
    embeddings = encode_texts(model, df_linkable["text"].tolist(), batch_size=batch_size)
    print(f"  Embeddings: {time.time()-t0:.1f}s — shape={embeddings.shape}")

    return _evaluate_bidirectional_pairs(df_linkable, embeddings, k_values)
