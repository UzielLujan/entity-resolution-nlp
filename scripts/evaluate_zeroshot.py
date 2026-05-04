"""Evaluación zero-shot del Bi-Encoder sobre pares cross-database del INER.

Calcula Hit@K (K∈{1,5,10,20,50}) y MRR sobre los 9,855 pares positivos confirmados
(mismo entity_id en distintas bases). Sirve como diagnóstico pre-MNRL para medir
la capacidad de recuperación de los modelos base sin fine-tuning.

Nota: quedan 1,569 pares residuales (mismo expediente, nombre distinto tras
normalización) mezclados como falsos negativos en el dataset — pendiente de revisión
y corrección del etiquetado.

Uso:
    python scripts/evaluate_zeroshot.py --model BETO
    python scripts/evaluate_zeroshot.py --model RoBERTa-bne
    python scripts/evaluate_zeroshot.py --model BETO --model RoBERTa-bne
    python scripts/evaluate_zeroshot.py --all

Los resultados se guardan en ~/Data/INER/outputs/evaluation/zeroshot_<model>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from record_linkage.config import EVALUATION_DIR, MODELS_DIR, PROCESSED_DIR
from record_linkage.models.biencoder import build_biencoder, encode_texts

KNOWN_MODELS = {
    "BETO": "BETO",
    "RoBERTa-biomedical": "RoBERTa-biomedical",
    "paraphrase-multilingual": "paraphrase-multilingual",
    # Pendiente de revisión: RoBERTa-bne-sts pre-entrenado para clasificación, no similitud.
    # Los pesos existen localmente (descargados manualmente, no vía download_model.py).
    # Discutir con asesor si tiene sentido evaluar o descartar este modelo.
    "RoBERTa-bne-sts": "RoBERTa-bne-sts",
}

K_VALUES = [1, 5, 10, 20, 50]


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    print(f"  Dataset cargado: {len(df):,} registros, {df['entity_id'].nunique():,} entidades")
    return df


def find_hard_pairs(df: pd.DataFrame):
    """Retorna registros con pares positivos cross-database confirmados.

    Filtra entidades presentes en más de una source_db (entity_id compartido).
    Estos son los 9,855 pares confirmados por la llave (expediente, nombre_v2_normalizado).
    Nota: el nombre de la función es un remanente — pendiente de renombrar junto
    con el rediseño del pipeline de evaluación.
    """
    # Entidades presentes en más de una base
    entity_sources = df.groupby("entity_id")["source_db"].nunique()
    linkable_entities = entity_sources[entity_sources > 1].index

    df_linkable = df[df["entity_id"].isin(linkable_entities)].copy()
    print(f"  Entidades vinculables: {len(linkable_entities):,}")
    print(f"  Registros vinculables: {len(df_linkable):,}")
    return df_linkable


def compute_recall_at_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_entity_ids: np.ndarray,
    candidate_entity_ids: np.ndarray,
    k_values: list,
) -> dict:
    """Calcula Recall@K y MRR para un conjunto de consultas vs candidatos.

    Para cada registro de consulta, recupera los K candidatos más similares
    y verifica si alguno comparte entity_id (match positivo).

    Args:
        query_embeddings: (n_queries, dim) — ya normalizados
        candidate_embeddings: (n_candidates, dim) — ya normalizados
        query_entity_ids: (n_queries,)
        candidate_entity_ids: (n_candidates,)
        k_values: lista de K a evaluar

    Returns:
        dict con Recall@K para cada K, MRR, y estadísticas de similitud
    """
    # Similitud coseno por multiplicación matricial (embeddings normalizados)
    sim_matrix = query_embeddings @ candidate_embeddings.T  # (n_queries, n_candidates)

    max_k = max(k_values)
    recalls = {k: 0 for k in k_values}
    reciprocal_ranks = []
    positive_sims = []
    negative_sims_sample = []

    n_queries = len(query_entity_ids)

    for i in range(n_queries):
        sims = sim_matrix[i]  # (n_candidates,)
        true_entity = query_entity_ids[i]

        # Índices de candidatos positivos (misma entidad, excluyendo el propio registro si está)
        positive_mask = candidate_entity_ids == true_entity
        negative_mask = ~positive_mask

        if not positive_mask.any():
            continue

        # Similitudes de positivos y negativos para análisis del espacio métrico
        positive_sims.extend(sims[positive_mask].tolist())
        neg_sims = sims[negative_mask]
        if len(neg_sims) > 0:
            # Muestra aleatoria para no saturar memoria
            sample_idx = np.random.choice(len(neg_sims), min(10, len(neg_sims)), replace=False)
            negative_sims_sample.extend(neg_sims[sample_idx].tolist())

        # Top-K candidatos por similitud
        top_k_indices = np.argpartition(sims, -max_k)[-max_k:]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]

        top_k_entity_ids = candidate_entity_ids[top_k_indices]

        # Recall@K
        for k in k_values:
            if true_entity in top_k_entity_ids[:k]:
                recalls[k] += 1

        # MRR: posición del primer positivo en el ranking completo
        sorted_indices = np.argsort(sims)[::-1]
        sorted_entity_ids = candidate_entity_ids[sorted_indices]
        rank = np.where(sorted_entity_ids == true_entity)[0]
        if len(rank) > 0:
            reciprocal_ranks.append(1.0 / (rank[0] + 1))

    n_valid = n_queries
    results = {
        f"Recall@{k}": round(recalls[k] / n_valid, 4) for k in k_values
    }
    results["MRR"] = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0
    results["n_queries"] = n_valid

    # Análisis del espacio métrico (§4 de metricas_evaluacion.md)
    if positive_sims and negative_sims_sample:
        mu_pos = float(np.mean(positive_sims))
        mu_neg = float(np.mean(negative_sims_sample))
        sigma_pos = float(np.std(positive_sims))
        sigma_neg = float(np.std(negative_sims_sample))
        pooled_std = float(np.sqrt((sigma_pos**2 + sigma_neg**2) / 2))
        delta = (mu_pos - mu_neg) / pooled_std if pooled_std > 0 else 0.0

        results["space_metrics"] = {
            "mu_pos": round(mu_pos, 4),
            "mu_neg": round(mu_neg, 4),
            "sigma_pos": round(sigma_pos, 4),
            "sigma_neg": round(sigma_neg, 4),
            "delta_separability": round(delta, 4),
        }

    return results


def evaluate_model(model_name: str, dataset_path: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluando: {model_name}")
    print(f"{'='*60}")

    model_path = MODELS_DIR / "pretrained" / model_name
    if not model_path.exists():
        print(f"  ERROR: Modelo no encontrado en {model_path}")
        print(f"  Ejecuta primero: python scripts/download_model.py --all")
        return {}

    print(f"  Cargando modelo desde {model_path}...")
    t0 = time.time()
    model = build_biencoder(model_path)
    model.max_seq_length = 512
    print(f"  Modelo cargado en {time.time()-t0:.1f}s — max_seq_length={model.max_seq_length}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Dispositivo: {device}")

    print("\n  Cargando dataset zero-shot...")
    df = load_dataset(dataset_path)
    df_linkable = find_hard_pairs(df)

    # Separar por base de datos para evaluación cross-database
    sources = df_linkable["source_db"].unique()
    print(f"  Bases de datos: {list(sources)}")

    # Evaluar todas las combinaciones de bases (A→B, A→C, B→C)
    source_list = sorted(sources)
    all_results = {}

    print("\n  Codificando todos los registros vinculables...")
    t0 = time.time()
    all_texts = df_linkable["text"].tolist()
    all_embeddings = encode_texts(model, all_texts, batch_size=64)
    print(f"  Embeddings generados en {time.time()-t0:.1f}s — shape: {all_embeddings.shape}")

    entity_ids_array = df_linkable["entity_id"].values
    source_db_array = df_linkable["source_db"].values

    # Evaluación por par de bases
    for i, src_a in enumerate(source_list):
        for src_b in source_list[i+1:]:
            pair_key = f"{src_a} → {src_b}"
            print(f"\n  Evaluando: {pair_key}")

            mask_a = source_db_array == src_a
            mask_b = source_db_array == src_b

            emb_a = all_embeddings[mask_a]
            emb_b = all_embeddings[mask_b]
            ids_a = entity_ids_array[mask_a]
            ids_b = entity_ids_array[mask_b]

            # Solo consultas con match real en la otra base
            linkable_ids = set(ids_a) & set(ids_b)
            query_mask = np.isin(ids_a, list(linkable_ids))

            if query_mask.sum() == 0:
                print(f"    Sin pares positivos cross-database — omitiendo")
                continue

            results = compute_recall_at_k(
                query_embeddings=emb_a[query_mask],
                candidate_embeddings=emb_b,
                query_entity_ids=ids_a[query_mask],
                candidate_entity_ids=ids_b,
                k_values=K_VALUES,
            )
            all_results[pair_key] = results

            print(f"    n_queries: {results['n_queries']}")
            for k in K_VALUES:
                print(f"    Recall@{k:2d}: {results[f'Recall@{k}']:.4f}")
            print(f"    MRR:       {results['MRR']:.4f}")
            if "space_metrics" in results:
                sm = results["space_metrics"]
                print(f"    μ_pos={sm['mu_pos']:.3f}  μ_neg={sm['mu_neg']:.3f}  Δ={sm['delta_separability']:.3f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluación zero-shot del Bi-Encoder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", action="append", dest="models",
                       help="Nombre del modelo (BETO, RoBERTa-bne). Repetir para varios.")
    group.add_argument("--all", action="store_true", help="Evalúa todos los modelos conocidos")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Ruta alternativa al dataset.parquet (por defecto: tesis0_sin_tokens/dataset.parquet)")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path) if args.dataset_path else PROCESSED_DIR / "tesis0_sin_tokens" / "dataset.parquet"
    if not dataset_path.exists():
        print(f"ERROR: Dataset no encontrado en {dataset_path}")
        print("Genera primero: python scripts/run_dataset.py --perfil tesis0 --no-special-tokens")
        return 1

    models_to_eval = list(KNOWN_MODELS.keys()) if args.all else args.models

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    all_model_results = {}
    for model_name in models_to_eval:
        results = evaluate_model(model_name, dataset_path)
        if results:
            all_model_results[model_name] = results

    # Guardar resultados
    if all_model_results:
        suffix = Path(dataset_path).parent.name
        output_path = EVALUATION_DIR / f"zeroshot_results_{suffix}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_model_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Resultados guardados en {output_path}")

        # Tabla resumen
        print("\n" + "="*60)
        print("RESUMEN — Recall@K y MRR (zero-shot)")
        print("="*60)
        for model_name, model_results in all_model_results.items():
            print(f"\n{model_name}")
            for pair_key, pair_results in model_results.items():
                print(f"  {pair_key}")
                for k in K_VALUES:
                    print(f"    Recall@{k:2d}: {pair_results[f'Recall@{k}']:.4f}", end="  ")
                print(f"\n    MRR: {pair_results['MRR']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
