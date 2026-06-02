"""Orquestación de evaluación del Cross-Encoder.

Recibe un checkpoint del CE y un parquet de pares (`text_a`, `text_b`, `label`),
puntúa los pares y reporta las métricas de clasificación binaria.
"""

import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from record_linkage.evaluation.metrics import compute_binary_classification_metrics
from record_linkage.models.crossencoder import build_crossencoder, score_pairs


def _texts_from_pairs(pairs_df: pd.DataFrame, records_df: pd.DataFrame) -> tuple[list, list]:
    """Lookup de text_a y text_b desde records_df via record_id."""
    id_to_text = dict(zip(records_df["record_id"].astype(int), records_df["text"].astype(str)))
    texts_a = [id_to_text[int(r)] for r in pairs_df["record_id_a"]]
    texts_b = [id_to_text[int(r)] for r in pairs_df["record_id_b"]]
    return texts_a, texts_b


def evaluate_crossencoder_checkpoint(
    checkpoint_path: Union[str, Path],
    pairs_path: Path,
    dataset_path: Path,
    threshold: float = 0.5,
    batch_size: int = 32,
    max_length: int = 512,
) -> dict:
    """Evalúa un CE entrenado sobre un parquet de pares etiquetados.

    Args:
        checkpoint_path: ruta al directorio del checkpoint (con config.json + pytorch_model.bin).
        pairs_path:      parquet de pares (cols: record_id_a, record_id_b, label).
        dataset_path:    parquet de registros (cols: record_id, text) para lookup de texto.
        threshold:       umbral para binarizar scores. Default 0.5; en producción
                         se calibra sobre validación.

    Returns:
        dict con métricas + threshold + path del checkpoint + score stats.
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"\n{'='*60}\nEvaluando CE: {checkpoint_path}\n{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Dispositivo: {device}")

    print(f"  Cargando modelo...")
    t0 = time.time()
    model, tokenizer = build_crossencoder(checkpoint_path)
    model = model.to(device)
    print(f"  Modelo cargado en {time.time()-t0:.1f}s")

    print(f"\n  Cargando registros: {dataset_path}")
    records_df = pd.read_parquet(dataset_path)

    print(f"  Cargando pares: {pairs_path}")
    df = pd.read_parquet(pairs_path)
    required = {"record_id_a", "record_id_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet sin columnas requeridas: {missing}")
    print(f"  {len(df):,} pares | "
          f"{(df['label']==1).sum():,} positivos | "
          f"{(df['label']==0).sum():,} negativos")

    texts_a, texts_b = _texts_from_pairs(df, records_df)

    print(f"\n  Puntuando pares...")
    t0 = time.time()
    scores = score_pairs(
        model, tokenizer, texts_a, texts_b,
        batch_size=batch_size, max_length=max_length, device=device,
    ).numpy()
    print(f"  Scores generados en {time.time()-t0:.1f}s")

    labels = df["label"].astype(int).values
    metrics = compute_binary_classification_metrics(scores, labels, threshold=threshold)

    print(f"\n  Threshold={threshold}")
    print(f"  F1={metrics['f1']:.4f}  Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}")
    print(f"  Accuracy={metrics['accuracy']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']}")
    print(f"  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")

    # Distribución de scores para histogramas
    score_stats = {
        "score_mean_positives": float(scores[labels == 1].mean()) if (labels == 1).any() else None,
        "score_mean_negatives": float(scores[labels == 0].mean()) if (labels == 0).any() else None,
        "score_std_positives":  float(scores[labels == 1].std())  if (labels == 1).any() else None,
        "score_std_negatives":  float(scores[labels == 0].std())  if (labels == 0).any() else None,
    }

    return {
        "checkpoint":  str(checkpoint_path),
        "pairs_path":  str(pairs_path),
        "metrics":     metrics,
        "score_stats": score_stats,
    }


def find_optimal_threshold(
    checkpoint_path: Union[str, Path],
    pairs_path: Path,
    dataset_path: Path,
    batch_size: int = 32,
    max_length: int = 512,
    n_thresholds: int = 51,
) -> dict:
    """Barre umbrales en [0, 1] sobre el parquet de validación y devuelve el F1-óptimo.

    Útil para calibrar el threshold antes de evaluar sobre test.
    """
    from sklearn.metrics import f1_score

    checkpoint_path = Path(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_crossencoder(checkpoint_path)
    model = model.to(device)

    records_df = pd.read_parquet(dataset_path)
    df = pd.read_parquet(pairs_path)
    texts_a, texts_b = _texts_from_pairs(df, records_df)
    scores = score_pairs(
        model, tokenizer, texts_a, texts_b,
        batch_size=batch_size, max_length=max_length, device=device,
    ).numpy()
    labels = df["label"].astype(int).values

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)

    return {
        "best_threshold": round(best_t, 4),
        "best_f1":        round(best_f1, 4),
        "n_thresholds":   n_thresholds,
    }


def calibrate_crossencoder(
    checkpoint_path: Union[str, Path],
    val_pairs_path: Path,
    test_pairs_path: Path,
    dataset_path: Path,
    output_dir: Union[str, Path] = "~/Data/INER/outputs/evaluation/calibration",
    tau_dec: float = 0.12,
    batch_size: int = 32,
    max_length: int = 512,
    n_bins: int = 10,
) -> dict:
    """Vía A — calibra el CE (temperature scaling) y reporta incertidumbre por vínculo.

    Procedimiento POST-HOC sobre el CE ya entrenado (NO re-entrena):
      1. Corre el CE sobre VALIDACIÓN → logits crudos.
      2. Ajusta la temperatura T minimizando NLL sobre val.
      3. Calcula ECE antes/después + reliability curves (val).
      4. Corre el CE sobre TEST → logits → prob. calibrada → incertidumbre por par
         (entropía sobre prob. calibrada; margen |p - tau_dec| sobre prob. cruda).

    Artefactos persistidos:
      - <checkpoint>/calibration.json                          (T + ECE val)
      - <output_dir>/calibration_results_<ckpt>.json           (resumen completo)
      - <output_dir>/uncertainty_<ckpt>_test.parquet           (incertidumbre por par)

    Returns:
        dict con T, ECE val (antes/después), reliability curves y stats de test.
    """
    from record_linkage.evaluation.calibration import (
        apply_temperature, binary_entropy, decision_margin,
        expected_calibration_error, fit_temperature, reliability_curve,
    )

    checkpoint_path = Path(checkpoint_path).expanduser()
    val_pairs_path  = Path(val_pairs_path).expanduser()
    test_pairs_path = Path(test_pairs_path).expanduser()
    dataset_path    = Path(dataset_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Si el checkpoint es un subdir convencional (best/final/...), usar el nombre del run
    ckpt_name = checkpoint_path.name
    if ckpt_name in ("best", "final", "checkpoint", "last"):
        ckpt_name = checkpoint_path.parent.name

    print(f"\n{'='*60}\nCalibración CE (Vía A): {ckpt_name}\n{'='*60}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_crossencoder(checkpoint_path)
    model = model.to(device)
    records_df = pd.read_parquet(dataset_path)

    def _logits_and_labels(pairs_path: Path):
        df = pd.read_parquet(pairs_path)
        texts_a, texts_b = _texts_from_pairs(df, records_df)
        logits = score_pairs(
            model, tokenizer, texts_a, texts_b,
            batch_size=batch_size, max_length=max_length, device=device,
            return_logits=True,
        ).numpy()
        labels = df["label"].astype(int).values
        return df, logits, labels

    # --- 1-3. Ajuste de T y diagnóstico sobre validación ---
    print(f"\n  [val] cargando y puntuando: {val_pairs_path}")
    _, logits_val, labels_val = _logits_and_labels(val_pairs_path)
    probs_val_raw = apply_temperature(logits_val, 1.0)  # sin calibrar (T=1)

    temperature = fit_temperature(logits_val, labels_val)
    probs_val_cal = apply_temperature(logits_val, temperature)

    ece_before = expected_calibration_error(probs_val_raw, labels_val, n_bins=n_bins)
    ece_after  = expected_calibration_error(probs_val_cal, labels_val, n_bins=n_bins)
    print(f"  T ajustada = {temperature:.4f}  |  ECE val: {ece_before:.4f} → {ece_after:.4f}")

    # --- 4. Incertidumbre por vínculo sobre test ---
    print(f"\n  [test] cargando y puntuando: {test_pairs_path}")
    df_test, logits_test, labels_test = _logits_and_labels(test_pairs_path)
    probs_test_raw = apply_temperature(logits_test, 1.0)
    probs_test_cal = apply_temperature(logits_test, temperature)
    entropia = binary_entropy(probs_test_cal)               # incertidumbre (post-calibración)
    margen   = decision_margin(probs_test_raw, tau_dec)     # incertidumbre (pre-calibración)

    # Parquet por-vínculo
    uncertainty_df = pd.DataFrame({
        "record_id_a":   df_test["record_id_a"].values,
        "record_id_b":   df_test["record_id_b"].values,
        "label":         labels_test,
        "score_raw":     probs_test_raw,
        "prob_calibrada": probs_test_cal,
        "entropia":      entropia,
        "margen":        margen,
    })
    parquet_path = output_dir / f"uncertainty_{ckpt_name}_test.parquet"
    uncertainty_df.to_parquet(parquet_path, index=False)

    results = {
        "checkpoint":        str(checkpoint_path),
        "temperature":       round(float(temperature), 6),
        "tau_dec":           tau_dec,
        "val_pairs_path":    str(val_pairs_path),
        "test_pairs_path":   str(test_pairs_path),
        "ece_val_before":    round(ece_before, 6),
        "ece_val_after":     round(ece_after, 6),
        "ece_test_after":    round(expected_calibration_error(probs_test_cal, labels_test, n_bins=n_bins), 6),
        "reliability_val_before": reliability_curve(probs_val_raw, labels_val, n_bins=n_bins),
        "reliability_val_after":  reliability_curve(probs_val_cal, labels_val, n_bins=n_bins),
        "entropy_test_mean": round(float(entropia.mean()), 6),
        "margin_test_mean":  round(float(margen.mean()), 6),
        "n_val":             int(len(labels_val)),
        "n_test":            int(len(labels_test)),
        "uncertainty_parquet": str(parquet_path),
    }

    # --- Persistencia ---
    results_path = output_dir / f"calibration_results_{ckpt_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ckpt_calib_path = checkpoint_path / "calibration.json"
    with open(ckpt_calib_path, "w") as f:
        json.dump({
            "temperature":    round(float(temperature), 6),
            "fitted_on_split": "val",
            "ece_val_before": round(ece_before, 6),
            "ece_val_after":  round(ece_after, 6),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Guardado:")
    print(f"    T → {ckpt_calib_path}")
    print(f"    resultados → {results_path}")
    print(f"    incertidumbre por par → {parquet_path}")
    return results
