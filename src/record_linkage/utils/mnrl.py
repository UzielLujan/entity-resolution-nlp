"""Utilidades de diagnóstico para entrenamiento MNRL."""

import json
from pathlib import Path

import torch


def dump_mnrl_batch(
    anchors: list[str],
    positives: list[str],
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    step: int,
    epoch: int,
    viz_dir: Path,
) -> None:
    """Guarda textos y matriz de similitud de un batch MNRL en JSON.

    La matriz sim[i][j] es la similitud coseno entre anchor_i y positive_j.
    La diagonal (i == j) corresponde a los pares positivos del batch.

    Args:
        anchors:    Lista de textos serializados del lado A (anclas).
        positives:  Lista de textos serializados del lado B (positivos).
        emb_a:      Embeddings L2-normalizados de anchors, shape (B, D).
        emb_b:      Embeddings L2-normalizados de positives, shape (B, D).
        step:       Paso dentro de la época (base 0).
        epoch:      Número de época (base 1).
        viz_dir:    Directorio donde se guardan los JSONs (se crea si no existe).
    """
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        sim = (emb_a @ emb_b.T).cpu().tolist()

    payload = {
        "epoch": epoch,
        "step": step,
        "batch_size": len(anchors),
        "anchors": list(anchors),
        "positives": list(positives),
        "similarity_matrix": sim,
    }

    out_path = viz_dir / f"epoch{epoch:02d}_step{step:05d}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    txt_path = render_sim_matrix(out_path)
    print(f"  [viz] batch guardado → {out_path.name} | {txt_path.name}")


def render_sim_matrix(json_path: Path) -> Path:
    """Lee un JSON de dump_mnrl_batch y guarda la matriz de similitud como .txt legible.

    Formato de salida:
        - Encabezado con epoch, step y batch_size
        - Cabeceras de columna A0…A(B-1)
        - Filas P0…P(B-1) con valores redondeados a 3 decimales
        - Diagonal (par positivo) marcada con corchetes

    Args:
        json_path: Ruta al JSON generado por dump_mnrl_batch.

    Returns:
        Ruta al .txt generado (mismo directorio, mismo nombre base).
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    epoch = data["epoch"]
    step  = data["step"]
    sim   = data["similarity_matrix"]
    b     = len(sim)

    col_w = 8  # ancho de cada celda
    header = f"MNRL — epoch {epoch:02d}  step {step:05d}  (batch={b})\n\n"

    col_labels = "".join(f"{'A'+str(j):>{col_w}}" for j in range(b))
    lines = [header, " " * 4 + col_labels + "\n"]

    for i, row in enumerate(sim):
        cells = []
        for j, val in enumerate(row):
            formatted = f"{val:.3f}"
            cell = f"[{formatted}]" if i == j else f" {formatted} "
            cells.append(f"{cell:>{col_w}}")
        lines.append(f"P{i:<3}" + "".join(cells) + "\n")

    out_path = json_path.with_suffix(".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return out_path
