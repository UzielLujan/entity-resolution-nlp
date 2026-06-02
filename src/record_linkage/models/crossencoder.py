"""Cross-Encoder DITTO-style para Etapa 2 (re-ranking).

Recibe pares (text_a, text_b) y produce un score escalar por par. Internamente usa
AutoModelForSequenceClassification con num_labels=1 (regression head) y se entrena
con BCEWithLogitsLoss — el head emite logit, sigmoid lo convierte a probabilidad.

A diferencia del Bi-Encoder (dos torres independientes que producen embeddings y
luego se compara con coseno), el Cross-Encoder concatena los dos textos en una sola
secuencia `[CLS] text_a [SEP] text_b [SEP]` con atención cruzada — el modelo puede
"ver" ambos registros simultáneamente, lo que permite detectar diferencias sutiles
(ej. edad 45 vs 46, diagnósticos distintos) que el BE no distingue.

Arquitectura: BERT-style con head de regresión (num_labels=1), no clasificación
binaria de 2 clases. Razón: BCEWithLogitsLoss es numéricamente más estable y la
salida sigmoid se interpreta directamente como probabilidad de match.
"""

from pathlib import Path
from typing import Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_crossencoder(model_path: Union[str, Path], num_labels: int = 1):
    """Carga modelo + tokenizer desde un directorio local de HuggingFace.

    Returns:
        (model, tokenizer). El modelo NO incluye sigmoid al final; aplicar
        torch.sigmoid(model(...).logits) en inferencia.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return model, tokenizer


def encode_pairs(
    tokenizer,
    texts_a: list[str],
    texts_b: list[str],
    max_length: int = 512,
    device: Union[str, torch.device] = "cpu",
) -> dict:
    """Tokeniza pares como `[CLS] text_a [SEP] text_b [SEP]` con token_type_ids distintos.

    En registros largos del INER, max_length=512 deja ~256 tokens por lado tras tokens
    especiales — las variantes skip_null del 2×2 caben mejor que keep_null.
    """
    return tokenizer(
        texts_a, texts_b,
        padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)


@torch.no_grad()
def score_pairs(
    model,
    tokenizer,
    texts_a: list[str],
    texts_b: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: Union[str, torch.device] = "cuda",
    return_logits: bool = False,
) -> torch.Tensor:
    """Aplica el CE a una lista de pares.

    Args:
        return_logits: si False (default), devuelve probabilidades sigmoid(logit) —
                       para inferencia y evaluación. Si True, devuelve los logits
                       crudos (pre-sigmoide) — necesarios para calibración
                       (temperature scaling), que opera sobre logits.

    El training loop usa logits directos con BCEWithLogitsLoss por estabilidad numérica.
    """
    model.eval()
    model.to(device)

    all_scores = []
    for i in range(0, len(texts_a), batch_size):
        batch_a = texts_a[i : i + batch_size]
        batch_b = texts_b[i : i + batch_size]
        inputs = encode_pairs(tokenizer, batch_a, batch_b, max_length=max_length, device=device)
        logits = model(**inputs).logits.squeeze(-1)
        out = logits if return_logits else torch.sigmoid(logits)
        all_scores.append(out.cpu())

    return torch.cat(all_scores)
