"""Utilidades de la loss BCEWithLogits para el Cross-Encoder.

Esta loss combina sigmoid + BCE en una sola operación estable numéricamente:
no aplicamos sigmoid manualmente al output del modelo durante training,
sino que pasamos los logits directos a `BCEWithLogitsLoss`.
BCE Significa Binary Cross Entropy
Para inferencia/eval, sí aplicamos `torch.sigmoid(logits)` para convertir a
probabilidades (ver `models.crossencoder.score_pairs`).
"""

import torch
from torch import nn


def make_bce_loss(pos_weight: float = 1.0) -> nn.BCEWithLogitsLoss:
    """BCEWithLogitsLoss con peso opcional para la clase positiva.

    Args:
        pos_weight: si > 1.0, sube el peso del error en falsos negativos (útil cuando
                    los positivos son raros). Default 1.0 = balance.
    """
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
