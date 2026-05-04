"""Bi-Encoder siamés basado en Sentence-Transformers.

Arquitectura: Transformer (backbone intercambiable) + MeanPooling.
Soporta inferencia zero-shot y fine-tuning con MNRL.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer


def _load_transformer(model_path: str) -> Transformer:
    """Carga un backbone como Transformer, con fallback para modelos sin model_type."""
    try:
        return Transformer(model_path)
    except ValueError as e:
        if "model_type" not in str(e):
            raise

    from transformers import RobertaModel, RobertaTokenizerFast
    hf_model = RobertaModel.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    patched_dir = Path(tempfile.mkdtemp(prefix="st_patch_"))
    hf_model.config.model_type = "roberta"
    hf_model.save_pretrained(patched_dir)
    tokenizer.save_pretrained(patched_dir)
    return Transformer(str(patched_dir))


def build_biencoder(model_name_or_path: Union[str, Path]) -> SentenceTransformer:
    """Construye un Bi-Encoder desde un backbone HuggingFace o ruta local.

    Apila Transformer + MeanPooling — el backbone es completamente intercambiable.
    Para modelos ya empaquetados como SentenceTransformer (tienen sentence_bert_config.json),
    se cargan directamente sin reconstruir el stack.

    Args:
        model_name_or_path: ID de HuggingFace Hub o ruta local al modelo.
            Ejemplos:
                'dccuchile/bert-base-spanish-wwm-cased'   (BETO, desde Hub)
                '~/Data/INER/models/pretrained/BETO'      (BETO, local)
                '~/Data/INER/models/pretrained/RoBERTa-bne'

    Returns:
        SentenceTransformer listo para encode() o fine-tuning con MNRL.
    """
    model_path = str(model_name_or_path)

    # Intentar cargar como SentenceTransformer nativo (tiene sentence_bert_config.json)
    try:
        model = SentenceTransformer(model_path)
        return model
    except Exception:
        pass

    # Construir desde backbone crudo: Transformer + MeanPooling
    transformer = _load_transformer(model_path)
    pooling = Pooling(
        transformer.get_embedding_dimension(),
        pooling_mode="mean",
    )
    return SentenceTransformer(modules=[transformer, pooling])


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Codifica una lista de textos a embeddings normalizados.

    Args:
        model: Bi-Encoder cargado con build_biencoder().
        texts: Lista de textos serializados.
        batch_size: Tamaño de batch. 32 es seguro para RTX 3050 6GB.
        normalize: Si True, normaliza a norma unitaria (producto punto == coseno).
        show_progress: Muestra barra de progreso.

    Returns:
        np.ndarray de shape (n_textos, dim_embedding).
    """
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
