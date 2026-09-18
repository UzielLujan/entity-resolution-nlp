"""Baselines clásicos para comparar las etapas del pipeline neuronal."""

from record_linkage.baselines.classical_reranker import (
    ClassicalReranker,
    select_f1_threshold,
)
from record_linkage.baselines.tfidf_encoder import TfidfEncoder

__all__ = ["ClassicalReranker", "TfidfEncoder", "select_f1_threshold"]
