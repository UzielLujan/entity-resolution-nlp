"""Representacion TF-IDF reutilizable para recuperacion y reranking clasicos."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted


class TfidfEncoder:
    """Combina n-gramas de palabras y caracteres en vectores dispersos L2."""

    def __init__(
        self,
        word_max_features: int = 50_000,
        char_max_features: int = 50_000,
        min_df: int = 2,
        word_weight: float = 0.5,
    ) -> None:
        if not 0.0 <= word_weight <= 1.0:
            raise ValueError("word_weight debe estar en [0, 1]")
        self.word_weight = word_weight
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            strip_accents="unicode",
            min_df=min_df,
            max_features=word_max_features,
            sublinear_tf=True,
            dtype=np.float32,
            norm="l2",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            strip_accents="unicode",
            min_df=min_df,
            max_features=char_max_features,
            sublinear_tf=True,
            dtype=np.float32,
            norm="l2",
        )

    def fit(self, texts: list[str]) -> "TfidfEncoder":
        """Ajusta ambos vocabularios e IDF con el corpus de entrenamiento."""
        if not texts:
            raise ValueError("Se requiere al menos un texto para ajustar TF-IDF")
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)
        return self

    def transform_components(self, texts: list[str]) -> tuple[csr_matrix, csr_matrix]:
        """Devuelve por separado las matrices TF-IDF de palabras y caracteres."""
        return self.transform_word(texts), self.transform_char(texts)

    def transform_word(self, texts: list[str]) -> csr_matrix:
        self._check_fitted()
        return self.word_vectorizer.transform(texts).tocsr()

    def transform_char(self, texts: list[str]) -> csr_matrix:
        self._check_fitted()
        return self.char_vectorizer.transform(texts).tocsr()

    def transform(self, texts: list[str]) -> csr_matrix:
        """Devuelve la representacion combinada y normalizada para recuperacion."""
        word_matrix, char_matrix = self.transform_components(texts)
        return self.combine_components(word_matrix, char_matrix)

    def combine_components(
        self,
        word_matrix: csr_matrix,
        char_matrix: csr_matrix,
    ) -> csr_matrix:
        """Combina matrices ya transformadas sin repetir la vectorizacion."""
        word_scale = np.sqrt(self.word_weight)
        char_scale = np.sqrt(1.0 - self.word_weight)
        combined = hstack(
            (word_matrix * word_scale, char_matrix * char_scale),
            format="csr",
            dtype=np.float32,
        )
        return normalize(combined, norm="l2", copy=False).tocsr()

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocabulary_sizes(self) -> dict[str, int]:
        self._check_fitted()
        return {
            "word": len(self.word_vectorizer.vocabulary_),
            "char": len(self.char_vectorizer.vocabulary_),
        }

    def _check_fitted(self) -> None:
        check_is_fitted(self.word_vectorizer, "vocabulary_")
        check_is_fitted(self.char_vectorizer, "vocabulary_")
