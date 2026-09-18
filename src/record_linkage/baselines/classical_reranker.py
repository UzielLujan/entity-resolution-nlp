"""Reranker clasico con TF-IDF y regresion logistica.

El reranker usa exclusivamente el texto serializado y las etiquetas. Si los
pares incluyen similitud de recuperacion, esta se ignora deliberadamente.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from record_linkage.baselines.tfidf_encoder import TfidfEncoder


PAIR_COLUMNS = {"record_id_a", "record_id_b"}
RECORD_COLUMNS = {"record_id", "text"}


def select_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 51,
) -> dict:
    """Selecciona en validacion el umbral positivo con mayor F1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape:
        raise ValueError("scores y labels deben tener la misma forma")
    if n_thresholds < 2:
        raise ValueError("n_thresholds debe ser al menos 2")

    best_threshold = 0.5
    best_f1 = -1.0
    thresholds = np.linspace(0.0, 1.0, n_thresholds)[1:]
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        current_f1 = f1_score(labels, predictions, zero_division=0)
        if current_f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(current_f1)

    return {
        "best_threshold": round(best_threshold, 4),
        "best_f1": round(best_f1, 4),
        "n_thresholds": len(thresholds),
    }


class ClassicalReranker:
    """TF-IDF por registro y clasificador lineal sobre caracteristicas del par.

    Las caracteristicas son simetricas respecto al orden de los registros:
    diferencia absoluta y producto elemento a elemento del TF-IDF de palabras,
    mas similitud coseno de palabras y caracteres. Los n-gramas de caracteres
    solo aportan su coseno para mantener acotado el uso de memoria.
    """

    def __init__(
        self,
        word_max_features: int = 50_000,
        char_max_features: int = 50_000,
        min_df: int = 2,
        c: float = 1.0,
        max_iter: int = 1_000,
        seed: int = 42,
        encoder: TfidfEncoder | None = None,
    ) -> None:
        self.encoder = encoder or TfidfEncoder(
            word_max_features=word_max_features,
            char_max_features=char_max_features,
            min_df=min_df,
        )
        # Alias conservados para inspeccion y compatibilidad con pruebas existentes.
        self.word_vectorizer = self.encoder.word_vectorizer
        self.char_vectorizer = self.encoder.char_vectorizer
        self.classifier = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=max_iter,
            random_state=seed,
        )
        self.fit_summary: dict = {}
        self._fitted = False

    def fit(
        self,
        records_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
        fit_encoder: bool = True,
    ) -> dict:
        """Ajusta vocabularios y regresion usando exclusivamente el split train."""
        self._validate_records(records_df)
        self._validate_pairs(pairs_df, require_labels=True)

        train_ids = self._referenced_ids(pairs_df)
        train_texts = self._texts_for_ids(records_df, train_ids)
        if fit_encoder:
            self.encoder.fit(train_texts)
        else:
            self.encoder.vocabulary_sizes

        features, diagnostics = self._pair_features(records_df, pairs_df)
        labels = pairs_df["label"].astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("train debe contener ambas clases: 0 y 1")

        self.classifier.fit(features, labels)
        self._fitted = True
        self.fit_summary = {
            "n_train_records": int(len(train_ids)),
            "n_train_pairs": int(len(pairs_df)),
            "n_positive_pairs": int(labels.sum()),
            "word_vocabulary_size": int(self.encoder.vocabulary_sizes["word"]),
            "char_vocabulary_size": int(self.encoder.vocabulary_sizes["char"]),
            "pair_feature_shape": [int(value) for value in features.shape],
            "pair_feature_nnz": int(features.nnz),
            "mean_word_cosine": float(diagnostics["word_cosine"].mean()),
            "mean_char_cosine": float(diagnostics["char_cosine"].mean()),
            "classifier_iterations": int(self.classifier.n_iter_[0]),
        }
        return self.fit_summary

    def score_pairs(
        self,
        records_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Devuelve P(match) y similitudes lexicas auxiliares por par."""
        if not self._fitted:
            raise RuntimeError("El baseline debe ajustarse antes de puntuar pares")
        self._validate_records(records_df)
        self._validate_pairs(pairs_df, require_labels=False)
        features, diagnostics = self._pair_features(records_df, pairs_df)
        scores = self.classifier.predict_proba(features)[:, 1]
        return scores, diagnostics

    def save(self, path: Path) -> None:
        """Serializa vectorizadores, clasificador y configuracion ajustada."""
        if not self._fitted:
            raise RuntimeError("No se puede guardar un baseline sin ajustar")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "ClassicalReranker":
        """Carga un baseline serializado por :meth:`save`."""
        with Path(path).open("rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError(f"El artefacto no contiene {cls.__name__}")
        return model

    def _pair_features(
        self,
        records_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
    ) -> tuple[csr_matrix, dict[str, np.ndarray]]:
        record_ids = self._referenced_ids(pairs_df)
        texts = self._texts_for_ids(records_df, record_ids)
        id_to_row = {int(record_id): row for row, record_id in enumerate(record_ids)}
        rows_a = np.fromiter(
            (id_to_row[int(value)] for value in pairs_df["record_id_a"]),
            dtype=np.int64,
            count=len(pairs_df),
        )
        rows_b = np.fromiter(
            (id_to_row[int(value)] for value in pairs_df["record_id_b"]),
            dtype=np.int64,
            count=len(pairs_df),
        )

        word_matrix, char_matrix = self.encoder.transform_components(texts)
        word_a = word_matrix[rows_a]
        word_b = word_matrix[rows_b]
        word_difference = word_a - word_b
        np.abs(word_difference.data, out=word_difference.data)
        word_product = word_a.multiply(word_b)
        word_cosine = np.asarray(word_product.sum(axis=1)).ravel().astype(np.float32)

        char_product = char_matrix[rows_a].multiply(char_matrix[rows_b])
        char_cosine = np.asarray(char_product.sum(axis=1)).ravel().astype(np.float32)

        scalar_features = csr_matrix(np.column_stack((word_cosine, char_cosine)))
        features = hstack(
            (word_difference, word_product, scalar_features),
            format="csr",
            dtype=np.float32,
        )
        diagnostics = {
            "word_cosine": word_cosine,
            "char_cosine": char_cosine,
        }
        return features, diagnostics

    @staticmethod
    def _validate_records(records_df: pd.DataFrame) -> None:
        missing = RECORD_COLUMNS - set(records_df.columns)
        if missing:
            raise ValueError(f"Dataset de registros sin columnas requeridas: {missing}")
        if records_df["record_id"].duplicated().any():
            raise ValueError("record_id debe ser unico en el dataset de registros")
        if records_df["text"].isna().any():
            raise ValueError("text no puede contener valores nulos")

    @staticmethod
    def _validate_pairs(pairs_df: pd.DataFrame, require_labels: bool) -> None:
        required = PAIR_COLUMNS | ({"label"} if require_labels else set())
        missing = required - set(pairs_df.columns)
        if missing:
            raise ValueError(f"Parquet de pares sin columnas requeridas: {missing}")
        if pairs_df.empty:
            raise ValueError("El conjunto de pares no puede estar vacio")
        if require_labels and not set(pairs_df["label"].unique()).issubset({0, 1}):
            raise ValueError("label debe contener exclusivamente 0 y 1")

    @staticmethod
    def _referenced_ids(pairs_df: pd.DataFrame) -> np.ndarray:
        return np.unique(
            np.concatenate(
                (
                    pairs_df["record_id_a"].astype(np.int64).to_numpy(),
                    pairs_df["record_id_b"].astype(np.int64).to_numpy(),
                )
            )
        )

    @staticmethod
    def _texts_for_ids(records_df: pd.DataFrame, record_ids: np.ndarray) -> list[str]:
        indexed = records_df.set_index("record_id", verify_integrity=True)
        missing = np.setdiff1d(record_ids, indexed.index.to_numpy())
        if len(missing):
            preview = ", ".join(str(value) for value in missing[:5])
            raise ValueError(f"Pares con record_id ausentes del dataset: {preview}")
        return indexed.loc[record_ids, "text"].astype(str).tolist()
