import numpy as np
import pandas as pd

from record_linkage.evaluation.classical_pipeline_eval import evaluate_classical_pipeline


class _FixedScoreModel:
    def score_pairs(self, records_df, pairs_df):
        scores = pairs_df["similarity"].to_numpy(dtype=float)
        diagnostics = {
            "word_cosine": scores.copy(),
            "char_cosine": scores.copy(),
        }
        return scores, diagnostics


def test_hard_positives_count_as_end_to_end_false_negatives():
    records = pd.DataFrame({"record_id": [1, 2], "text": ["a", "b"]})
    val_pairs = pd.DataFrame(
        {
            "record_id_a": [1, 1, 1],
            "record_id_b": [2, 2, 2],
            "label": [1, 0, 1],
            "similarity": [0.9, 0.1, 0.0],
            "type": ["positive", "hard_negative", "hard_positive"],
        }
    )

    metrics, predictions = evaluate_classical_pipeline(
        _FixedScoreModel(),
        records,
        val_pairs,
        val_pairs,
        n_thresholds=11,
    )

    assert metrics["test_retrieved_pairs"]["fn"] == 0
    assert metrics["test_end_to_end"]["fn"] == 1
    assert metrics["test_false_negatives_from_retrieval"] == 1
    assert predictions.loc[predictions["type"] == "hard_positive", "score"].item() == 0
    assert predictions.loc[
        predictions["type"] == "hard_positive", "prediction"
    ].item() == 0
    assert np.isclose(metrics["test_end_to_end"]["recall"], 0.5)
