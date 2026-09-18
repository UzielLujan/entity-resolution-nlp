import numpy as np
import pandas as pd
import pytest

from record_linkage.baselines import ClassicalReranker, select_f1_threshold


@pytest.fixture
def records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 5, 6],
            "text": [
                "nombre juan perez edad 40 monterrey",
                "nombre juan peres edad 40 monterrey",
                "nombre maria lopez edad 55 puebla",
                "nombre maria lopes edad 55 puebla",
                "nombre carlos diaz edad 30 merida",
                "nombre ana ruiz edad 28 toluca",
            ],
        }
    )


@pytest.fixture
def train_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id_a": [1, 3, 1, 1, 3, 5],
            "record_id_b": [2, 4, 3, 5, 6, 6],
            "label": [1, 1, 0, 0, 0, 0],
            "similarity": [0.9, 0.9, 0.2, 0.1, 0.1, 0.1],
        }
    )


def _baseline() -> ClassicalReranker:
    return ClassicalReranker(
        word_max_features=200,
        char_max_features=300,
        min_df=1,
        max_iter=200,
        seed=7,
    )


def test_scores_are_probabilities_and_order_invariant(records, train_pairs):
    model = _baseline()
    summary = model.fit(records, train_pairs)
    forward = pd.DataFrame({"record_id_a": [1, 3], "record_id_b": [2, 4]})
    reverse = pd.DataFrame({"record_id_a": [2, 4], "record_id_b": [1, 3]})

    forward_scores, _ = model.score_pairs(records, forward)
    reverse_scores, _ = model.score_pairs(records, reverse)

    assert summary["n_train_pairs"] == len(train_pairs)
    assert np.all((forward_scores >= 0.0) & (forward_scores <= 1.0))
    np.testing.assert_allclose(forward_scores, reverse_scores, rtol=1e-6, atol=1e-7)


def test_biencoder_similarity_is_not_used(records, train_pairs):
    model = _baseline()
    model.fit(records, train_pairs)
    pairs = train_pairs.iloc[:3].copy()
    low_similarity = pairs.assign(similarity=0.0)
    high_similarity = pairs.assign(similarity=1.0)

    low_scores, _ = model.score_pairs(records, low_similarity)
    high_scores, _ = model.score_pairs(records, high_similarity)

    np.testing.assert_allclose(low_scores, high_scores)


def test_vocabulary_is_fitted_only_on_train_records(records, train_pairs):
    held_out = pd.DataFrame(
        {"record_id": [999], "text": ["exclusivevalidationtoken"]}
    )
    records_with_held_out = pd.concat((records, held_out), ignore_index=True)
    model = _baseline()

    model.fit(records_with_held_out, train_pairs)

    assert "exclusivevalidationtoken" not in model.word_vectorizer.vocabulary_


def test_model_round_trip(tmp_path, records, train_pairs):
    model = _baseline()
    model.fit(records, train_pairs)
    pairs = train_pairs.iloc[:2]
    expected, _ = model.score_pairs(records, pairs)
    path = tmp_path / "model.pkl"

    model.save(path)
    loaded = ClassicalReranker.load(path)
    observed, _ = loaded.score_pairs(records, pairs)

    np.testing.assert_allclose(observed, expected)


def test_missing_record_is_rejected(records, train_pairs):
    model = _baseline()
    model.fit(records, train_pairs)
    invalid_pairs = pd.DataFrame({"record_id_a": [1], "record_id_b": [999]})

    with pytest.raises(ValueError, match="ausentes"):
        model.score_pairs(records, invalid_pairs)


def test_select_f1_threshold_uses_validation_scores():
    result = select_f1_threshold(
        scores=np.array([0.05, 0.15, 0.85, 0.95]),
        labels=np.array([0, 0, 1, 1]),
        n_thresholds=11,
    )

    assert result["best_f1"] == 1.0
    assert 0.2 <= result["best_threshold"] <= 0.8
