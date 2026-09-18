import pandas as pd

from record_linkage.baselines import TfidfEncoder
from record_linkage.evaluation.tfidf_retrieval_eval import evaluate_sparse_retrieval


def test_tfidf_retrieval_finds_matching_entities():
    records = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "entity_id": [10, 10, 20, 20],
            "source_db": ["A", "B", "A", "B"],
            "text": ["juan perez", "juan peres", "maria lopez", "maria lopes"],
        }
    )
    encoder = TfidfEncoder(
        word_max_features=100,
        char_max_features=200,
        min_df=1,
    )
    vectors = encoder.fit_transform(records["text"].tolist())

    evaluation = evaluate_sparse_retrieval(records, vectors, k_values=[1, 2])

    assert evaluation["macro_average"]["Hit@1"] == 1.0
    assert evaluation["macro_average"]["Recall@2"] == 1.0
    assert set(evaluation["results"]) == {"A -> B", "B -> A"}


def test_encoder_combined_vectors_are_l2_normalized():
    texts = ["registro uno", "registro dos", "registro tres"]
    encoder = TfidfEncoder(
        word_max_features=100,
        char_max_features=200,
        min_df=1,
    )

    vectors = encoder.fit_transform(texts)
    squared_norms = vectors.multiply(vectors).sum(axis=1).A1

    assert all(abs(norm - 1.0) < 1e-6 for norm in squared_norms)
