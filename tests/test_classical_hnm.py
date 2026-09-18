import pandas as pd

from record_linkage.baselines import TfidfEncoder
from record_linkage.baselines.hard_pair_mining import mine_tfidf_pairs


def test_mining_produces_unique_cross_source_pairs():
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

    pairs = mine_tfidf_pairs(records, vectors, top_k=2, batch_size=2)
    source_by_id = records.set_index("record_id")["source_db"]

    assert set(pairs["label"]) == {0, 1}
    assert not pairs.duplicated(["record_id_a", "record_id_b"]).any()
    assert all(
        source_by_id[left] != source_by_id[right]
        for left, right in zip(pairs["record_id_a"], pairs["record_id_b"])
    )
    assert ((pairs["label"] == 1) == pairs["type"].eq("positive")).all()
