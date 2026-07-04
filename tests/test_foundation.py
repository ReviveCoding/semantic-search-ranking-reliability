from __future__ import annotations

import numpy as np
import pytest

from semantic_reranker.contracts import (
    GAIN_VECTOR,
    LABEL_TO_ID,
    expected_esci_gain,
)
from semantic_reranker.metrics import (
    ndcg_at_k,
    top1_label,
)
from semantic_reranker.text import (
    build_product_text,
    build_query_product_pair,
)


def test_label_and_gain_contract() -> None:
    assert LABEL_TO_ID == {
        "E": 0,
        "S": 1,
        "C": 2,
        "I": 3,
    }

    assert GAIN_VECTOR.tolist() == [
        1.0,
        0.1,
        0.01,
        0.0,
    ]


def test_expected_gain() -> None:
    probabilities = np.array(
        [[0.0, 1.0, 0.0, 0.0]]
    )

    assert expected_esci_gain(
        probabilities
    ).tolist() == [0.1]


def test_expected_gain_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        expected_esci_gain(
            np.array([0.1, 0.2, 0.3, 0.4])
        )


def test_product_text_uses_allowed_fields() -> None:
    text = build_product_text(
        {
            "product_title": "Wireless Mouse",
            "product_brand": "Example",
            "product_color": "Black",
            "product_bullet_point": "USB receiver included",
        }
    )

    assert "[TITLE] Wireless Mouse" in text
    assert "[BRAND] Example" in text
    assert "[COLOR] Black" in text
    assert "[BULLETS] USB receiver included" in text


def test_pair_rejects_missing_query() -> None:
    with pytest.raises(ValueError):
        build_query_product_pair(
            {
                "query": " ",
                "product_title": "Keyboard",
            }
        )


def test_ranking_metrics_follow_registered_gains() -> None:
    assert ndcg_at_k(
        ["E", "S", "I"],
        [3.0, 2.0, 1.0],
        k=3,
    ) == 1.0

    assert top1_label(
        ["I", "E"],
        [0.2, 0.9],
    ) == "E"