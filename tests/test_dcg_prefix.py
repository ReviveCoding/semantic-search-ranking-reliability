import numpy as np
import pytest

from semantic_reranker.metrics import dcg_at_k


@pytest.mark.parametrize(
    "values", [[3.0, 1.0, 0.0], (3.0, 1.0, 0.0), np.array([3.0, 1.0, 0.0])]
)
@pytest.mark.parametrize("k", [1, 2, 5, 10**30])
def test_dcg_uses_same_prefix_for_supported_sequences(values, k):
    prefix = np.asarray(list(values)[:k], dtype=np.float64)
    expected = float(np.sum(prefix / np.log2(np.arange(2, len(prefix) + 2))))
    assert dcg_at_k(values, k) == expected


def test_dcg_does_not_traverse_unused_tail():
    class PrefixOnly(list):
        def __iter__(self):
            raise AssertionError("Full-sequence traversal is unnecessary")

    assert dcg_at_k(PrefixOnly([3.0, 1.0, 0.0]), 1) == 3.0
