"""ESCI gain conversion contracts and invalid probability inputs."""

import numpy as np
import pytest

from semantic_reranker.contracts import expected_esci_gain


def test_expected_gain_matches_class_gain_for_one_hot_probabilities():
    scores = expected_esci_gain(np.eye(4))
    np.testing.assert_allclose(scores, np.array([1.0, 0.1, 0.01, 0.0]))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_expected_gain_rejects_nonfinite_probabilities(invalid):
    probabilities = np.array([[1.0, invalid, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        expected_esci_gain(probabilities)


def test_expected_gain_rejects_wrong_class_dimension():
    with pytest.raises(ValueError, match="Expected 4 ESCI classes"):
        expected_esci_gain(np.ones((2, 3)))
