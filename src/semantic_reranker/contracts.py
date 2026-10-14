from __future__ import annotations

from typing import Final

import numpy as np


LABEL_TO_ID: Final[dict[str, int]] = {
    "E": 0,
    "S": 1,
    "C": 2,
    "I": 3,
}

ID_TO_LABEL: Final[dict[int, str]] = {
    value: key
    for key, value in LABEL_TO_ID.items()
}

GAIN_BY_LABEL: Final[dict[str, float]] = {
    "E": 1.0,
    "S": 0.1,
    "C": 0.01,
    "I": 0.0,
}

GAIN_VECTOR: Final[np.ndarray] = np.array(
    [
        GAIN_BY_LABEL[ID_TO_LABEL[index]]
        for index in range(len(ID_TO_LABEL))
    ],
    dtype=np.float64,
)


def expected_esci_gain(
    probabilities: np.ndarray,
) -> np.ndarray:
    """Convert four-class probabilities into expected ESCI gain."""

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape "
            "(candidate_count, class_count)."
        )

    if probabilities.shape[1] != len(LABEL_TO_ID):
        raise ValueError(
            f"Expected {len(LABEL_TO_ID)} ESCI classes, got {probabilities.shape[1]}."
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("ESCI probabilities must contain only finite values.")

    return probabilities @ GAIN_VECTOR