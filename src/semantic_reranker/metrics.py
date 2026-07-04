from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from semantic_reranker.contracts import GAIN_BY_LABEL


def dcg_at_k(
    gains: Sequence[float],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    gains_array = np.asarray(
        list(gains)[:k],
        dtype=np.float64,
    )

    if gains_array.size == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, gains_array.size + 2)
    )

    return float(np.sum(gains_array / discounts))


def ndcg_at_k(
    labels: Sequence[str],
    scores: Sequence[float],
    k: int = 10,
) -> float:
    if len(labels) != len(scores):
        raise ValueError(
            "labels and scores must have the same length."
        )

    if not labels:
        return 0.0

    gains = np.asarray(
        [GAIN_BY_LABEL[label] for label in labels],
        dtype=np.float64,
    )

    scores_array = np.asarray(
        scores,
        dtype=np.float64,
    )

    predicted_order = np.argsort(-scores_array)
    ideal_order = np.argsort(-gains)

    observed_dcg = dcg_at_k(
        gains[predicted_order],
        k,
    )

    ideal_dcg = dcg_at_k(
        gains[ideal_order],
        k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return observed_dcg / ideal_dcg


def top1_label(
    labels: Sequence[str],
    scores: Sequence[float],
) -> str:
    if len(labels) != len(scores):
        raise ValueError(
            "labels and scores must have the same length."
        )

    if not labels:
        raise ValueError("Cannot select from an empty query.")

    return labels[int(np.argmax(scores))]