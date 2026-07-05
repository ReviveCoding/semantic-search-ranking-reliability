from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.reliability_report import (
    build_support_slice_table,
    classify_top1_change,
    support_bucket,
)


def test_support_bucket_contract() -> None:
    assert support_bucket(0) == "UNSEEN_IN_TRAIN"
    assert support_bucket(1) == "SINGLETON_IN_TRAIN"
    assert support_bucket(2) == "LOW_SUPPORT_2_TO_3"
    assert support_bucket(3) == "LOW_SUPPORT_2_TO_3"
    assert support_bucket(4) == "MEDIUM_SUPPORT_4_TO_7"
    assert support_bucket(7) == "MEDIUM_SUPPORT_4_TO_7"
    assert support_bucket(8) == "HIGH_SUPPORT_8_PLUS"


def test_top1_change_taxonomy_contract() -> None:
    assert classify_top1_change("I", "E") == "I_TO_E_OR_S"
    assert classify_top1_change("C", "S") == "C_TO_E_OR_S"
    assert classify_top1_change("I", "C") == "I_TO_C"
    assert classify_top1_change("E", "I") == "E_OR_S_TO_I"
    assert classify_top1_change("S", "C") == "E_OR_S_TO_C"


def test_support_slice_uses_frozen_lexical_anchor() -> None:
    paired = pd.DataFrame(
        {
            "lexical_support_bucket": [
                "UNSEEN_IN_TRAIN",
                "UNSEEN_IN_TRAIN",
            ],
            "lexical_train_support": [0, 0],
            "lexical_esci_label": ["I", "E"],
            "semantic_esci_label": ["E", "E"],
        }
    )

    result = build_support_slice_table(paired)

    assert len(result) == 1
    assert result.iloc[0]["query_count"] == 2
    assert (
        result.iloc[0][
            "top1_high_relevance_delta"
        ] == 0.5
    )
    assert (
        result.iloc[0][
            "top1_irrelevant_delta"
        ] == -0.5
    )
