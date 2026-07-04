from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.development import (
    assign_query_disjoint_partition,
    lexical_title_coverage,
    rank_lexical_candidates,
    stable_query_partition,
)


def test_query_partition_is_deterministic() -> None:
    first = stable_query_partition(
        query_id=12345,
        modulus=10,
    )

    second = stable_query_partition(
        query_id=12345,
        modulus=10,
    )

    assert first == second
    assert 0 <= first < 10


def test_query_disjoint_assignment_keeps_query_together() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "product_id": ["a", "b", "c"],
        }
    )

    assigned = assign_query_disjoint_partition(
        frame,
        validation_partitions={8, 9},
    )

    assert (
        assigned.groupby("query_id")[
            "development_partition"
        ]
        .nunique()
        .eq(1)
        .all()
    )


def test_lexical_title_coverage() -> None:
    score = lexical_title_coverage(
        "wireless mouse",
        "Ergonomic Wireless Mouse",
    )

    assert score == 1.0


def test_lexical_rank_uses_product_id_on_score_tie() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1, 1],
            "query": ["mouse", "mouse"],
            "product_id": ["z", "a"],
            "product_title": [
                "Wireless Mouse",
                "Wireless Mouse",
            ],
            "esci_label": ["E", "S"],
        }
    )

    ranked = rank_lexical_candidates(frame)

    assert ranked.iloc[0]["product_id"] == "a"
    assert ranked.iloc[0]["lexical_rank"] == 1