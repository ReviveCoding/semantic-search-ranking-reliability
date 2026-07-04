from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.semantic_development import (
    load_deberta_tokenizer,
    paired_query_bootstrap,
    rank_semantic_candidates,
    registered_amp_policy,
    require_fp32_trainable_parameters,
    to_pair_frame,
)
from semantic_reranker.text import normalize_text


def test_normalize_text_removes_nan() -> None:
    assert normalize_text(float("nan")) == ""


def test_pair_frame_has_no_nan_text_tokens() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1],
            "product_id": ["p1"],
            "query": ["wireless mouse"],
            "product_title": ["Mouse"],
            "product_brand": [np.nan],
            "product_color": [np.nan],
            "product_bullet_point": [np.nan],
            "esci_label": ["E"],
        }
    )

    paired = to_pair_frame(frame)

    assert "nan" not in paired[
        "model_product_text"
    ].iloc[0].lower()


def test_semantic_rank_prefers_higher_gain() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [1, 1],
            "product_id": ["b", "a"],
            "semantic_expected_gain": [0.1, 0.9],
        }
    )

    ranked = rank_semantic_candidates(frame)

    assert ranked.iloc[0]["product_id"] == "a"
    assert ranked.iloc[0]["semantic_rank"] == 1


def test_bootstrap_has_expected_delta_direction() -> None:
    lexical = pd.DataFrame(
        {
            "query_id": [1, 2],
            "ndcg_at_10": [0.5, 0.5],
            "top1_high_relevance": [0, 0],
            "top1_irrelevant": [1, 1],
        }
    )

    semantic = pd.DataFrame(
        {
            "query_id": [1, 2],
            "ndcg_at_10": [1.0, 1.0],
            "top1_high_relevance": [1, 1],
            "top1_irrelevant": [0, 0],
        }
    )

    result = paired_query_bootstrap(
        lexical,
        semantic,
        replicate_count=20,
        seed=7,
    )

    assert result["ndcg_at_10_delta"] == 0.5
    assert (
        result[
            "top1_high_relevance_delta"
        ] == 1.0
    )
    assert result["top1_irrelevant_delta"] == -1.0

def test_deberta_tokenizer_uses_slow_sentencepiece_contract(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTokenizer:
        pass

    def fake_from_pretrained(
        backbone: str,
        **kwargs: object,
    ) -> FakeTokenizer:
        captured["backbone"] = backbone
        captured.update(kwargs)
        return FakeTokenizer()

    monkeypatch.setattr(
        "semantic_reranker.semantic_development."
        "AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    tokenizer = load_deberta_tokenizer(
        backbone="microsoft/deberta-v3-base",
        revision="test-revision",
    )

    assert isinstance(tokenizer, FakeTokenizer)
    assert captured["backbone"] == "microsoft/deberta-v3-base"
    assert captured["revision"] == "test-revision"
    assert captured["use_fast"] is False


def test_registered_amp_policy_uses_bf16_without_scaler() -> None:
    assert registered_amp_policy() == {
        "autocast_dtype": "bfloat16",
        "grad_scaler_enabled": False,
        "trainable_parameter_dtype": "float32",
    }


def test_trainable_parameter_contract_requires_fp32() -> None:
    layer = torch.nn.Linear(2, 2)

    summary = require_fp32_trainable_parameters(layer)

    assert summary["trainable_parameter_dtypes"] == [
        "torch.float32"
    ]

    layer.to(dtype=torch.float16)

    with pytest.raises(RuntimeError):
        require_fp32_trainable_parameters(layer)
