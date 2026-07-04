from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.official_test import (
    require_single_official_test_receipt,
)


def test_one_shot_guard_allows_empty_parent(
    tmp_path: Path,
) -> None:
    output_parent = tmp_path / "official_test"

    require_single_official_test_receipt(output_parent)


def test_one_shot_guard_rejects_existing_receipt(
    tmp_path: Path,
) -> None:
    output_parent = tmp_path / "official_test"
    run_root = output_parent / "run_1"

    run_root.mkdir(parents=True)

    (
        run_root /
        "official_test_evaluation_receipt.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        require_single_official_test_receipt(output_parent)