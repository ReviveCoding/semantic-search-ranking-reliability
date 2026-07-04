from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.esci_source import (
    is_git_lfs_pointer,
    require_columns,
)


def test_git_lfs_pointer_is_detected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.parquet"

    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n",
        encoding="utf-8",
    )

    assert is_git_lfs_pointer(path)


def test_regular_parquet_prefix_is_not_pointer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "example.parquet"
    path.write_bytes(b"PAR1")

    assert not is_git_lfs_pointer(path)


def test_required_columns_rejects_missing_column() -> None:
    with pytest.raises(ValueError):
        require_columns(
            actual={"query_id"},
            required={"query_id", "source"},
            label="sources csv",
        )