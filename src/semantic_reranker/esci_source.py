from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


REQUIRED_EXAMPLE_COLUMNS = {
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "split",
}

REQUIRED_PRODUCT_COLUMNS = {
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
}

REQUIRED_SOURCE_COLUMNS = {
    "query_id",
    "source",
}

REQUIRED_FILE_NAMES = (
    "shopping_queries_dataset_examples.parquet",
    "shopping_queries_dataset_products.parquet",
    "shopping_queries_dataset_sources.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def is_git_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(128)

    return prefix.startswith(
        b"version https://git-lfs.github.com/spec/v1"
    )


def require_columns(
    actual: set[str],
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required.difference(actual))

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def audit_esci_us_task1(
    raw_root: Path,
    output_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    dataset_root = raw_root / "shopping_queries_dataset"

    examples_path = (
        dataset_root /
        "shopping_queries_dataset_examples.parquet"
    )
    products_path = (
        dataset_root /
        "shopping_queries_dataset_products.parquet"
    )
    sources_path = (
        dataset_root /
        "shopping_queries_dataset_sources.csv"
    )

    required_paths = [
        examples_path,
        products_path,
        sources_path,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required ESCI file is missing: {path}"
            )

        if path.suffix == ".parquet" and is_git_lfs_pointer(path):
            raise RuntimeError(
                f"Git LFS content was not downloaded: {path}"
            )

    example_columns = set(
        pq.ParquetFile(examples_path).schema_arrow.names
    )
    product_columns = set(
        pq.ParquetFile(products_path).schema_arrow.names
    )
    source_columns = set(
        pd.read_csv(sources_path, nrows=0).columns
    )

    require_columns(
        example_columns,
        REQUIRED_EXAMPLE_COLUMNS,
        "examples parquet",
    )
    require_columns(
        product_columns,
        REQUIRED_PRODUCT_COLUMNS,
        "products parquet",
    )
    require_columns(
        source_columns,
        REQUIRED_SOURCE_COLUMNS,
        "sources csv",
    )

    examples = pd.read_parquet(
        examples_path,
        columns=[
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
    )

    us_task1 = examples.loc[
        (examples["product_locale"] == "us") &
        (examples["small_version"] == 1)
    ].copy()

    if us_task1.empty:
        raise RuntimeError(
            "No US Task 1 rows were found."
        )

    split_counts = {
        str(name): int(count)
        for name, count in (
            us_task1["split"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    if not {"train", "test"}.issubset(split_counts):
        raise RuntimeError(
            "US Task 1 must contain train and test rows."
        )

    label_counts = {
        str(name): int(count)
        for name, count in (
            us_task1["esci_label"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    observed_labels = set(
        us_task1["esci_label"].dropna().unique()
    )

    allowed_labels = {"E", "S", "C", "I"}

    if not observed_labels.issubset(allowed_labels):
        raise RuntimeError(
            "Unexpected ESCI labels: "
            f"{sorted(observed_labels.difference(allowed_labels))}"
        )

    file_receipt = {
        path.name: {
            "path": str(path),
            "byte_count": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in required_paths
    }

    payload = {
        "status": "ESCI_US_TASK1_SOURCE_CONTRACT_PASS",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "repository": (
                "https://github.com/"
                "amazon-science/esci-data.git"
            ),
            "commit": source_commit,
            "raw_root": str(raw_root),
            "subset": {
                "product_locale": "us",
                "small_version": 1,
            },
        },
        "raw_file_receipt": file_receipt,
        "schema": {
            "example_columns": sorted(example_columns),
            "product_columns": sorted(product_columns),
            "source_columns": sorted(source_columns),
        },
        "task1_us_summary": {
            "row_count": int(len(us_task1)),
            "unique_query_count": int(
                us_task1["query_id"].nunique()
            ),
            "unique_product_count": int(
                us_task1["product_id"].nunique()
            ),
            "split_counts": split_counts,
            "label_counts": label_counts,
        },
        "claim_boundary": {
            "raw_source_is_official_esci": True,
            "production_click_logs_available": False,
            "production_impression_logs_available": False,
            "official_test_row_level_used_for_model_selection": False,
        },
        "next_gate": (
            "BUILD_QUERY_DISJOINT_DEVELOPMENT_DATASET_"
            "AND_FROZEN_LEXICAL_COMPARATOR"
        ),
    }

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    receipt_path = (
        output_root /
        "esci_us_task1_source_contract.json"
    )

    receipt_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload["receipt_path"] = str(receipt_path)
    payload["receipt_sha256"] = sha256_file(receipt_path)

    return payload