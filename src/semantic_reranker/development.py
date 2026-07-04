from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from semantic_reranker.contracts import GAIN_BY_LABEL
from semantic_reranker.metrics import dcg_at_k


HIGH_RELEVANCE_LABELS = {"E", "S"}
IRRELEVANT_LABEL = "I"
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require_exact_hash(
    path: Path,
    expected_sha256: str,
    label: str,
) -> None:
    actual_sha256 = sha256_file(path)

    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} hash mismatch. "
            f"Expected={expected_sha256} "
            f"Actual={actual_sha256}"
        )


def stable_query_partition(
    query_id: object,
    modulus: int = 10,
) -> int:
    if modulus < 2:
        raise ValueError("modulus must be at least 2.")

    digest = hashlib.sha256(
        str(query_id).encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    ) % modulus


def normalize_tokens(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(str(value))
        if token
    }


def lexical_title_coverage(
    query: object,
    title: object,
) -> float:
    query_tokens = normalize_tokens(query)

    if not query_tokens:
        return 0.0

    title_tokens = normalize_tokens(title)

    return len(
        query_tokens.intersection(title_tokens)
    ) / len(query_tokens)


def assign_query_disjoint_partition(
    frame: pd.DataFrame,
    validation_partitions: set[int],
    modulus: int = 10,
) -> pd.DataFrame:
    required = {"query_id"}

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing split keys: {sorted(missing)}"
        )

    assigned = frame.copy()

    assigned["query_hash_partition"] = assigned[
        "query_id"
    ].map(
        lambda value: stable_query_partition(
            value,
            modulus=modulus,
        )
    )

    assigned["development_partition"] = np.where(
        assigned["query_hash_partition"].isin(
            validation_partitions
        ),
        "validation",
        "fit",
    )

    per_query_partition_count = (
        assigned.groupby("query_id")[
            "development_partition"
        ]
        .nunique()
    )

    if not (per_query_partition_count == 1).all():
        raise RuntimeError(
            "A query was assigned to multiple development partitions."
        )

    return assigned


def rank_lexical_candidates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "query_id",
        "query",
        "product_id",
        "product_title",
        "esci_label",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Lexical baseline missing columns: {sorted(missing)}"
        )

    ranked = frame.copy()

    ranked["lexical_title_coverage"] = [
        lexical_title_coverage(query, title)
        for query, title in zip(
            ranked["query"],
            ranked["product_title"],
        )
    ]

    ranked["_product_sort"] = ranked[
        "product_id"
    ].astype(str)

    ranked = ranked.sort_values(
        [
            "query_id",
            "lexical_title_coverage",
            "_product_sort",
        ],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()

    ranked["lexical_rank"] = (
        ranked.groupby("query_id")
        .cumcount()
        .add(1)
    )

    return ranked.drop(columns=["_product_sort"])


def mean_ndcg_at_k(
    ranked: pd.DataFrame,
    k: int = 10,
) -> float:
    scores: list[float] = []

    for _, query_rows in ranked.groupby(
        "query_id",
        sort=False,
    ):
        ordered_gains = [
            GAIN_BY_LABEL[label]
            for label in query_rows["esci_label"]
        ]

        ideal_gains = sorted(
            ordered_gains,
            reverse=True,
        )

        observed_dcg = dcg_at_k(
            ordered_gains,
            k,
        )

        ideal_dcg = dcg_at_k(
            ideal_gains,
            k,
        )

        if ideal_dcg == 0.0:
            scores.append(0.0)
        else:
            scores.append(observed_dcg / ideal_dcg)

    if not scores:
        raise ValueError("No queries available for nDCG.")

    return float(np.mean(scores))


def partition_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for partition_name, partition_rows in frame.groupby(
        "development_partition",
        sort=True,
    ):
        label_counts = (
            partition_rows["esci_label"]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "development_partition": str(
                    partition_name
                ),
                "candidate_row_count": int(
                    len(partition_rows)
                ),
                "query_count": int(
                    partition_rows["query_id"].nunique()
                ),
                "exact_count": int(label_counts.get("E", 0)),
                "substitute_count": int(
                    label_counts.get("S", 0)
                ),
                "complement_count": int(
                    label_counts.get("C", 0)
                ),
                "irrelevant_count": int(
                    label_counts.get("I", 0)
                ),
            }
        )

    return pd.DataFrame(rows)


def audit_development_lexical_baseline(
    raw_root: Path,
    source_contract_path: Path,
    output_root: Path,
    modulus: int = 10,
    validation_partitions: set[int] | None = None,
) -> dict[str, Any]:
    if validation_partitions is None:
        validation_partitions = {8, 9}

    if not validation_partitions:
        raise ValueError(
            "validation_partitions must not be empty."
        )

    if any(
        partition < 0 or partition >= modulus
        for partition in validation_partitions
    ):
        raise ValueError(
            "validation_partitions must be within the modulus range."
        )

    source_contract = json.loads(
        source_contract_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        source_contract.get("status") !=
        "ESCI_US_TASK1_SOURCE_CONTRACT_PASS"
    ):
        raise ValueError(
            "Source contract does not have PASS status."
        )

    dataset_root = raw_root / "shopping_queries_dataset"

    examples_path = (
        dataset_root /
        "shopping_queries_dataset_examples.parquet"
    )

    products_path = (
        dataset_root /
        "shopping_queries_dataset_products.parquet"
    )

    require_exact_hash(
        examples_path,
        source_contract["raw_file_receipt"][
            "shopping_queries_dataset_examples.parquet"
        ]["sha256"],
        "Examples parquet",
    )

    require_exact_hash(
        products_path,
        source_contract["raw_file_receipt"][
            "shopping_queries_dataset_products.parquet"
        ]["sha256"],
        "Products parquet",
    )

    examples = pq.read_table(
        examples_path,
        columns=[
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "split",
        ],
        filters=[
            ("product_locale", "=", "us"),
            ("small_version", "=", 1),
            ("split", "=", "train"),
        ],
    ).to_pandas()

    if examples.empty:
        raise RuntimeError(
            "No US Task 1 training rows were loaded."
        )

    if set(examples["split"].unique()) != {"train"}:
        raise RuntimeError(
            "Non-training rows entered development audit."
        )

    products = pq.read_table(
        products_path,
        columns=[
            "product_id",
            "product_locale",
            "product_title",
            "product_brand",
            "product_color",
            "product_bullet_point",
        ],
        filters=[
            ("product_locale", "=", "us"),
        ],
    ).to_pandas()

    products = products.drop_duplicates()

    if products.duplicated(
        subset=["product_locale", "product_id"],
        keep=False,
    ).any():
        raise RuntimeError(
            "Duplicate locale-product keys in product catalog."
        )

    candidates = examples.merge(
        products,
        on=["product_locale", "product_id"],
        how="left",
        validate="m:1",
    )

    if len(candidates) != len(examples):
        raise RuntimeError(
            "Candidate join changed the source row count."
        )

    candidates["product_title"] = candidates[
        "product_title"
    ].fillna("")

    candidates = assign_query_disjoint_partition(
        candidates,
        validation_partitions=validation_partitions,
        modulus=modulus,
    )

    fit_rows = candidates.loc[
        candidates["development_partition"] == "fit"
    ].copy()

    validation_rows = candidates.loc[
        candidates["development_partition"] == "validation"
    ].copy()

    fit_query_ids = set(
        fit_rows["query_id"].unique()
    )

    validation_query_ids = set(
        validation_rows["query_id"].unique()
    )

    if fit_query_ids.intersection(validation_query_ids):
        raise RuntimeError(
            "Query leakage detected between fit and validation."
        )

    ranked_validation = rank_lexical_candidates(
        validation_rows
    )

    top1 = ranked_validation.loc[
        ranked_validation["lexical_rank"] == 1
    ].copy()

    if len(top1) != validation_rows[
        "query_id"
    ].nunique():
        raise RuntimeError(
            "Lexical comparator did not select one candidate "
            "for every validation query."
        )

    top1_high_relevance = top1["esci_label"].isin(
        HIGH_RELEVANCE_LABELS
    )

    top1_irrelevant = top1["esci_label"].eq(
        IRRELEVANT_LABEL
    )

    top1_label_counts = {
        str(label): int(count)
        for label, count in (
            top1["esci_label"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    partition_report = partition_summary(candidates)

    lexical_metrics = {
        "eligible_validation_query_count": int(
            top1["query_id"].nunique()
        ),
        "validation_candidate_row_count": int(
            len(validation_rows)
        ),
        "ndcg_at_10": mean_ndcg_at_k(
            ranked_validation,
            k=10,
        ),
        "top1_high_relevance_precision": float(
            top1_high_relevance.mean()
        ),
        "top1_irrelevant_rate": float(
            top1_irrelevant.mean()
        ),
        "top1_label_counts": top1_label_counts,
        "mean_top1_lexical_title_coverage": float(
            top1["lexical_title_coverage"].mean()
        ),
    }

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    partition_summary_path = (
        output_root /
        "development_partition_summary.csv"
    )

    top1_path = (
        output_root /
        "lexical_baseline_validation_top1.csv"
    )

    receipt_path = (
        output_root /
        "development_lexical_baseline_receipt.json"
    )

    partition_report.to_csv(
        partition_summary_path,
        index=False,
    )

    top1[
        [
            "query_id",
            "query",
            "product_id",
            "product_title",
            "esci_label",
            "lexical_title_coverage",
            "lexical_rank",
        ]
    ].to_csv(
        top1_path,
        index=False,
    )

    payload = {
        "status": (
            "QUERY_DISJOINT_DEVELOPMENT_AND_"
            "FROZEN_LEXICAL_BASELINE_COMPLETE"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_contract_path": str(
            source_contract_path
        ),
        "source_contract_sha256": sha256_file(
            source_contract_path
        ),
        "examples_parquet_sha256": sha256_file(
            examples_path
        ),
        "products_parquet_sha256": sha256_file(
            products_path
        ),
        "official_test_row_level_used": False,
        "development_split": {
            "source_split": "train",
            "group_key": "query_id",
            "hash_algorithm": "sha256",
            "hash_modulus": modulus,
            "validation_partitions": sorted(
                validation_partitions
            ),
            "fit_partitions": [
                partition
                for partition in range(modulus)
                if partition not in validation_partitions
            ],
            "query_group_leakage_detected": False,
        },
        "development_dataset_summary": {
            "total_candidate_row_count": int(
                len(candidates)
            ),
            "total_query_count": int(
                candidates["query_id"].nunique()
            ),
            "fit_candidate_row_count": int(
                len(fit_rows)
            ),
            "fit_query_count": int(
                fit_rows["query_id"].nunique()
            ),
            "validation_candidate_row_count": int(
                len(validation_rows)
            ),
            "validation_query_count": int(
                validation_rows["query_id"].nunique()
            ),
            "missing_product_title_count": int(
                candidates["product_title"].eq("").sum()
            ),
            "missing_product_brand_count": int(
                candidates["product_brand"].isna().sum()
            ),
            "missing_product_color_count": int(
                candidates["product_color"].isna().sum()
            ),
            "missing_product_bullet_point_count": int(
                candidates[
                    "product_bullet_point"
                ].isna().sum()
            ),
        },
        "lexical_comparator": {
            "name": "TITLE_TOKEN_COVERAGE_TOP_1",
            "runtime_features": [
                "query",
                "product_title",
            ],
            "score": (
                "unique query-token coverage in product title"
            ),
            "sort_order": [
                "lexical_title_coverage_descending",
                "product_id_ascending",
            ],
            "label_used_as_feature": False,
            "support_feature_used": False,
            "metrics": lexical_metrics,
        },
        "next_gate": (
            "BUILD_GPU_CROSS_ENCODER_DEVELOPMENT_DRIVER_"
            "AGAINST_FROZEN_LEXICAL_BASELINE"
        ),
    }

    receipt_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload["receipt_path"] = str(receipt_path)
    payload["receipt_sha256"] = sha256_file(
        receipt_path
    )
    payload["partition_summary_path"] = str(
        partition_summary_path
    )
    payload["top1_path"] = str(top1_path)

    return payload