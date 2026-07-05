from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


HIGH_RELEVANCE = {"E", "S"}

SUPPORT_BUCKET_ORDER = [
    "UNSEEN_IN_TRAIN",
    "SINGLETON_IN_TRAIN",
    "LOW_SUPPORT_2_TO_3",
    "MEDIUM_SUPPORT_4_TO_7",
    "HIGH_SUPPORT_8_PLUS",
]

CHANGE_ORDER = [
    "I_TO_E_OR_S",
    "C_TO_E_OR_S",
    "I_TO_C",
    "E_OR_S_TO_E_OR_S",
    "C_TO_C",
    "I_TO_I",
    "E_OR_S_TO_C",
    "E_OR_S_TO_I",
    "C_TO_I",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(
    path: Path,
    expected_status: str,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("status") != expected_status:
        raise ValueError(
            f"Unexpected status for {path.name}: "
            f"{payload.get('status')}"
        )

    return payload


def support_bucket(
    support_count: int,
) -> str:
    if support_count <= 0:
        return "UNSEEN_IN_TRAIN"

    if support_count == 1:
        return "SINGLETON_IN_TRAIN"

    if support_count <= 3:
        return "LOW_SUPPORT_2_TO_3"

    if support_count <= 7:
        return "MEDIUM_SUPPORT_4_TO_7"

    return "HIGH_SUPPORT_8_PLUS"


def label_group(
    label: str,
) -> str:
    if label in HIGH_RELEVANCE:
        return "E_OR_S"

    if label == "C":
        return "C"

    if label == "I":
        return "I"

    raise ValueError(f"Unexpected ESCI label: {label}")


def classify_top1_change(
    lexical_label: str,
    semantic_label: str,
) -> str:
    lexical_group = label_group(lexical_label)
    semantic_group = label_group(semantic_label)

    return f"{lexical_group}_TO_{semantic_group}"


def top1_metrics(
    labels: pd.Series,
) -> dict[str, float]:
    if labels.empty:
        raise ValueError("Cannot calculate metrics from no labels.")

    return {
        "top1_high_relevance_precision": float(
            labels.isin(HIGH_RELEVANCE).mean()
        ),
        "top1_irrelevant_rate": float(
            labels.eq("I").mean()
        ),
    }


def format_percentage(
    value: float,
) -> str:
    return f"{value * 100:.2f}%"


def markdown_table(
    frame: pd.DataFrame,
) -> str:
    if frame.empty:
        return "_No rows._"

    return frame.to_markdown(
        index=False,
        floatfmt=".6f",
    )


def build_train_support(
    raw_root: Path,
) -> pd.Series:
    examples_path = (
        raw_root /
        "shopping_queries_dataset" /
        "shopping_queries_dataset_examples.parquet"
    )

    if not examples_path.is_file():
        raise FileNotFoundError(
            f"ESCI examples parquet is missing: {examples_path}"
        )

    train_rows = pq.read_table(
        examples_path,
        columns=[
            "product_id",
            "product_locale",
            "small_version",
            "split",
        ],
        filters=[
            ("product_locale", "=", "us"),
            ("small_version", "=", 1),
            ("split", "=", "train"),
        ],
    ).to_pandas()

    if train_rows.empty:
        raise RuntimeError(
            "No US Task 1 training rows were loaded."
        )

    if set(train_rows["split"].unique()) != {"train"}:
        raise RuntimeError(
            "Non-training rows entered support diagnostics."
        )

    return (
        train_rows["product_id"]
        .astype(str)
        .value_counts()
    )


def build_support_slice_table(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for bucket in SUPPORT_BUCKET_ORDER:
        rows = paired.loc[
            paired["lexical_support_bucket"] == bucket
        ].copy()

        if rows.empty:
            continue

        lexical = top1_metrics(
            rows["lexical_esci_label"]
        )
        semantic = top1_metrics(
            rows["semantic_esci_label"]
        )

        records.append(
            {
                "frozen_lexical_support_bucket": bucket,
                "query_count": int(len(rows)),
                "mean_lexical_train_support": float(
                    rows["lexical_train_support"].mean()
                ),
                "lexical_top1_high_relevance_precision": (
                    lexical[
                        "top1_high_relevance_precision"
                    ]
                ),
                "semantic_top1_high_relevance_precision": (
                    semantic[
                        "top1_high_relevance_precision"
                    ]
                ),
                "top1_high_relevance_delta": (
                    semantic[
                        "top1_high_relevance_precision"
                    ] -
                    lexical[
                        "top1_high_relevance_precision"
                    ]
                ),
                "lexical_top1_irrelevant_rate": (
                    lexical["top1_irrelevant_rate"]
                ),
                "semantic_top1_irrelevant_rate": (
                    semantic["top1_irrelevant_rate"]
                ),
                "top1_irrelevant_delta": (
                    semantic["top1_irrelevant_rate"] -
                    lexical["top1_irrelevant_rate"]
                ),
            }
        )

    return pd.DataFrame(records)


def build_consistency_table(
    development: dict[str, Any],
    official: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for metric_name in (
        "ndcg_at_10",
        "top1_high_relevance_precision",
        "top1_irrelevant_rate",
    ):
        development_lexical = float(
            development["frozen_lexical_metrics"][
                metric_name
            ]
        )
        development_semantic = float(
            development["semantic_metrics"][
                metric_name
            ]
        )
        official_lexical = float(
            official["frozen_lexical_metrics"][
                metric_name
            ]
        )
        official_semantic = float(
            official["semantic_metrics"][
                metric_name
            ]
        )

        records.append(
            {
                "metric": metric_name,
                "development_lexical": development_lexical,
                "development_semantic": development_semantic,
                "development_delta": (
                    development_semantic -
                    development_lexical
                ),
                "official_test_lexical": official_lexical,
                "official_test_semantic": official_semantic,
                "official_test_delta": (
                    official_semantic -
                    official_lexical
                ),
                "official_minus_development_delta": (
                    (official_semantic - official_lexical) -
                    (development_semantic - development_lexical)
                ),
            }
        )

    return pd.DataFrame(records)


def build_taxonomy(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    taxonomy = (
        paired.groupby(
            "top1_change_type",
            dropna=False,
        )
        .size()
        .rename("query_count")
        .reset_index()
    )

    taxonomy["share_of_queries"] = (
        taxonomy["query_count"] / len(paired)
    )

    order_map = {
        name: index
        for index, name in enumerate(CHANGE_ORDER)
    }

    taxonomy["_order"] = taxonomy[
        "top1_change_type"
    ].map(order_map)

    taxonomy = taxonomy.sort_values(
        ["_order", "top1_change_type"],
        kind="mergesort",
    ).drop(columns="_order")

    return taxonomy


def build_examples(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    changed = paired.loc[
        paired["top1_change_type"].ne(
            "E_OR_S_TO_E_OR_S"
        ) &
        paired["top1_change_type"].ne("C_TO_C") &
        paired["top1_change_type"].ne("I_TO_I")
    ].copy()

    if changed.empty:
        return pd.DataFrame()

    order_map = {
        name: index
        for index, name in enumerate(CHANGE_ORDER)
    }

    changed["_change_order"] = changed[
        "top1_change_type"
    ].map(order_map)

    changed = changed.sort_values(
        [
            "_change_order",
            "query_id",
            "lexical_product_id",
            "semantic_product_id",
        ],
        kind="mergesort",
    )

    examples = (
        changed.groupby(
            "top1_change_type",
            sort=False,
        )
        .head(5)
        .copy()
    )

    return examples[
        [
            "top1_change_type",
            "lexical_support_bucket",
            "lexical_train_support",
            "query_id",
            "query",
            "lexical_esci_label",
            "lexical_product_title",
            "semantic_esci_label",
            "semantic_predicted_label",
            "semantic_product_title",
            "semantic_expected_gain",
        ]
    ]


def artifact_manifest(
    output_root: Path,
) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}

    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue

        if path.name == "report_artifact_manifest.json":
            continue

        manifest[path.relative_to(output_root).as_posix()] = {
            "byte_count": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }

    return manifest


def build_read_only_reliability_report(
    raw_root: Path,
    source_contract_path: Path,
    lexical_receipt_path: Path,
    development_metrics_path: Path,
    official_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}"
        )

    source = load_json(
        source_contract_path,
        "ESCI_US_TASK1_SOURCE_CONTRACT_PASS",
    )
    lexical = load_json(
        lexical_receipt_path,
        "QUERY_DISJOINT_DEVELOPMENT_AND_"
        "FROZEN_LEXICAL_BASELINE_COMPLETE",
    )
    development = load_json(
        development_metrics_path,
        "GPU_CROSS_ENCODER_DEVELOPMENT_COMPLETE",
    )
    official = load_json(
        official_receipt_path,
        "ONE_SHOT_OFFICIAL_TEST_EVALUATION_COMPLETE",
    )

    if development.get("gate_pass") is not True:
        raise RuntimeError(
            "Development gate was not passed."
        )

    if (
        official.get("model_selection_after_test_allowed")
        is not False
    ):
        raise RuntimeError(
            "Official receipt does not close model selection."
        )

    if (
        official.get("official_test_usage_role") !=
        "final_evaluation_only"
    ):
        raise RuntimeError(
            "Official receipt has an unexpected test usage role."
        )

    if (
        official.get("official_test_row_level_used")
        is not True
    ):
        raise RuntimeError(
            "Official receipt does not record test usage."
        )

    official_root = official_receipt_path.parent

    lexical_top1_path = (
        official_root /
        "lexical_test_top1.csv"
    )
    semantic_top1_path = (
        official_root /
        "semantic_test_top1.csv"
    )
    paired_metrics_path = (
        official_root /
        "paired_test_query_ranking_metrics.csv"
    )

    for path in (
        lexical_top1_path,
        semantic_top1_path,
        paired_metrics_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required official-test artifact is missing: {path}"
            )

    lexical_top1 = pd.read_csv(
        lexical_top1_path,
        dtype={"product_id": str},
    )
    semantic_top1 = pd.read_csv(
        semantic_top1_path,
        dtype={"product_id": str},
    )

    if lexical_top1["query_id"].duplicated().any():
        raise RuntimeError(
            "Lexical Top-1 artifact has duplicate queries."
        )

    if semantic_top1["query_id"].duplicated().any():
        raise RuntimeError(
            "Semantic Top-1 artifact has duplicate queries."
        )

    lexical_top1 = lexical_top1.rename(
        columns={
            "product_id": "lexical_product_id",
            "product_title": "lexical_product_title",
            "esci_label": "lexical_esci_label",
            "lexical_title_coverage": (
                "lexical_title_coverage"
            ),
        }
    )

    semantic_top1 = semantic_top1.rename(
        columns={
            "product_id": "semantic_product_id",
            "product_title": "semantic_product_title",
            "esci_label": "semantic_esci_label",
        }
    )

    paired = lexical_top1.merge(
        semantic_top1[
            [
                "query_id",
                "semantic_product_id",
                "semantic_product_title",
                "semantic_esci_label",
                "semantic_expected_gain",
                "semantic_predicted_label",
            ]
        ],
        on="query_id",
        how="inner",
        validate="1:1",
    )

    expected_query_count = int(
        official["dataset_summary"]["test_query_count"]
    )

    if len(paired) != expected_query_count:
        raise RuntimeError(
            "Top-1 artifact query coverage does not match "
            "official-test receipt."
        )

    train_support = build_train_support(raw_root)

    paired["lexical_train_support"] = (
        paired["lexical_product_id"]
        .astype(str)
        .map(train_support)
        .fillna(0)
        .astype(int)
    )

    paired["lexical_support_bucket"] = (
        paired["lexical_train_support"]
        .map(support_bucket)
    )

    paired["top1_change_type"] = [
        classify_top1_change(
            lexical_label,
            semantic_label,
        )
        for lexical_label, semantic_label in zip(
            paired["lexical_esci_label"],
            paired["semantic_esci_label"],
        )
    ]

    consistency = build_consistency_table(
        development,
        official,
    )

    support_slices = build_support_slice_table(
        paired
    )

    taxonomy = build_taxonomy(paired)
    examples = build_examples(paired)

    bootstrap = official["paired_query_bootstrap"]
    semantic_metrics = official["semantic_metrics"]
    lexical_metrics = official["frozen_lexical_metrics"]

    runtime_hours = float(
        official["elapsed_seconds"]
    ) / 3600.0

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    consistency_path = (
        output_root /
        "development_vs_official_test_consistency.csv"
    )
    support_path = (
        output_root /
        "frozen_lexical_anchor_support_slices.csv"
    )
    taxonomy_path = (
        output_root /
        "top1_change_taxonomy.csv"
    )
    examples_path = (
        output_root /
        "top1_change_examples.csv"
    )
    report_path = (
        output_root /
        "final_reliability_report.md"
    )
    summary_path = (
        output_root /
        "read_only_reliability_summary.json"
    )
    manifest_path = (
        output_root /
        "report_artifact_manifest.json"
    )

    consistency.to_csv(
        consistency_path,
        index=False,
    )
    support_slices.to_csv(
        support_path,
        index=False,
    )
    taxonomy.to_csv(
        taxonomy_path,
        index=False,
    )
    examples.to_csv(
        examples_path,
        index=False,
    )

    report = f"""# Reliability-First Semantic Search Ranking

## Final status

**MODEL SELECTION CLOSED**

The one-shot official ESCI test evaluation is complete. This report is
read-only and does not modify the model, weights, score map, input
contract, training configuration, or official-test receipt.

## Final official held-out test result

| Metric | Frozen lexical baseline | GPU semantic reranker | Delta |
|---|---:|---:|---:|
| nDCG@10 | {lexical_metrics["ndcg_at_10"]:.6f} | {semantic_metrics["ndcg_at_10"]:.6f} | {bootstrap["ndcg_at_10_delta"]:+.6f} |
| Top-1 Exact-or-Substitute precision | {format_percentage(lexical_metrics["top1_high_relevance_precision"])} | {format_percentage(semantic_metrics["top1_high_relevance_precision"])} | {format_percentage(bootstrap["top1_high_relevance_delta"])} |
| Top-1 Irrelevant rate | {format_percentage(lexical_metrics["top1_irrelevant_rate"])} | {format_percentage(semantic_metrics["top1_irrelevant_rate"])} | {format_percentage(bootstrap["top1_irrelevant_delta"])} |
| Four-class macro F1 | — | {semantic_metrics["macro_f1"]:.6f} | — |

## Paired query bootstrap evidence

| Metric | Point delta | 95% confidence interval |
|---|---:|---:|
| nDCG@10 | {bootstrap["ndcg_at_10_delta"]:+.6f} | [{bootstrap["ndcg_at_10_delta_ci_95"][0]:+.6f}, {bootstrap["ndcg_at_10_delta_ci_95"][1]:+.6f}] |
| Top-1 Exact-or-Substitute precision | {format_percentage(bootstrap["top1_high_relevance_delta"])} | [{format_percentage(bootstrap["top1_high_relevance_delta_ci_95"][0])}, {format_percentage(bootstrap["top1_high_relevance_delta_ci_95"][1])}] |
| Top-1 Irrelevant rate | {format_percentage(bootstrap["top1_irrelevant_delta"])} | [{format_percentage(bootstrap["top1_irrelevant_delta_ci_95"][0])}, {format_percentage(bootstrap["top1_irrelevant_delta_ci_95"][1])}] |

## Development-to-test consistency

{markdown_table(consistency)}

## Frozen-lexical-anchor support slices

The slice is anchored to the frozen lexical comparator's selected Top-1
product and its training-only product appearance count. Support is not a
model feature and this table is post-hoc diagnostic evidence only.

{markdown_table(support_slices)}

## Top-1 change taxonomy

{markdown_table(taxonomy)}

Representative examples are stored separately in
`top1_change_examples.csv`.

## Runtime and hardware

- GPU: {official["gpu"]["name"]}
- BF16 supported: {official["gpu"]["bf16_supported"]}
- Peak allocated GPU memory: {official["gpu"]["peak_allocated_gb"]:.3f} GB
- Peak reserved GPU memory: {official["gpu"]["peak_reserved_gb"]:.3f} GB
- Final elapsed runtime: {official["elapsed_seconds"]:.2f} seconds ({runtime_hours:.2f} hours)
- Train rows: {official["dataset_summary"]["train_candidate_row_count"]}
- Test rows: {official["dataset_summary"]["test_candidate_row_count"]}
- Train queries: {official["dataset_summary"]["train_query_count"]}
- Test queries: {official["dataset_summary"]["test_query_count"]}
- Train/test query overlap detected: {official["dataset_summary"]["train_test_query_overlap_detected"]}

## Reproducibility lineage

- ESCI source commit: {source["source"]["commit"]}
- Backbone: {official["backbone"]}
- Backbone revision: {official["backbone_revision"]}
- Official-test receipt SHA-256: {sha256_file(official_receipt_path)}
- Development receipt SHA-256: {sha256_file(development_metrics_path)}
- Frozen lexical receipt SHA-256: {sha256_file(lexical_receipt_path)}
- Source-contract receipt SHA-256: {sha256_file(source_contract_path)}
- Final model `model.safetensors` SHA-256:
  {official["model_artifact_manifest"]["model.safetensors"]["sha256"]}

## Resume claim

Built a GPU-accelerated four-class semantic query-product reranker on the
official Amazon ESCI benchmark using a pinned DeBERTa-v3 cross-encoder and
gain-aligned ranking score. In a one-shot held-out official test evaluation,
improved nDCG@10 from {lexical_metrics["ndcg_at_10"]:.3f} to
{semantic_metrics["ndcg_at_10"]:.3f}, increased Top-1 Exact-or-Substitute
precision by {format_percentage(bootstrap["top1_high_relevance_delta"])},
and reduced irrelevant Top-1 selections by
{format_percentage(abs(bootstrap["top1_irrelevant_delta"]))}.

## Claim boundary

This is an offline candidate-set semantic reranking result on static ESCI
relevance labels. It is not a production deployment result and does not
measure CTR, conversion, user behavior, advertising causality, or
end-to-end retrieval quality.
"""

    report_path.write_text(
        report.rstrip() + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": (
            "READ_ONLY_RELIABILITY_REPORT_COMPLETE"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "read_only_contract": {
            "model_selection_closed": True,
            "official_test_re_evaluated": False,
            "model_loaded": False,
            "model_trained": False,
            "test_used_for_model_selection": False,
        },
        "official_test_receipt": {
            "path": str(official_receipt_path),
            "sha256": sha256_file(official_receipt_path),
            "status": official["status"],
            "usage_role": official[
                "official_test_usage_role"
            ],
        },
        "final_metrics": {
            "frozen_lexical": lexical_metrics,
            "semantic": semantic_metrics,
            "paired_query_bootstrap": bootstrap,
        },
        "runtime": {
            "elapsed_seconds": float(
                official["elapsed_seconds"]
            ),
            "elapsed_hours": runtime_hours,
            "gpu": official["gpu"],
            "train_summary": official["train_summary"],
        },
        "provenance": {
            "source_contract_path": str(
                source_contract_path
            ),
            "source_contract_sha256": sha256_file(
                source_contract_path
            ),
            "lexical_receipt_path": str(
                lexical_receipt_path
            ),
            "lexical_receipt_sha256": sha256_file(
                lexical_receipt_path
            ),
            "development_metrics_path": str(
                development_metrics_path
            ),
            "development_metrics_sha256": sha256_file(
                development_metrics_path
            ),
            "backbone": official["backbone"],
            "backbone_revision": official[
                "backbone_revision"
            ],
            "source_commit": source["source"]["commit"],
        },
        "report_paths": {
            "report_markdown": str(report_path),
            "consistency_csv": str(consistency_path),
            "support_slices_csv": str(support_path),
            "taxonomy_csv": str(taxonomy_path),
            "examples_csv": str(examples_path),
        },
        "next_gate": (
            "PROJECT_CLOSED_FOR_MODEL_SELECTION_"
            "AND_READY_FOR_PORTFOLIO_DOCUMENTATION"
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    manifest = artifact_manifest(output_root)

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(
        summary_path
    )
    summary["manifest_path"] = str(manifest_path)
    summary["manifest_sha256"] = sha256_file(
        manifest_path
    )

    return summary
