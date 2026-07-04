from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from semantic_reranker.contracts import (
    GAIN_VECTOR,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from semantic_reranker.development import (
    rank_lexical_candidates,
)
from semantic_reranker.semantic_development import (
    PairCollator,
    PairDataset,
    load_deberta_tokenizer,
    paired_query_bootstrap,
    per_query_ranking_metrics,
    rank_semantic_candidates,
    require_bf16_cuda,
    require_exact_hash,
    require_fp32_trainable_parameters,
    resolve_model_revision,
    score_cross_encoder,
    set_reproducible_seed,
    sha256_file,
    summary_metrics,
    to_pair_frame,
)


def require_single_official_test_receipt(
    output_parent: Path,
) -> None:
    if not output_parent.exists():
        return

    receipts = list(
        output_parent.rglob(
            "official_test_evaluation_receipt.json"
        )
    )

    if receipts:
        rendered = [
            str(path)
            for path in receipts
        ]

        raise RuntimeError(
            "Official-test evaluation already exists: "
            f"{rendered}"
        )


def load_contract_json(
    path: Path,
    expected_status: str,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("status") != expected_status:
        raise ValueError(
            f"Unexpected receipt status for {path.name}: "
            f"{payload.get('status')}"
        )

    return payload


def load_esci_rows(
    raw_root: Path,
    source_contract: dict[str, Any],
    split: str,
) -> pd.DataFrame:
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
            ("split", "=", split),
        ],
    ).to_pandas()

    if examples.empty:
        raise RuntimeError(
            f"No US Task 1 rows found for split={split}."
        )

    if set(examples["split"].unique()) != {split}:
        raise RuntimeError(
            f"Non-{split} rows entered the final evaluation."
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
            "Duplicate locale-product keys in catalog."
        )

    merged = examples.merge(
        products,
        on=["product_locale", "product_id"],
        how="left",
        validate="m:1",
    )

    if len(merged) != len(examples):
        raise RuntimeError(
            "Product merge changed candidate-row count."
        )

    return merged


def build_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        PairDataset(frame),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PairCollator(
            tokenizer,
            max_length=max_length,
        ),
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def train_final_model(
    model: Any,
    train_frame: pd.DataFrame,
    tokenizer: Any,
    device: torch.device,
    epochs: int,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    seed: int,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be at least one.")

    require_fp32_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epoch_records: list[dict[str, Any]] = []

    for epoch_index in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loader = build_loader(
            train_frame,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch_index,
        )

        losses: list[float] = []
        optimizer_steps = 0
        micro_batches = 0

        for batch in loader:
            row_ids = batch.pop("row_ids")
            del row_ids

            model_batch = {
                name: tensor.to(
                    device,
                    non_blocking=True,
                )
                for name, tensor in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                outputs = model(**model_batch)
                loss = (
                    outputs.loss /
                    gradient_accumulation_steps
                )

            loss.backward()

            losses.append(
                float(
                    loss.detach().float().item() *
                    gradient_accumulation_steps
                )
            )

            micro_batches += 1

            if (
                micro_batches %
                gradient_accumulation_steps == 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_gradient_norm,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        if (
            micro_batches %
            gradient_accumulation_steps != 0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_gradient_norm,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        epoch_records.append(
            {
                "epoch_index": epoch_index + 1,
                "processed_micro_batches": micro_batches,
                "optimizer_steps": optimizer_steps,
                "mean_training_loss": float(
                    np.mean(losses)
                ),
                "final_training_loss": float(
                    losses[-1]
                ),
            }
        )

    return {
        "epoch_count": epochs,
        "epoch_records": epoch_records,
        "total_micro_batches": int(
            sum(
                record["processed_micro_batches"]
                for record in epoch_records
            )
        ),
        "total_optimizer_steps": int(
            sum(
                record["optimizer_steps"]
                for record in epoch_records
            )
        ),
    }


def build_model_manifest(
    model_dir: Path,
) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(model_dir).as_posix(): {
            "byte_count": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in model_dir.rglob("*")
        if path.is_file()
    }


def run_one_shot_official_test(
    raw_root: Path,
    source_contract_path: Path,
    lexical_receipt_path: Path,
    development_metrics_path: Path,
    output_parent: Path,
    output_root: Path,
    backbone_revision: str,
    final_epochs: int,
) -> dict[str, Any]:
    require_single_official_test_receipt(output_parent)

    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable for final training."
        )

    source_contract = load_contract_json(
        source_contract_path,
        "ESCI_US_TASK1_SOURCE_CONTRACT_PASS",
    )

    lexical_receipt = load_contract_json(
        lexical_receipt_path,
        "QUERY_DISJOINT_DEVELOPMENT_AND_"
        "FROZEN_LEXICAL_BASELINE_COMPLETE",
    )

    development_metrics = load_contract_json(
        development_metrics_path,
        "GPU_CROSS_ENCODER_DEVELOPMENT_COMPLETE",
    )

    if development_metrics.get("gate_pass") is not True:
        raise RuntimeError(
            "Development gate did not pass. "
            "Official test evaluation is unauthorized."
        )

    if (
        development_metrics.get(
            "official_test_row_level_used"
        )
        is not False
    ):
        raise RuntimeError(
            "Development receipt indicates test usage."
        )

    if (
        development_metrics.get("backbone_revision") !=
        backbone_revision
    ):
        raise RuntimeError(
            "Final backbone revision differs from "
            "development-frozen revision."
        )

    if (
        lexical_receipt.get(
            "official_test_row_level_used"
        )
        is not False
    ):
        raise RuntimeError(
            "Frozen lexical baseline indicates test usage."
        )

    set_reproducible_seed(20260704)

    train_rows = load_esci_rows(
        raw_root,
        source_contract,
        split="train",
    )

    test_rows = load_esci_rows(
        raw_root,
        source_contract,
        split="test",
    )

    train_frame = to_pair_frame(train_rows)
    test_frame = to_pair_frame(test_rows)

    if set(train_frame["query_id"]).intersection(
        set(test_frame["query_id"])
    ):
        raise RuntimeError(
            "Official train and test query IDs overlap."
        )

    backbone = "microsoft/deberta-v3-base"

    tokenizer = load_deberta_tokenizer(
        backbone=backbone,
        revision=backbone_revision,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            backbone,
            revision=backbone_revision,
            num_labels=4,
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
        )
    )

    device = torch.device("cuda:0")
    require_bf16_cuda(device)

    model.to(
        device=device,
        dtype=torch.float32,
    )

    parameter_contract = (
        require_fp32_trainable_parameters(model)
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()

    train_summary = train_final_model(
        model=model,
        train_frame=train_frame,
        tokenizer=tokenizer,
        device=device,
        epochs=final_epochs,
        max_length=256,
        batch_size=8,
        learning_rate=2.0e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        max_gradient_norm=1.0,
        seed=20260704,
    )

    test_loader = build_loader(
        test_frame,
        tokenizer=tokenizer,
        max_length=256,
        batch_size=16,
        shuffle=False,
        seed=20260704,
    )

    semantic_scores = score_cross_encoder(
        model=model,
        loader=test_loader,
        device=device,
        amp_dtype=torch.bfloat16,
        max_batches=None,
    )

    if len(semantic_scores) != len(test_frame):
        raise RuntimeError(
            "Final semantic scoring did not cover every test row."
        )

    scored_test = test_frame.merge(
        semantic_scores,
        on="row_id",
        how="inner",
        validate="1:1",
    )

    semantic_ranked = rank_semantic_candidates(
        scored_test
    )

    lexical_ranked = rank_lexical_candidates(
        test_frame
    )

    semantic_query_metrics = per_query_ranking_metrics(
        semantic_ranked,
        rank_column="semantic_rank",
    )

    lexical_query_metrics = per_query_ranking_metrics(
        lexical_ranked,
        rank_column="lexical_rank",
    )

    semantic_metrics = summary_metrics(
        semantic_query_metrics
    )

    lexical_metrics = summary_metrics(
        lexical_query_metrics
    )

    semantic_metrics["macro_f1"] = float(
        f1_score(
            scored_test["label_id"],
            scored_test[
                "semantic_predicted_label_id"
            ],
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
    )

    bootstrap = paired_query_bootstrap(
        lexical_query_metrics,
        semantic_query_metrics,
        replicate_count=1000,
        seed=20260704,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    model_dir = output_root / "model"
    metrics_path = (
        output_root /
        "official_test_evaluation_receipt.json"
    )
    query_metrics_path = (
        output_root /
        "paired_test_query_ranking_metrics.csv"
    )

    semantic_top1_path = (
        output_root /
        "semantic_test_top1.csv"
    )

    lexical_top1_path = (
        output_root /
        "lexical_test_top1.csv"
    )

    semantic_top1 = semantic_ranked.loc[
        semantic_ranked["semantic_rank"] == 1
    ].copy()

    lexical_top1 = lexical_ranked.loc[
        lexical_ranked["lexical_rank"] == 1
    ].copy()

    semantic_top1[
        [
            "query_id",
            "query",
            "product_id",
            "product_title",
            "esci_label",
            "semantic_expected_gain",
            "semantic_predicted_label",
        ]
    ].to_csv(
        semantic_top1_path,
        index=False,
    )

    lexical_top1[
        [
            "query_id",
            "query",
            "product_id",
            "product_title",
            "esci_label",
            "lexical_title_coverage",
        ]
    ].to_csv(
        lexical_top1_path,
        index=False,
    )

    paired_test_metrics = lexical_query_metrics.merge(
        semantic_query_metrics,
        on="query_id",
        how="inner",
        suffixes=("_lexical", "_semantic"),
        validate="1:1",
    )

    paired_test_metrics.to_csv(
        query_metrics_path,
        index=False,
    )

    model.save_pretrained(
        model_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(model_dir)

    elapsed_seconds = time.perf_counter() - started

    payload = {
        "status": (
            "ONE_SHOT_OFFICIAL_TEST_EVALUATION_COMPLETE"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "official_test_row_level_used": True,
        "official_test_usage_role": (
            "final_evaluation_only"
        ),
        "model_selection_after_test_allowed": False,
        "backbone": backbone,
        "backbone_revision": backbone_revision,
        "final_training_contract": {
            "train_source_split": "train",
            "final_epochs": final_epochs,
            "loss": "standard_four_class_cross_entropy",
            "max_length": 256,
            "train_batch_size": 8,
            "evaluation_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "mixed_precision": {
                "autocast_dtype": "bfloat16",
                "grad_scaler_enabled": False,
                "trainable_parameter_dtype": "float32",
            },
            "seed": 20260704,
        },
        "model_input_contract": {
            "sequence_a": "query",
            "sequence_b": [
                "product_title",
                "product_brand",
                "product_color",
                "product_bullet_point",
            ],
            "forbidden_features": [
                "query_id",
                "product_id",
                "support_frequency",
                "esci_label",
                "test_statistics",
            ],
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
            "examples_parquet_sha256": sha256_file(
                raw_root /
                "shopping_queries_dataset" /
                "shopping_queries_dataset_examples.parquet"
            ),
            "products_parquet_sha256": sha256_file(
                raw_root /
                "shopping_queries_dataset" /
                "shopping_queries_dataset_products.parquet"
            ),
        },
        "dataset_summary": {
            "train_candidate_row_count": int(
                len(train_frame)
            ),
            "train_query_count": int(
                train_frame["query_id"].nunique()
            ),
            "test_candidate_row_count": int(
                len(test_frame)
            ),
            "test_query_count": int(
                test_frame["query_id"].nunique()
            ),
            "train_test_query_overlap_detected": False,
        },
        "parameter_contract": parameter_contract,
        "train_summary": train_summary,
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "bf16_supported": bool(
                torch.cuda.is_bf16_supported()
            ),
            "peak_allocated_gb": round(
                torch.cuda.max_memory_allocated(device) /
                1024**3,
                3,
            ),
            "peak_reserved_gb": round(
                torch.cuda.max_memory_reserved(device) /
                1024**3,
                3,
            ),
        },
        "elapsed_seconds": elapsed_seconds,
        "frozen_lexical_metrics": lexical_metrics,
        "semantic_metrics": semantic_metrics,
        "paired_query_bootstrap": bootstrap,
        "model_artifact_manifest": build_model_manifest(
            model_dir
        ),
        "next_gate": (
            "CLOSE_MODEL_SELECTION_AND_BUILD_"
            "READ_ONLY_RELIABILITY_REPORT"
        ),
    }

    metrics_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload["receipt_path"] = str(metrics_path)
    payload["receipt_sha256"] = sha256_file(
        metrics_path
    )
    payload["paired_test_metrics_path"] = str(
        query_metrics_path
    )

    return payload