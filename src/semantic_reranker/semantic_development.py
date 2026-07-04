from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from semantic_reranker.contracts import (
    GAIN_VECTOR,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from semantic_reranker.development import (
    assign_query_disjoint_partition,
    rank_lexical_candidates,
)
from semantic_reranker.metrics import dcg_at_k
from semantic_reranker.text import build_query_product_pair


HIGH_RELEVANCE_LABELS = {"E", "S"}
IRRELEVANT_LABEL = "I"
MODEL_FIELDS = (
    "query",
    "product_title",
    "product_brand",
    "product_color",
    "product_bullet_point",
)


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


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def registered_amp_policy() -> dict[str, object]:
    """Return the fixed mixed-precision contract."""

    return {
        "autocast_dtype": "bfloat16",
        "grad_scaler_enabled": False,
        "trainable_parameter_dtype": "float32",
    }


def require_bf16_cuda(
    device: torch.device,
) -> torch.dtype:
    if device.type != "cuda":
        raise RuntimeError(
            "Semantic development requires CUDA."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable for semantic development."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Current CUDA device does not support BF16."
        )

    return torch.bfloat16


def require_fp32_trainable_parameters(
    model: Any,
) -> dict[str, object]:
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError(
            "Semantic model has no trainable parameters."
        )

    dtypes = {
        parameter.dtype
        for parameter in trainable
    }

    if dtypes != {torch.float32}:
        observed = sorted(
            str(dtype)
            for dtype in dtypes
        )

        raise RuntimeError(
            "Trainable parameters must be FP32 before "
            f"BF16 autocast. Observed={observed}"
        )

    return {
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in trainable
            )
        ),
        "trainable_parameter_dtypes": [
            "torch.float32"
        ],
    }

def resolve_model_revision(
    backbone: str,
    requested_revision: str | None,
) -> str:
    if requested_revision:
        return requested_revision

    revision = HfApi().model_info(backbone).sha

    if not revision:
        raise RuntimeError(
            "Hub model revision could not be resolved."
        )

    return revision


def load_deberta_tokenizer(
    backbone: str,
    revision: str,
) -> Any:
    """Load DeBERTa's SentencePiece tokenizer deterministically.

    DeBERTa-v3 ships a SentencePiece model. The slow tokenizer avoids
    runtime conversion to a fast tokenizer and therefore does not depend
    on optional conversion backends such as tiktoken.
    """

    return AutoTokenizer.from_pretrained(
        backbone,
        revision=revision,
        use_fast=False,
    )
def to_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(MODEL_FIELDS).union(
        {
            "query_id",
            "product_id",
            "esci_label",
        }
    )

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Missing semantic-model columns: {sorted(missing)}"
        )

    paired = frame.copy().reset_index(drop=True)

    for column in MODEL_FIELDS:
        paired[column] = paired[column].fillna("")

    pairs = [
        build_query_product_pair(record)
        for record in paired[
            list(MODEL_FIELDS)
        ].to_dict(orient="records")
    ]

    paired["model_query_text"] = [
        pair[0]
        for pair in pairs
    ]

    paired["model_product_text"] = [
        pair[1]
        for pair in pairs
    ]

    paired["label_id"] = paired["esci_label"].map(
        LABEL_TO_ID
    )

    if paired["label_id"].isna().any():
        unexpected = sorted(
            paired.loc[
                paired["label_id"].isna(),
                "esci_label",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unexpected ESCI labels: {unexpected}"
        )

    paired["label_id"] = paired["label_id"].astype("int64")
    paired["row_id"] = np.arange(len(paired))

    return paired


class PairDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self.queries = frame["model_query_text"].tolist()
        self.products = frame["model_product_text"].tolist()
        self.labels = frame["label_id"].tolist()
        self.row_ids = frame["row_id"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        return {
            "query": self.queries[index],
            "product": self.products[index],
            "label": self.labels[index],
            "row_id": self.row_ids[index],
        }


class PairCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        batch: list[dict[str, object]],
    ) -> dict[str, torch.Tensor]:
        queries = [
            str(item["query"])
            for item in batch
        ]

        products = [
            str(item["product"])
            for item in batch
        ]

        tokenized = self.tokenizer(
            queries,
            products,
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

        tokenized["labels"] = torch.tensor(
            [
                int(item["label"])
                for item in batch
            ],
            dtype=torch.long,
        )

        tokenized["row_ids"] = torch.tensor(
            [
                int(item["row_id"])
                for item in batch
            ],
            dtype=torch.long,
        )

        return tokenized


def load_development_frames(
    raw_root: Path,
    source_contract_path: Path,
    lexical_receipt_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_contract = json.loads(
        source_contract_path.read_text(
            encoding="utf-8"
        )
    )

    lexical_receipt = json.loads(
        lexical_receipt_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        source_contract.get("status") !=
        "ESCI_US_TASK1_SOURCE_CONTRACT_PASS"
    ):
        raise ValueError("Source contract is not PASS.")

    if (
        lexical_receipt.get("status") !=
        "QUERY_DISJOINT_DEVELOPMENT_AND_"
        "FROZEN_LEXICAL_BASELINE_COMPLETE"
    ):
        raise ValueError(
            "Frozen lexical receipt has an unexpected status."
        )

    if (
        lexical_receipt.get("official_test_row_level_used")
        is not False
    ):
        raise ValueError(
            "Frozen lexical receipt indicates test-row use."
        )

    split_contract = lexical_receipt[
        "development_split"
    ]

    if (
        split_contract["hash_modulus"] != 10 or
        split_contract["validation_partitions"] != [8, 9]
    ):
        raise ValueError(
            "Lexical receipt does not match the registered "
            "query-disjoint split."
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
            "Non-training rows entered semantic development."
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

    candidates = examples.merge(
        products,
        on=["product_locale", "product_id"],
        how="left",
        validate="m:1",
    )

    if len(candidates) != len(examples):
        raise RuntimeError(
            "Product join changed candidate row count."
        )

    candidates = assign_query_disjoint_partition(
        candidates,
        validation_partitions={8, 9},
        modulus=10,
    )

    fit = candidates.loc[
        candidates["development_partition"] == "fit"
    ].copy()

    validation = candidates.loc[
        candidates["development_partition"] == "validation"
    ].copy()

    fit_queries = set(fit["query_id"].unique())
    validation_queries = set(
        validation["query_id"].unique()
    )

    if fit_queries.intersection(validation_queries):
        raise RuntimeError(
            "Query leakage between fit and validation."
        )

    provenance = {
        "source_contract_sha256": sha256_file(
            source_contract_path
        ),
        "lexical_receipt_sha256": sha256_file(
            lexical_receipt_path
        ),
        "examples_parquet_sha256": sha256_file(
            examples_path
        ),
        "products_parquet_sha256": sha256_file(
            products_path
        ),
        "fit_row_count": int(len(fit)),
        "fit_query_count": int(
            fit["query_id"].nunique()
        ),
        "validation_row_count": int(
            len(validation)
        ),
        "validation_query_count": int(
            validation["query_id"].nunique()
        ),
    }

    return fit, validation, provenance


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    row_ids = batch.pop("row_ids")

    model_batch = {
        name: tensor.to(
            device,
            non_blocking=True,
        )
        for name, tensor in batch.items()
    }

    return model_batch, row_ids


def build_loader(
    frame: pd.DataFrame,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = PairDataset(frame)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
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


def train_cross_encoder(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    amp_dtype: torch.dtype,
    max_batches: int | None,
) -> dict[str, Any]:
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be at least 1."
        )

    require_fp32_trainable_parameters(model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    optimizer.zero_grad(set_to_none=True)

    losses: list[float] = []
    optimizer_steps = 0
    processed_batches = 0

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        if (
            max_batches is not None and
            batch_index > max_batches
        ):
            break

        model_batch, _ = move_batch_to_device(
            batch,
            device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
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

        processed_batches += 1

        should_step = (
            processed_batches %
            gradient_accumulation_steps == 0
        )

        reached_limit = (
            max_batches is not None and
            processed_batches == max_batches
        )

        if should_step or reached_limit:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_gradient_norm,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    if (
        processed_batches > 0 and
        processed_batches % gradient_accumulation_steps != 0 and
        max_batches is None
    ):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_gradient_norm,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    if processed_batches == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    return {
        "processed_micro_batches": processed_batches,
        "optimizer_steps": optimizer_steps,
        "mean_training_loss": float(np.mean(losses)),
        "final_training_loss": float(losses[-1]),
    }


def score_cross_encoder(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int | None,
) -> pd.DataFrame:
    model.eval()

    row_id_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            if (
                max_batches is not None and
                batch_index > max_batches
            ):
                break

            model_batch, row_ids = move_batch_to_device(
                batch,
                device,
            )

            model_batch.pop("labels")

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=True,
            ):
                logits = model(**model_batch).logits

            probabilities = torch.softmax(
                logits.float(),
                dim=-1,
            ).cpu().numpy()

            row_id_parts.append(row_ids.numpy())
            probability_parts.append(probabilities)
            predicted_parts.append(
                probabilities.argmax(axis=1)
            )

    if not row_id_parts:
        raise RuntimeError(
            "No validation batches were scored."
        )

    row_ids = np.concatenate(row_id_parts)
    probabilities = np.concatenate(
        probability_parts,
        axis=0,
    )
    predicted_ids = np.concatenate(predicted_parts)

    expected_gain = probabilities @ GAIN_VECTOR

    return pd.DataFrame(
        {
            "row_id": row_ids,
            "semantic_expected_gain": expected_gain,
            "semantic_predicted_label_id": predicted_ids,
            "semantic_predicted_label": [
                ID_TO_LABEL[int(value)]
                for value in predicted_ids
            ],
        }
    )


def rank_semantic_candidates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    ranked = frame.copy()

    ranked["_product_sort"] = ranked[
        "product_id"
    ].astype(str)

    ranked = ranked.sort_values(
        [
            "query_id",
            "semantic_expected_gain",
            "_product_sort",
        ],
        ascending=[True, False, True],
        kind="mergesort",
    ).copy()

    ranked["semantic_rank"] = (
        ranked.groupby("query_id")
        .cumcount()
        .add(1)
    )

    return ranked.drop(columns=["_product_sort"])


def per_query_ranking_metrics(
    ranked: pd.DataFrame,
    rank_column: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for query_id, rows in ranked.groupby(
        "query_id",
        sort=False,
    ):
        ordered = rows.sort_values(
            rank_column,
            kind="mergesort",
        )

        ordered_labels = ordered["esci_label"].tolist()

        gains = [
            GAIN_VECTOR[
                LABEL_TO_ID[label]
            ]
            for label in ordered_labels
        ]

        ideal_gains = sorted(
            gains,
            reverse=True,
        )

        observed_dcg = dcg_at_k(gains, 10)
        ideal_dcg = dcg_at_k(ideal_gains, 10)

        ndcg = (
            0.0
            if ideal_dcg == 0.0
            else observed_dcg / ideal_dcg
        )

        top1_label = ordered_labels[0]

        records.append(
            {
                "query_id": query_id,
                "ndcg_at_10": ndcg,
                "top1_high_relevance": int(
                    top1_label in HIGH_RELEVANCE_LABELS
                ),
                "top1_irrelevant": int(
                    top1_label == IRRELEVANT_LABEL
                ),
                "top1_label": top1_label,
            }
        )

    return pd.DataFrame(records)


def summary_metrics(
    per_query: pd.DataFrame,
) -> dict[str, float]:
    return {
        "ndcg_at_10": float(
            per_query["ndcg_at_10"].mean()
        ),
        "top1_high_relevance_precision": float(
            per_query[
                "top1_high_relevance"
            ].mean()
        ),
        "top1_irrelevant_rate": float(
            per_query["top1_irrelevant"].mean()
        ),
    }


def paired_query_bootstrap(
    lexical_query_metrics: pd.DataFrame,
    semantic_query_metrics: pd.DataFrame,
    replicate_count: int,
    seed: int,
) -> dict[str, Any]:
    paired = lexical_query_metrics.merge(
        semantic_query_metrics,
        on="query_id",
        how="inner",
        suffixes=("_lexical", "_semantic"),
        validate="1:1",
    )

    if len(paired) != len(lexical_query_metrics):
        raise RuntimeError(
            "Semantic and lexical query sets differ."
        )

    rng = np.random.default_rng(seed)

    output: dict[str, Any] = {
        "bootstrap_unit": "query_id",
        "replicate_count": replicate_count,
        "random_seed": seed,
    }

    for metric_name in (
        "ndcg_at_10",
        "top1_high_relevance",
        "top1_irrelevant",
    ):
        delta = (
            paired[f"{metric_name}_semantic"]
            .to_numpy(dtype=float) -
            paired[f"{metric_name}_lexical"]
            .to_numpy(dtype=float)
        )

        sampled_means = np.empty(
            replicate_count,
            dtype=float,
        )

        for start in range(0, replicate_count, 50):
            stop = min(start + 50, replicate_count)

            indices = rng.integers(
                low=0,
                high=len(delta),
                size=(stop - start, len(delta)),
            )

            sampled_means[start:stop] = (
                delta[indices].mean(axis=1)
            )

        output[f"{metric_name}_delta"] = float(
            delta.mean()
        )

        output[f"{metric_name}_delta_ci_95"] = [
            float(np.quantile(sampled_means, 0.025)),
            float(np.quantile(sampled_means, 0.975)),
        ]

    return output


def run_semantic_development(
    raw_root: Path,
    source_contract_path: Path,
    lexical_receipt_path: Path,
    output_root: Path,
    mode: str,
    requested_revision: str | None,
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full.")

    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists: {output_root}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable for semantic development."
        )

    set_reproducible_seed(20260704)

    fit, validation, provenance = load_development_frames(
        raw_root=raw_root,
        source_contract_path=source_contract_path,
        lexical_receipt_path=lexical_receipt_path,
    )

    fit = to_pair_frame(fit)
    validation = to_pair_frame(validation)

    backbone = "microsoft/deberta-v3-base"
    revision = resolve_model_revision(
        backbone,
        requested_revision,
    )

    tokenizer = load_deberta_tokenizer(
        backbone=backbone,
        revision=revision,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            backbone,
            revision=revision,
            num_labels=4,
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
        )
    )

    device = torch.device("cuda:0")
    amp_dtype = require_bf16_cuda(device)

    model.to(
        device=device,
        dtype=torch.float32,
    )

    parameter_summary = require_fp32_trainable_parameters(
        model
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    fit_loader = build_loader(
        fit,
        tokenizer=tokenizer,
        max_length=256,
        batch_size=8,
        shuffle=True,
        seed=20260704,
    )

    validation_loader = build_loader(
        validation,
        tokenizer=tokenizer,
        max_length=256,
        batch_size=16,
        shuffle=False,
        seed=20260704,
    )

    if mode == "smoke":
        max_fit_batches = 4
        max_validation_batches = 8
    else:
        max_fit_batches = None
        max_validation_batches = None

    started = time.perf_counter()

    train_summary = train_cross_encoder(
        model=model,
        loader=fit_loader,
        device=device,
        learning_rate=2.0e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        max_gradient_norm=1.0,
        amp_dtype=amp_dtype,
        max_batches=max_fit_batches,
    )

    semantic_scores = score_cross_encoder(
        model=model,
        loader=validation_loader,
        device=device,
        amp_dtype=amp_dtype,
        max_batches=max_validation_batches,
    )

    elapsed_seconds = time.perf_counter() - started

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    base_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "mode": mode,
        "official_test_row_level_used": False,
        "backbone": backbone,
        "backbone_revision": revision,
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
        "training_contract": {
            "loss": "standard_four_class_cross_entropy",
            "max_length": 256,
            "train_batch_size": 8,
            "evaluation_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2.0e-5,
            "weight_decay": 0.01,
            "mixed_precision": registered_amp_policy(),
            "seed": 20260704,
        },
        "provenance": provenance,
        "train_summary": train_summary,
        "validation_scored_candidate_count": int(
            len(semantic_scores)
        ),
        "elapsed_seconds": elapsed_seconds,
        "parameter_contract": parameter_summary,
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
    }

    if mode == "smoke":
        predicted_counts = {
            ID_TO_LABEL[int(label_id)]: int(count)
            for label_id, count in (
                semantic_scores[
                    "semantic_predicted_label_id"
                ]
                .value_counts()
                .sort_index()
                .items()
            )
        }

        base_payload.update(
            {
                "status": (
                    "GPU_CROSS_ENCODER_SMOKE_COMPLETE"
                ),
                "semantic_prediction_label_counts": (
                    predicted_counts
                ),
                "next_gate": (
                    "PIN_BACKBONE_REVISION_AND_RUN_FULL_"
                    "QUERY_DISJOINT_DEVELOPMENT_TRAINING"
                ),
            }
        )

        receipt_path = (
            output_root /
            "semantic_smoke_receipt.json"
        )

        receipt_path.write_text(
            json.dumps(
                base_payload,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        base_payload["receipt_path"] = str(
            receipt_path
        )
        base_payload["receipt_sha256"] = sha256_file(
            receipt_path
        )

        return base_payload

    scored_validation = validation.merge(
        semantic_scores,
        on="row_id",
        how="inner",
        validate="1:1",
    )

    if len(scored_validation) != len(validation):
        raise RuntimeError(
            "Full validation scoring did not cover every row."
        )

    semantic_ranked = rank_semantic_candidates(
        scored_validation
    )

    lexical_ranked = rank_lexical_candidates(
        validation
    )

    semantic_query_metrics = per_query_ranking_metrics(
        semantic_ranked,
        rank_column="semantic_rank",
    )

    lexical_query_metrics = per_query_ranking_metrics(
        lexical_ranked,
        rank_column="lexical_rank",
    )

    semantic_metric_summary = summary_metrics(
        semantic_query_metrics
    )

    lexical_metric_summary = summary_metrics(
        lexical_query_metrics
    )

    frozen_receipt = json.loads(
        lexical_receipt_path.read_text(
            encoding="utf-8"
        )
    )

    frozen_metrics = frozen_receipt[
        "lexical_comparator"
    ]["metrics"]

    for name, observed in (
        (
            "ndcg_at_10",
            lexical_metric_summary["ndcg_at_10"],
        ),
        (
            "top1_high_relevance_precision",
            lexical_metric_summary[
                "top1_high_relevance_precision"
            ],
        ),
        (
            "top1_irrelevant_rate",
            lexical_metric_summary[
                "top1_irrelevant_rate"
            ],
        ),
    ):
        expected = float(frozen_metrics[name])

        if not np.isclose(
            observed,
            expected,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Frozen lexical metric drifted for {name}. "
                f"Expected={expected} Observed={observed}"
            )

    macro_f1 = float(
        f1_score(
            scored_validation["label_id"],
            scored_validation[
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

    semantic_metric_summary["macro_f1"] = macro_f1

    gate_pass = (
        bootstrap["ndcg_at_10_delta_ci_95"][0] > 0 and
        bootstrap[
            "top1_high_relevance_delta_ci_95"
        ][0] > 0 and
        bootstrap[
            "top1_irrelevant_delta_ci_95"
        ][1] <= 0
    )

    metrics_path = output_root / "development_metrics.json"
    query_metrics_path = (
        output_root /
        "paired_query_ranking_metrics.csv"
    )
    checkpoint_dir = output_root / "model"

    merged_query_metrics = lexical_query_metrics.merge(
        semantic_query_metrics,
        on="query_id",
        how="inner",
        suffixes=("_lexical", "_semantic"),
        validate="1:1",
    )

    merged_query_metrics.to_csv(
        query_metrics_path,
        index=False,
    )

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(checkpoint_dir)

    model_files = {
        path.relative_to(checkpoint_dir).as_posix(): {
            "byte_count": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in checkpoint_dir.rglob("*")
        if path.is_file()
    }

    base_payload.update(
        {
            "status": (
                "GPU_CROSS_ENCODER_DEVELOPMENT_COMPLETE"
            ),
            "frozen_lexical_metrics": lexical_metric_summary,
            "semantic_metrics": semantic_metric_summary,
            "paired_query_bootstrap": bootstrap,
            "model_artifact_manifest": model_files,
            "gate_pass": gate_pass,
            "next_gate": (
                "FREEZE_GPU_CROSS_ENCODER_AND_AUTHORIZE_"
                "ONE_OFFICIAL_TEST_EVALUATION"
                if gate_pass
                else
                "STOP_OFFICIAL_TEST_SEMANTIC_MODEL_DID_NOT_"
                "PASS_PRE_REGISTERED_DEVELOPMENT_GATE"
            ),
        }
    )

    metrics_path.write_text(
        json.dumps(
            base_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    base_payload["metrics_path"] = str(metrics_path)
    base_payload["metrics_sha256"] = sha256_file(
        metrics_path
    )

    return base_payload