from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name} under {root}; found {len(matches)}."
        )
    return matches[0]


def load_json(path: Path, status: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = payload.get("status")
    if actual != status:
        raise RuntimeError(
            f"Unexpected status in {path.name}: {actual!r}; expected {status!r}."
        )
    return payload


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def reduction_pp(value: float) -> str:
    return f"{abs(value) * 100:.2f} percentage points"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        escaped = [
            str(cell).replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")
            for cell in row
        ]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.rstrip() + "\n", encoding="utf-8", newline="\n")


def git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_read_only_contract(read_only: dict[str, Any]) -> None:
    contract = read_only.get("read_only_contract", {})
    bad = [
        key
        for key in (
            "model_loaded",
            "model_trained",
            "official_test_re_evaluated",
            "test_used_for_model_selection",
        )
        if contract.get(key) is not False
    ]
    if bad:
        raise RuntimeError(
            "Read-only reliability contract was violated for: "
            + ", ".join(bad)
        )


def main(project_root: Path) -> None:
    project_root = project_root.resolve()
    artifacts = project_root / "artifacts"

    source_path = find_one(
        artifacts / "esci_source_contract",
        "esci_us_task1_source_contract.json",
    )
    lexical_path = find_one(
        artifacts / "development_lexical_baseline",
        "development_lexical_baseline_receipt.json",
    )
    development_path = find_one(
        artifacts / "semantic_development_full",
        "development_metrics.json",
    )
    official_path = find_one(
        artifacts / "official_test_evaluation",
        "official_test_evaluation_receipt.json",
    )
    read_only_path = find_one(
        artifacts / "read_only_reliability_report",
        "read_only_reliability_summary.json",
    )

    source = load_json(source_path, "ESCI_US_TASK1_SOURCE_CONTRACT_PASS")
    _lexical_receipt = load_json(
        lexical_path,
        "QUERY_DISJOINT_DEVELOPMENT_AND_FROZEN_LEXICAL_BASELINE_COMPLETE",
    )
    development = load_json(
        development_path,
        "GPU_CROSS_ENCODER_DEVELOPMENT_COMPLETE",
    )
    official = load_json(
        official_path,
        "ONE_SHOT_OFFICIAL_TEST_EVALUATION_COMPLETE",
    )
    read_only = load_json(
        read_only_path,
        "READ_ONLY_RELIABILITY_REPORT_COMPLETE",
    )

    if development.get("gate_pass") is not True:
        raise RuntimeError("Development gate must be PASS before packaging.")
    if official.get("model_selection_after_test_allowed") is not False:
        raise RuntimeError("Official receipt does not close model selection.")
    if official.get("official_test_usage_role") != "final_evaluation_only":
        raise RuntimeError("Unexpected official-test usage role.")
    require_read_only_contract(read_only)

    report_paths = read_only.get("report_paths", {})
    support_path = Path(report_paths["support_slices_csv"])
    taxonomy_path = Path(report_paths["taxonomy_csv"])
    if not support_path.is_file() or not taxonomy_path.is_file():
        raise RuntimeError("Read-only report support/taxonomy artifacts are missing.")

    support_rows = read_csv_rows(support_path)
    taxonomy_rows = read_csv_rows(taxonomy_path)

    lexical = official["frozen_lexical_metrics"]
    semantic = official["semantic_metrics"]
    boot = official["paired_query_bootstrap"]
    dataset = official["dataset_summary"]
    gpu = official["gpu"]
    model_hash = official["model_artifact_manifest"]["model.safetensors"]["sha256"]

    resume_claim = (
        "Built a GPU-accelerated four-class semantic query-product reranker "
        "on the official Amazon ESCI benchmark using a pinned DeBERTa-v3 "
        "cross-encoder and gain-aligned ranking score. In a one-shot held-out "
        f"official test evaluation, improved nDCG@10 from {lexical['ndcg_at_10']:.3f} "
        f"to {semantic['ndcg_at_10']:.3f}, increased Top-1 Exact-or-Substitute "
        f"precision by {pp(boot['top1_high_relevance_delta'])}, and reduced "
        f"irrelevant Top-1 selections by {reduction_pp(boot['top1_irrelevant_delta'])} "
        "versus a frozen lexical baseline."
    )

    result_table = md_table(
        ["Metric", "Frozen lexical", "Semantic reranker", "Change"],
        [
            [
                "nDCG@10",
                f"{lexical['ndcg_at_10']:.6f}",
                f"{semantic['ndcg_at_10']:.6f}",
                f"{boot['ndcg_at_10_delta']:+.6f}",
            ],
            [
                "Top-1 E/S precision",
                pct(lexical["top1_high_relevance_precision"]),
                pct(semantic["top1_high_relevance_precision"]),
                pp(boot["top1_high_relevance_delta"]),
            ],
            [
                "Top-1 irrelevant rate",
                pct(lexical["top1_irrelevant_rate"]),
                pct(semantic["top1_irrelevant_rate"]),
                pp(boot["top1_irrelevant_delta"]),
            ],
            [
                "Four-class macro F1",
                "N/A",
                f"{semantic['macro_f1']:.6f}",
                "N/A",
            ],
        ],
    )
    bootstrap_table = md_table(
        ["Metric", "Point estimate", "95% paired-query bootstrap CI"],
        [
            [
                "nDCG@10",
                f"{boot['ndcg_at_10_delta']:+.6f}",
                (
                    f"[{boot['ndcg_at_10_delta_ci_95'][0]:+.6f}, "
                    f"{boot['ndcg_at_10_delta_ci_95'][1]:+.6f}]"
                ),
            ],
            [
                "Top-1 E/S precision",
                pp(boot["top1_high_relevance_delta"]),
                (
                    f"[{pp(boot['top1_high_relevance_delta_ci_95'][0])}, "
                    f"{pp(boot['top1_high_relevance_delta_ci_95'][1])}]"
                ),
            ],
            [
                "Top-1 irrelevant rate",
                pp(boot["top1_irrelevant_delta"]),
                (
                    f"[{pp(boot['top1_irrelevant_delta_ci_95'][0])}, "
                    f"{pp(boot['top1_irrelevant_delta_ci_95'][1])}]"
                ),
            ],
        ],
    )
    consistency_table = md_table(
        [
            "Metric",
            "Development lexical",
            "Development semantic",
            "Official-test lexical",
            "Official-test semantic",
        ],
        [
            [
                "nDCG@10",
                f"{development['frozen_lexical_metrics']['ndcg_at_10']:.6f}",
                f"{development['semantic_metrics']['ndcg_at_10']:.6f}",
                f"{lexical['ndcg_at_10']:.6f}",
                f"{semantic['ndcg_at_10']:.6f}",
            ],
            [
                "Top-1 E/S precision",
                pct(development["frozen_lexical_metrics"]["top1_high_relevance_precision"]),
                pct(development["semantic_metrics"]["top1_high_relevance_precision"]),
                pct(lexical["top1_high_relevance_precision"]),
                pct(semantic["top1_high_relevance_precision"]),
            ],
            [
                "Top-1 irrelevant rate",
                pct(development["frozen_lexical_metrics"]["top1_irrelevant_rate"]),
                pct(development["semantic_metrics"]["top1_irrelevant_rate"]),
                pct(lexical["top1_irrelevant_rate"]),
                pct(semantic["top1_irrelevant_rate"]),
            ],
        ],
    )
    support_table = md_table(
        [
            "Frozen lexical support bucket",
            "Queries",
            "Lexical E/S",
            "Semantic E/S",
            "E/S change",
            "Lexical I",
            "Semantic I",
            "I-rate change",
        ],
        [
            [
                row["frozen_lexical_support_bucket"],
                row["query_count"],
                pct(float(row["lexical_top1_high_relevance_precision"])),
                pct(float(row["semantic_top1_high_relevance_precision"])),
                pp(float(row["top1_high_relevance_delta"])),
                pct(float(row["lexical_top1_irrelevant_rate"])),
                pct(float(row["semantic_top1_irrelevant_rate"])),
                pp(float(row["top1_irrelevant_delta"])),
            ]
            for row in support_rows
        ],
    )
    taxonomy_table = md_table(
        ["Top-1 transition", "Queries", "Share"],
        [
            [
                row["top1_change_type"],
                row["query_count"],
                pct(float(row["share_of_queries"])),
            ]
            for row in taxonomy_rows
        ],
    )

    readme = f"""# Reliability-First Semantic Search Ranking

GPU-accelerated semantic query-product reranking on the official Amazon Shopping Queries ESCI relevance benchmark.

## Final official held-out test result

{result_table}

## What this project demonstrates

- A transparent lexical comparator frozen before semantic training.
- Query-disjoint development validation using only official train rows.
- A pinned `microsoft/deberta-v3-base` revision: `{official['backbone_revision']}`.
- A four-class Exact / Substitute / Complement / Irrelevant cross-encoder.
- One official held-out test evaluation after the development gate passed.
- Closed model selection after final evaluation.

## Architecture

```mermaid
flowchart LR
    Q[Query] --> X[DeBERTa-v3 cross-encoder]
    P[Title + brand + color + bullets] --> X
    X --> C[E / S / C / I probabilities]
    C --> G[Expected ESCI gain]
    G --> R[Candidate-set reranking]
```

## Paired-query uncertainty evidence

{bootstrap_table}

## Claim boundary

This is an offline candidate-set semantic reranking result on static ESCI relevance labels. It does not claim production deployment, CTR lift, conversion lift, advertising causality, or end-to-end retrieval quality.

## Reproducibility anchors

- ESCI source commit: `{source['source']['commit']}`
- Official-test receipt SHA-256: `{sha256_file(official_path)}`
- Final model SHA-256: `{model_hash}`
"""

    technical = f"""# Technical Documentation Report

## Executive summary

{resume_claim}

## Evaluation design

Development used only official ESCI train rows under a query-disjoint split. The lexical comparator was frozen before GPU semantic training. The selected semantic configuration had to pass pre-registered development gates before exactly one official held-out evaluation.

{consistency_table}

## Official held-out result

{result_table}

## Paired-query uncertainty evidence

{bootstrap_table}

## Model and runtime

- Backbone: `{official['backbone']}`
- Revision: `{official['backbone_revision']}`
- Inputs: query, title, brand, color, and bullet points
- Objective: four-class cross entropy
- Ranking score: expected ESCI gain from class probabilities
- GPU: `{gpu['name']}`
- BF16 supported: `{gpu['bf16_supported']}`
- Peak allocated GPU memory: `{gpu['peak_allocated_gb']:.3f} GB`
- Final training runtime: `{official['elapsed_seconds'] / 3600:.2f} hours`

## Dataset lineage

- ESCI source commit: `{source['source']['commit']}`
- Train candidate rows: `{dataset['train_candidate_row_count']}`
- Test candidate rows: `{dataset['test_candidate_row_count']}`
- Train queries: `{dataset['train_query_count']}`
- Test queries: `{dataset['test_query_count']}`
- Train/test query overlap: `{dataset['train_test_query_overlap_detected']}`

## Frozen-lexical-anchor support diagnostics

Support is a training-only product appearance count, used here only for post-hoc diagnostics. It was not a model input or selection criterion.

{support_table}

## Top-1 transition taxonomy

{taxonomy_table}

## Reproducibility anchors

- Source receipt SHA-256: `{sha256_file(source_path)}`
- Frozen lexical receipt SHA-256: `{sha256_file(lexical_path)}`
- Development receipt SHA-256: `{sha256_file(development_path)}`
- Official-test receipt SHA-256: `{sha256_file(official_path)}`
- Read-only report summary SHA-256: `{sha256_file(read_only_path)}`
- Final model SHA-256: `{model_hash}`

## Claim boundary

Supported: offline candidate-set semantic reranking on the official ESCI relevance benchmark.

Not supported: production deployment, CTR or conversion lift, advertising causality, end-to-end retrieval quality, or production cold-start claims.
"""

    guide = f"""# Resume and Interview Claim Guide

## Primary resume bullet

{resume_claim}

## Search / ranking bullet

- Built a reliability-first semantic product-search reranker with a pinned DeBERTa-v3 cross-encoder; improved nDCG@10 from `{lexical['ndcg_at_10']:.3f}` to `{semantic['ndcg_at_10']:.3f}` and reduced irrelevant Top-1 selections by `{reduction_pp(boot['top1_irrelevant_delta'])}` on a one-shot official ESCI held-out test.

## Applied Scientist bullet

- Designed a query-disjoint development and one-shot held-out evaluation protocol for GPU semantic ranking; increased Top-1 Exact-or-Substitute precision by `{pp(boot['top1_high_relevance_delta'])}` with paired-query bootstrap confidence intervals supporting all pre-registered gates.

## 30-second explanation

I built an offline semantic candidate reranker for product search. I froze a transparent lexical baseline, trained a four-class DeBERTa-v3 cross-encoder on query and product metadata, and ranked candidates by expected relevance gain. The reliability work was query-disjoint development, a pinned backbone revision, a pre-registered development gate, and one official test evaluation. The final held-out result improved nDCG@10 from `{lexical['ndcg_at_10']:.3f}` to `{semantic['ndcg_at_10']:.3f}` while reducing irrelevant Top-1 selections by `{reduction_pp(boot['top1_irrelevant_delta'])}`.

## Limitations to state

- Offline candidate-set reranking, not full retrieval.
- Static relevance labels, not clicks or conversion logs.
- No production latency or deployment claim.
- No causal advertising or marketplace business-impact claim.
"""

    checklist = f"""# GitHub Publishing Checklist

## Before publishing

- [ ] Confirm `git status --short` is empty.
- [ ] Keep `data/raw/`, `.venv/`, model artifacts, Hugging Face cache, and local artifact runs untracked.
- [ ] Remove local usernames, absolute paths, tokens, and credentials from public files.
- [ ] Keep the official-test receipt unchanged.
- [ ] Do not re-run official test evaluation.
- [ ] Verify README rendering on GitHub.
- [ ] Choose a license before public release.

## Public claim

{resume_claim}

## Do not claim

- Production deployment
- CTR or conversion lift
- Advertising causality
- End-to-end retrieval quality
"""

    manifest = {
        "status": "PORTFOLIO_DOCUMENTATION_EVIDENCE_MANIFEST",
        "evidence_code_revision_before_packaging": git_head(project_root),
        "model_selection_closed": True,
        "source_commit": source["source"]["commit"],
        "backbone": official["backbone"],
        "backbone_revision": official["backbone_revision"],
        "official_test_receipt_sha256": sha256_file(official_path),
        "final_model_safetensors_sha256": model_hash,
        "final_metrics": {
            "frozen_lexical": lexical,
            "semantic": semantic,
            "paired_query_bootstrap": boot,
        },
        "claim_boundary": {
            "supported": (
                "Offline candidate-set semantic reranking on the official "
                "ESCI relevance benchmark."
            ),
            "not_supported": [
                "production deployment",
                "CTR lift",
                "conversion lift",
                "advertising causality",
                "end-to-end retrieval quality",
            ],
        },
    }

    write(project_root / "README.md", readme)
    write(project_root / "docs" / "09_technical_documentation_report.md", technical)
    write(project_root / "docs" / "10_resume_and_interview_claim_guide.md", guide)
    write(project_root / "docs" / "11_release_and_github_publishing_checklist.md", checklist)
    write(
        project_root / "docs" / "12_portfolio_evidence_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )

    payload = {
        "status": "PORTFOLIO_DOCUMENTATION_COMPLETE",
        "model_selection_closed": True,
        "official_test_re_evaluated": False,
        "model_loaded": False,
        "model_trained": False,
        "official_test_receipt_sha256": sha256_file(official_path),
        "documentation_paths": {
            "readme": str(project_root / "README.md"),
            "technical_report": str(project_root / "docs" / "09_technical_documentation_report.md"),
            "resume_interview_guide": str(project_root / "docs" / "10_resume_and_interview_claim_guide.md"),
            "publishing_checklist": str(project_root / "docs" / "11_release_and_github_publishing_checklist.md"),
            "evidence_manifest": str(project_root / "docs" / "12_portfolio_evidence_manifest.json"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: build_portfolio_documentation.py <project_root>")
    main(Path(sys.argv[1]))
