from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.reliability_report import (
    build_read_only_reliability_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only reliability report from existing "
            "official-test artifacts."
        )
    )

    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-contract",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--lexical-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--development-metrics",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--official-receipt",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = build_read_only_reliability_report(
        raw_root=args.raw_root.resolve(),
        source_contract_path=(
            args.source_contract.resolve()
        ),
        lexical_receipt_path=(
            args.lexical_receipt.resolve()
        ),
        development_metrics_path=(
            args.development_metrics.resolve()
        ),
        official_receipt_path=(
            args.official_receipt.resolve()
        ),
        output_root=args.output_root.resolve(),
    )

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
