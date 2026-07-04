from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.semantic_development import (
    run_semantic_development,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPU semantic query-product cross-encoder "
            "development or an integration smoke check."
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
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        required=True,
    )
    parser.add_argument(
        "--backbone-revision",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = run_semantic_development(
        raw_root=args.raw_root.resolve(),
        source_contract_path=(
            args.source_contract.resolve()
        ),
        lexical_receipt_path=(
            args.lexical_receipt.resolve()
        ),
        output_root=args.output_root.resolve(),
        mode=args.mode,
        requested_revision=args.backbone_revision,
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