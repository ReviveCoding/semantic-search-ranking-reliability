from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.official_test import (
    run_one_shot_official_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the authorized one-shot official ESCI "
            "test evaluation."
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
        "--output-parent",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--backbone-revision",
        required=True,
    )
    parser.add_argument(
        "--final-epochs",
        required=True,
        type=int,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = run_one_shot_official_test(
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
        output_parent=args.output_parent.resolve(),
        output_root=args.output_root.resolve(),
        backbone_revision=args.backbone_revision,
        final_epochs=args.final_epochs,
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