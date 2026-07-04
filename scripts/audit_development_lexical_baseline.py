from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.development import (
    audit_development_lexical_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a query-disjoint ESCI development split "
            "and evaluate the frozen lexical comparator."
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
        "--output-root",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = audit_development_lexical_baseline(
        raw_root=args.raw_root.resolve(),
        source_contract_path=(
            args.source_contract.resolve()
        ),
        output_root=args.output_root.resolve(),
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "official_test_row_level_used": (
                    payload[
                        "official_test_row_level_used"
                    ]
                ),
                "development_split": (
                    payload["development_split"]
                ),
                "development_dataset_summary": (
                    payload[
                        "development_dataset_summary"
                    ]
                ),
                "lexical_comparator": (
                    payload[
                        "lexical_comparator"
                    ]
                ),
                "next_gate": payload["next_gate"],
                "receipt_path": payload["receipt_path"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()