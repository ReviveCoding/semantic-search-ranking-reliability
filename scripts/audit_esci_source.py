from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_reranker.esci_source import audit_esci_us_task1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate official ESCI US Task 1 raw-source "
            "provenance and schema."
        )
    )

    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-commit",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = audit_esci_us_task1(
        raw_root=args.raw_root.resolve(),
        output_root=args.output_root.resolve(),
        source_commit=args.source_commit,
    )

    print(
        json.dumps(
            {
                "status": payload["status"],
                "source_commit": payload["source"]["commit"],
                "task1_us_summary": (
                    payload["task1_us_summary"]
                ),
                "claim_boundary": payload["claim_boundary"],
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