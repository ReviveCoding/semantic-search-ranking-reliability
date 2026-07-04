from __future__ import annotations

import re
from collections.abc import Mapping


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    if value is None:
        return ""

    return WHITESPACE_PATTERN.sub(
        " ",
        str(value),
    ).strip()


def build_product_text(
    record: Mapping[str, object],
) -> str:
    sections: list[str] = []

    for field_name, prefix in (
        ("product_title", "[TITLE]"),
        ("product_brand", "[BRAND]"),
        ("product_color", "[COLOR]"),
        ("product_bullet_point", "[BULLETS]"),
    ):
        value = normalize_text(record.get(field_name))

        if value:
            sections.append(f"{prefix} {value}")

    return "\n".join(sections)


def build_query_product_pair(
    record: Mapping[str, object],
) -> tuple[str, str]:
    query = normalize_text(record.get("query"))

    if not query:
        raise ValueError("Query text must be non-empty.")

    product_text = build_product_text(record)

    if not product_text:
        raise ValueError("Product text must be non-empty.")

    return query, product_text