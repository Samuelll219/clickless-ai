"""Validation for ClickLess AI constraint extraction."""

from __future__ import annotations

from typing import Any

from clickless_kg.vocab import (
    ALLERGEN_NAMES,
    CATEGORY_GROUPS,
    CATEGORY_NAMES,
    DIETARY_TAG_NAMES,
)


class ValidationError(ValueError):
    """Raised when extracted constraints are invalid."""


REQUIRED_FIELDS = {
    "category",
    "dietary_tags",
    "excluded_allergens",
    "max_price",
    "quantity",
}


def validate_constraints(data: dict[str, Any]) -> dict[str, Any]:
    """Validate extracted constraints against the shared KG vocabulary."""

    if not isinstance(data, dict):
        raise ValidationError("Extracted constraints must be a JSON object.")

    # Check fields
    fields = set(data.keys())

    missing = REQUIRED_FIELDS - fields
    extra = fields - REQUIRED_FIELDS

    if missing:
        raise ValidationError(
            f"Missing required fields: {sorted(missing)}"
        )

    if extra:
        raise ValidationError(
            f"Unexpected fields: {sorted(extra)}"
        )

    # Validate category
    valid_categories = set(CATEGORY_NAMES) | set(CATEGORY_GROUPS)

    if data["category"] not in valid_categories:
        raise ValidationError(
            f"Invalid category: {data['category']}"
        )

    # Validate dietary tags
    if not isinstance(data["dietary_tags"], list):
        raise ValidationError("dietary_tags must be a list.")

    invalid_tags = set(data["dietary_tags"]) - set(DIETARY_TAG_NAMES)

    if invalid_tags:
        raise ValidationError(
            f"Invalid dietary tags: {sorted(invalid_tags)}"
        )

    # Validate allergens
    if not isinstance(data["excluded_allergens"], list):
        raise ValidationError("excluded_allergens must be a list.")

    invalid_allergens = (
        set(data["excluded_allergens"]) - set(ALLERGEN_NAMES)
    )

    if invalid_allergens:
        raise ValidationError(
            f"Invalid allergens: {sorted(invalid_allergens)}"
        )

    # Validate maximum price
    max_price = data["max_price"]

    if max_price is not None:
        if isinstance(max_price, bool) or not isinstance(
            max_price, (int, float)
        ):
            raise ValidationError(
                "max_price must be a number or null."
            )

        if max_price < 0:
            raise ValidationError(
                "max_price cannot be negative."
            )

    # Validate quantity
    quantity = data["quantity"]

    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or quantity < 1
    ):
        raise ValidationError(
            "quantity must be an integer greater than or equal to 1."
        )

    # Remove accidental duplicates while preserving order
    data["dietary_tags"] = list(
        dict.fromkeys(data["dietary_tags"])
    )

    data["excluded_allergens"] = list(
        dict.fromkeys(data["excluded_allergens"])
    )

    return data
