"""Evaluation utilities for ClickLess AI constraint extraction.

Implements the constraint-extraction metrics used by the project:
- exact-match accuracy
- per-field accuracy

The evaluator compares validated predicted constraint dictionaries against
ground-truth dictionaries.
"""

from __future__ import annotations

from typing import Any, Iterable


CONSTRAINT_FIELDS = (
    "category",
    "dietary_tags",
    "excluded_allergens",
    "max_price",
    "quantity",
)


def _normalise_value(
    field: str,
    value: Any,
) -> Any:
    """Normalise values for fair evaluation.

    dietary_tags and excluded_allergens are treated as unordered sets,
    because their order has no semantic meaning.
    """

    if field in {
        "dietary_tags",
        "excluded_allergens",
    }:
        if value is None:
            return tuple()

        return tuple(sorted(set(value)))

    return value


def field_matches(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    field: str,
) -> bool:
    """Return whether one constraint field matches ground truth."""

    if field not in CONSTRAINT_FIELDS:
        raise ValueError(
            f"Unknown constraint field: {field}"
        )

    if field not in predicted or field not in expected:
        return False

    return _normalise_value(
        field,
        predicted[field],
    ) == _normalise_value(
        field,
        expected[field],
    )


def exact_match(
    predicted: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    """Return True only when every constraint field matches."""

    return all(
        field_matches(
            predicted,
            expected,
            field,
        )
        for field in CONSTRAINT_FIELDS
    )


def evaluate_predictions(
    examples: Iterable[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> dict[str, Any]:
    """Evaluate predicted constraints against ground truth.

    Parameters
    ----------
    examples:
        Iterable of (predicted, expected) constraint dictionary pairs.

    Returns
    -------
    dict
        total:
            Number of evaluated examples.

        exact_matches:
            Number of examples where all five fields match.

        exact_match_accuracy:
            exact_matches / total.

        field_correct:
            Number of correct predictions for each field.

        field_accuracy:
            Accuracy for each individual field.
    """

    examples = list(examples)

    total = len(examples)

    field_correct = {
        field: 0
        for field in CONSTRAINT_FIELDS
    }

    exact_matches = 0

    for predicted, expected in examples:
        all_correct = True

        for field in CONSTRAINT_FIELDS:
            correct = field_matches(
                predicted,
                expected,
                field,
            )

            if correct:
                field_correct[field] += 1
            else:
                all_correct = False

        if all_correct:
            exact_matches += 1

    if total == 0:
        field_accuracy = {
            field: 0.0
            for field in CONSTRAINT_FIELDS
        }

        exact_accuracy = 0.0

    else:
        field_accuracy = {
            field: field_correct[field] / total
            for field in CONSTRAINT_FIELDS
        }

        exact_accuracy = exact_matches / total

    return {
        "total": total,
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_accuracy,
        "field_correct": field_correct,
        "field_accuracy": field_accuracy,
    }


def evaluate_extractor(
    dataset: Iterable[dict[str, Any]],
    extractor,
) -> dict[str, Any]:
    """Run an extractor over a labelled evaluation dataset.

    Each dataset row must contain:

        {
            "prompt": "...",
            "expected": {
                "category": ...,
                "dietary_tags": [...],
                "excluded_allergens": [...],
                "max_price": ...,
                "quantity": ...
            }
        }

    The extractor must accept a prompt string and return a validated
    constraint dictionary.
    """

    pairs = []

    for example in dataset:
        if "prompt" not in example:
            raise ValueError(
                "Evaluation example is missing 'prompt'."
            )

        if "expected" not in example:
            raise ValueError(
                "Evaluation example is missing 'expected'."
            )

        predicted = extractor(
            example["prompt"]
        )

        pairs.append(
            (
                predicted,
                example["expected"],
            )
        )

    return evaluate_predictions(pairs)
