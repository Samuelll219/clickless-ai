"""Tests for ClickLess AI constraint-extraction evaluation."""

import pytest

from clickless_llm.evaluation import (
    evaluate_extractor,
    evaluate_predictions,
    exact_match,
    field_matches,
)


EXPECTED = {
    "category": "snacks",
    "dietary_tags": ["organic", "gluten_free"],
    "excluded_allergens": ["peanuts"],
    "max_price": 15,
    "quantity": 2,
}


def test_exact_match_identical_constraints():
    predicted = EXPECTED.copy()

    assert exact_match(predicted, EXPECTED)


def test_list_order_does_not_affect_match():
    predicted = {
        "category": "snacks",
        "dietary_tags": ["gluten_free", "organic"],
        "excluded_allergens": ["peanuts"],
        "max_price": 15,
        "quantity": 2,
    }

    assert exact_match(predicted, EXPECTED)


def test_exact_match_fails_when_one_field_differs():
    predicted = EXPECTED.copy()
    predicted["quantity"] = 3

    assert not exact_match(predicted, EXPECTED)


def test_field_matches():
    predicted = EXPECTED.copy()

    assert field_matches(
        predicted,
        EXPECTED,
        "category",
    )

    assert field_matches(
        predicted,
        EXPECTED,
        "dietary_tags",
    )


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        field_matches(
            EXPECTED,
            EXPECTED,
            "unknown_field",
        )


def test_evaluate_predictions_perfect():
    examples = [
        (EXPECTED.copy(), EXPECTED.copy()),
        (EXPECTED.copy(), EXPECTED.copy()),
    ]

    result = evaluate_predictions(examples)

    assert result["total"] == 2
    assert result["exact_matches"] == 2
    assert result["exact_match_accuracy"] == 1.0

    for accuracy in result["field_accuracy"].values():
        assert accuracy == 1.0


def test_evaluate_predictions_partial():
    first = EXPECTED.copy()

    second_expected = EXPECTED.copy()
    second_expected["quantity"] = 4

    second_predicted = EXPECTED.copy()
    second_predicted["quantity"] = 1

    examples = [
        (first, EXPECTED),
        (second_predicted, second_expected),
    ]

    result = evaluate_predictions(examples)

    assert result["total"] == 2
    assert result["exact_matches"] == 1
    assert result["exact_match_accuracy"] == 0.5

    assert result["field_accuracy"]["quantity"] == 0.5

    assert result["field_accuracy"]["category"] == 1.0
    assert result["field_accuracy"]["dietary_tags"] == 1.0
    assert result["field_accuracy"]["excluded_allergens"] == 1.0
    assert result["field_accuracy"]["max_price"] == 1.0


def test_empty_evaluation():
    result = evaluate_predictions([])

    assert result["total"] == 0
    assert result["exact_matches"] == 0
    assert result["exact_match_accuracy"] == 0.0

    for accuracy in result["field_accuracy"].values():
        assert accuracy == 0.0


def test_evaluate_extractor():
    dataset = [
        {
            "prompt": "Get me 2 organic gluten-free snacks under $15 "
            "without peanuts",
            "expected": EXPECTED,
        }
    ]

    def fake_extractor(prompt):
        assert "organic" in prompt
        return EXPECTED.copy()

    result = evaluate_extractor(
        dataset,
        fake_extractor,
    )

    assert result["total"] == 1
    assert result["exact_matches"] == 1
    assert result["exact_match_accuracy"] == 1.0


def test_missing_prompt_rejected():
    dataset = [
        {
            "expected": EXPECTED,
        }
    ]

    with pytest.raises(ValueError):
        evaluate_extractor(
            dataset,
            lambda prompt: EXPECTED,
        )


def test_missing_expected_rejected():
    dataset = [
        {
            "prompt": "Get me a snack",
        }
    ]

    with pytest.raises(ValueError):
        evaluate_extractor(
            dataset,
            lambda prompt: EXPECTED,
        )
