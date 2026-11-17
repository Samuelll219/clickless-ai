"""Tests for the ClickLess AI LLM pipeline."""

import pytest

from clickless_llm.extractor import extract_constraints
from clickless_llm.pipeline import extract_and_validate
from clickless_llm.validator import (
    ValidationError,
    validate_constraints,
)


# ---------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------

def test_validator_accepts_valid_constraints():
    data = {
        "category": "snacks",
        "dietary_tags": ["vegan", "organic"],
        "excluded_allergens": ["peanuts"],
        "max_price": 10,
        "quantity": 2,
    }

    result = validate_constraints(data)

    assert result == data


def test_validator_rejects_invalid_category():
    data = {
        "category": "healthy_food",
        "dietary_tags": [],
        "excluded_allergens": [],
        "max_price": None,
        "quantity": 1,
    }

    with pytest.raises(ValidationError):
        validate_constraints(data)


def test_validator_rejects_invalid_dietary_tag():
    data = {
        "category": "snacks",
        "dietary_tags": ["high_protein"],
        "excluded_allergens": [],
        "max_price": 10,
        "quantity": 1,
    }

    with pytest.raises(ValidationError):
        validate_constraints(data)


def test_validator_rejects_invalid_allergen():
    data = {
        "category": "snacks",
        "dietary_tags": [],
        "excluded_allergens": ["random_allergen"],
        "max_price": None,
        "quantity": 1,
    }

    with pytest.raises(ValidationError):
        validate_constraints(data)


def test_validator_rejects_negative_price():
    data = {
        "category": "snacks",
        "dietary_tags": [],
        "excluded_allergens": [],
        "max_price": -5,
        "quantity": 1,
    }

    with pytest.raises(ValidationError):
        validate_constraints(data)


def test_validator_rejects_zero_quantity():
    data = {
        "category": "snacks",
        "dietary_tags": [],
        "excluded_allergens": [],
        "max_price": None,
        "quantity": 0,
    }

    with pytest.raises(ValidationError):
        validate_constraints(data)


def test_validator_rejects_missing_field():
    data = {
        "category": "snacks",
        "dietary_tags": [],
        "excluded_allergens": [],
        "max_price": None,
    }

    with pytest.raises(ValidationError):
        validate_constraints(data)


def test_validator_removes_duplicates():
    data = {
        "category": "snacks",
        "dietary_tags": ["vegan", "vegan"],
        "excluded_allergens": ["peanuts", "peanuts"],
        "max_price": None,
        "quantity": 1,
    }

    result = validate_constraints(data)

    assert result["dietary_tags"] == ["vegan"]
    assert result["excluded_allergens"] == ["peanuts"]


# ---------------------------------------------------------------------
# Live Ollama extraction tests
#
# These require the local Ollama server and qwen2.5:7b.
# ---------------------------------------------------------------------

@pytest.mark.integration
def test_generic_snack_preserves_group():
    result = extract_and_validate(
        "I want a nut-free vegan snack"
    )

    assert result["category"] == "snacks"
    assert "vegan" in result["dietary_tags"]
    assert set(result["excluded_allergens"]) == {
        "peanuts",
        "tree_nuts",
    }
    assert result["max_price"] is None
    assert result["quantity"] == 1


@pytest.mark.integration
def test_specific_snack_bar():
    result = extract_and_validate(
        "Get me 2 organic gluten-free snack bars under $10"
    )

    assert result["category"] == "snack_bars"
    assert set(result["dietary_tags"]) == {
        "organic",
        "gluten_free",
    }
    assert result["excluded_allergens"] == []
    assert result["max_price"] == 10
    assert result["quantity"] == 2


@pytest.mark.integration
def test_multiple_constraints():
    result = extract_constraints(
        "Get me 3 gluten-free organic snacks under $12 "
        "without peanuts"
    )

    result = validate_constraints(result)

    assert result["category"] == "snacks"
    assert set(result["dietary_tags"]) == {
        "gluten_free",
        "organic",
    }
    assert result["excluded_allergens"] == ["peanuts"]
    assert result["max_price"] == 12
    assert result["quantity"] == 3
def test_validation_failure_is_used_for_retry(monkeypatch):
    """A validation error should be fed back into the corrective retry."""

    calls = []

    invalid = {
        "category": "snacks",
        "dietary_tags": ["high_protein"],
        "excluded_allergens": [],
        "max_price": 10,
        "quantity": 1,
    }

    valid = {
        "category": "snacks",
        "dietary_tags": ["vegan"],
        "excluded_allergens": [],
        "max_price": 10,
        "quantity": 1,
    }

    def fake_extract(request, model=None):
        calls.append(request)

        if len(calls) == 1:
            return invalid

        return valid

    monkeypatch.setattr(
        "clickless_llm.pipeline.extract_constraints",
        fake_extract,
    )

    result = extract_and_validate(
        "Get me a vegan snack under $10",
        max_retries=2,
    )

    assert result == valid

    # One initial attempt + one corrective retry.
    assert len(calls) == 2

    # The retry must preserve the original request.
    assert "Get me a vegan snack under $10" in calls[1]

    # Most importantly, the validation failure must be supplied
    # to the corrective attempt.
    assert "Validation error:" in calls[1]
    assert "high_protein" in calls[1]
