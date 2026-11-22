"""Tests for ClickLess AI Config C."""

from unittest.mock import Mock, patch

import pytest

from clickless_llm.config_c import run_config_c


VALID_CONSTRAINTS = {
    "category": "snack_bars",
    "dietary_tags": ["organic", "gluten_free"],
    "excluded_allergens": [],
    "max_price": 10,
    "quantity": 2,
}

MATCHING_PRODUCTS = [
    {
        "code": "B2",
        "name": "Organic Gluten-Free Bar",
        "brand": "Test",
        "price": 5.00,
        "category": "snack_bars",
        "dietary_tags": ["organic", "gluten_free"],
        "allergens": [],
    }
]


@patch("clickless_llm.config_c.extract_and_validate")
def test_matched_result(mock_extract):
    mock_extract.return_value = VALID_CONSTRAINTS.copy()

    kg = Mock()
    kg.find_products.return_value = MATCHING_PRODUCTS

    result = run_config_c(
        "Get me 2 organic gluten-free snack bars under $10",
        kg=kg,
    )

    assert result["status"] == "matched"
    assert result["products"] == MATCHING_PRODUCTS
    assert result["constraints"] == VALID_CONSTRAINTS
    assert result["quantity"] == 2

    kg.find_products.assert_called_once_with(
        VALID_CONSTRAINTS,
        limit=5,
    )


@patch("clickless_llm.config_c.extract_and_validate")
def test_no_match_is_explicit(mock_extract):
    mock_extract.return_value = VALID_CONSTRAINTS.copy()

    kg = Mock()
    kg.find_products.return_value = []

    result = run_config_c(
        "Get me 2 organic gluten-free snack bars under $10",
        kg=kg,
    )

    assert result["status"] == "no_match"
    assert result["products"] == []

    # Config C must preserve the original constraints.
    assert result["constraints"] == VALID_CONSTRAINTS


@patch("clickless_llm.config_c.extract_and_validate")
def test_quantity_is_preserved(mock_extract):
    constraints = VALID_CONSTRAINTS.copy()
    constraints["quantity"] = 4

    mock_extract.return_value = constraints

    kg = Mock()
    kg.find_products.return_value = MATCHING_PRODUCTS

    result = run_config_c(
        "Get me 4 organic gluten-free snack bars under $10",
        kg=kg,
    )

    assert result["quantity"] == 4
    assert result["constraints"]["quantity"] == 4


@patch("clickless_llm.config_c.extract_and_validate")
def test_custom_limit_is_passed_to_kg(mock_extract):
    mock_extract.return_value = VALID_CONSTRAINTS.copy()

    kg = Mock()
    kg.find_products.return_value = MATCHING_PRODUCTS

    run_config_c(
        "Get me an organic gluten-free snack bar",
        limit=3,
        kg=kg,
    )

    kg.find_products.assert_called_once_with(
        VALID_CONSTRAINTS,
        limit=3,
    )


def test_empty_request_rejected():
    with pytest.raises(ValueError):
        run_config_c("")


@pytest.mark.parametrize(
    "bad_limit",
    [
        0,
        -1,
        1.5,
        True,
        "5",
    ],
)
def test_invalid_limit_rejected(bad_limit):
    with pytest.raises(ValueError):
        run_config_c(
            "Get me a snack",
            limit=bad_limit,
        )


@patch("clickless_llm.config_c.KGRetriever.from_env")
@patch("clickless_llm.config_c.extract_and_validate")
def test_owned_kg_connection_is_closed(
    mock_extract,
    mock_from_env,
):
    mock_extract.return_value = VALID_CONSTRAINTS.copy()

    kg = Mock()
    kg.find_products.return_value = MATCHING_PRODUCTS
    mock_from_env.return_value = kg

    run_config_c(
        "Get me an organic gluten-free snack bar",
    )

    mock_from_env.assert_called_once()
    kg.close.assert_called_once()


@patch("clickless_llm.config_c.extract_and_validate")
def test_supplied_kg_connection_is_not_closed(mock_extract):
    mock_extract.return_value = VALID_CONSTRAINTS.copy()

    kg = Mock()
    kg.find_products.return_value = MATCHING_PRODUCTS

    run_config_c(
        "Get me an organic gluten-free snack bar",
        kg=kg,
    )

    kg.close.assert_not_called()


@pytest.mark.integration
def test_live_config_c_qwen_and_neo4j():
    """Real end-to-end test: prompt -> Qwen -> validator -> Neo4j."""

    result = run_config_c(
        "Get me an organic gluten-free snack under $15 "
        "without peanuts",
        limit=5,
    )

    assert result["status"] == "matched"
    assert result["products"]

    constraints = result["constraints"]

    assert constraints["category"] == "snacks"
    assert set(constraints["dietary_tags"]) == {
        "organic",
        "gluten_free",
    }
    assert constraints["excluded_allergens"] == [
        "peanuts"
    ]
    assert constraints["max_price"] == 15
    assert constraints["quantity"] == 1

    # Every returned product must respect the extracted constraints.
    for product in result["products"]:
        assert product["price"] <= 15
        assert "organic" in product["dietary_tags"]
        assert "gluten_free" in product["dietary_tags"]
        assert "peanuts" not in product["allergens"]
