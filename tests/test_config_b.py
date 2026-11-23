"""Tests for ClickLess AI Config B."""

from unittest.mock import Mock, patch

import pytest

from clickless_llm.config_b import (
    OLLAMA_URL,
    select_from_candidates,
)


CANDIDATES = [
    {
        "code": "A1",
        "name": "Regular Granola Bar",
        "brand": "Test",
        "price": 4.00,
        "category": "snack_bars",
        "dietary_tags": [],
        "allergens": ["gluten"],
    },
    {
        "code": "B2",
        "name": "Organic Gluten-Free Bar",
        "brand": "Test",
        "price": 5.00,
        "category": "snack_bars",
        "dietary_tags": ["organic", "gluten_free"],
        "allergens": [],
    },
    {
        "code": "C3",
        "name": "Organic Cookie",
        "brand": "Test",
        "price": 3.00,
        "category": "cookies",
        "dietary_tags": ["organic"],
        "allergens": ["gluten"],
    },
]


def _mock_ollama_response(code):
    """Create a fake Ollama HTTP response."""

    response = Mock()
    response.raise_for_status.return_value = None

    if code is None:
        content = '{"code": null}'
    else:
        content = f'{{"code": "{code}"}}'

    response.json.return_value = {
        "message": {
            "content": content,
        }
    }

    return response


def test_empty_candidates_returns_none():
    result = select_from_candidates(
        "I want a vegan snack",
        [],
    )

    assert result is None


def test_candidates_must_be_list():
    with pytest.raises(TypeError):
        select_from_candidates(
            "I want a vegan snack",
            "not-a-list",
        )


def test_empty_request_rejected():
    with pytest.raises(ValueError):
        select_from_candidates(
            "",
            CANDIDATES,
        )


@patch("clickless_llm.config_b.requests.post")
def test_valid_candidate_selection(mock_post):
    mock_post.return_value = _mock_ollama_response("B2")

    result = select_from_candidates(
        "I want an organic gluten-free snack bar under $10",
        CANDIDATES,
    )

    assert result is CANDIDATES[1]
    assert result["code"] == "B2"

    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args

    assert args[0] == OLLAMA_URL
    assert kwargs["json"]["model"]
    assert kwargs["json"]["stream"] is False


@patch("clickless_llm.config_b.requests.post")
def test_model_can_report_no_match(mock_post):
    mock_post.return_value = _mock_ollama_response(None)

    result = select_from_candidates(
        "I want a product that does not exist",
        CANDIDATES,
    )

    assert result is None


@patch("clickless_llm.config_b.requests.post")
def test_hallucinated_product_is_blocked(mock_post):
    mock_post.return_value = _mock_ollama_response("FAKE999")

    result = select_from_candidates(
        "I want an organic gluten-free snack bar",
        CANDIDATES,
    )

    assert result is None


@patch("clickless_llm.config_b.requests.post")
def test_only_first_twenty_candidates_are_sent(mock_post):
    mock_post.return_value = _mock_ollama_response(None)

    candidates = [
        {
            "code": f"P{i}",
            "name": f"Product {i}",
            "brand": "Test",
            "price": float(i),
            "category": "snacks",
            "dietary_tags": [],
            "allergens": [],
        }
        for i in range(25)
    ]

    select_from_candidates(
        "I want a snack",
        candidates,
    )

    _, kwargs = mock_post.call_args

    messages = kwargs["json"]["messages"]
    user_message = messages[1]["content"]

    # Products P0-P19 should be sent.
    assert '"P19"' in user_message

    # Products outside the top 20 must not reach the LLM.
    assert '"P20"' not in user_message
    assert '"P24"' not in user_message


@pytest.mark.integration
def test_live_ollama_selects_valid_candidate():
    result = select_from_candidates(
        "I want an organic gluten-free snack bar under $10",
        CANDIDATES,
    )

    assert result is not None
    assert result["code"] == "B2"


@pytest.mark.integration
def test_live_ollama_infeasible_candidates_return_none():
    infeasible = [
        CANDIDATES[0],
        CANDIDATES[2],
    ]

    result = select_from_candidates(
        "I want an organic gluten-free snack bar under $10",
        infeasible,
    )

    assert result is None


def _mock_with_quantity(content):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": content}}
    return response


@patch("clickless_llm.config_b.requests.post")
def test_return_quantity_reports_units(mock_post):
    mock_post.return_value = _mock_with_quantity('{"code": "B2", "quantity": 3}')
    product, quantity = select_from_candidates("3 organic gf bars", CANDIDATES, return_quantity=True)
    assert product["code"] == "B2" and quantity == 3


@patch("clickless_llm.config_b.requests.post")
def test_return_quantity_defaults_and_none(mock_post):
    mock_post.return_value = _mock_with_quantity('{"code": null}')
    assert select_from_candidates("x", CANDIDATES, return_quantity=True) == (None, 1)
    mock_post.return_value = _mock_with_quantity('{"code": "A1", "quantity": 0}')
    assert select_from_candidates("x", CANDIDATES, return_quantity=True) == (CANDIDATES[0], 1)
    assert select_from_candidates("x", [], return_quantity=True) == (None, 1)
