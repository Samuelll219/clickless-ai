import json

import pytest

from clickless_kg import vocab
from clickless_kg.catalog import acceptable_set, filter_products
from clickless_kg.curate import curate
from clickless_kg.load import schema_statements
from clickless_kg.retrieval import RETRIEVAL_QUERY, _params


def P(code, cat, price, tags=(), allergens=()):
    return {"code": code, "name": code, "brand": "", "category": cat, "price": price,
            "dietary_tags": list(tags), "allergens": list(allergens), "ingredients_text": "x"}


CAT = [
    P("a", "chips", 3.0, ["gluten_free", "organic"]),
    P("b", "chips", 2.5, ["gluten_free"], ["milk"]),
    P("c", "crackers", 4.0, ["gluten_free", "organic"], ["sesame"]),
    P("d", "yogurt", 2.0, ["organic"], ["milk"]),
    P("e", "chips", 3.0, ["gluten_free", "organic"]),
]


def codes(rows):
    return [r["code"] for r in rows]


def test_filter_semantics():
    assert codes(filter_products(CAT, "chips")) == ["b", "a", "e"]  # price asc, code tiebreak
    assert codes(filter_products(CAT, "chips", 2.9)) == ["b"]
    assert codes(filter_products(CAT, "chips", dietary_tags=["organic"])) == ["a", "e"]
    assert codes(filter_products(CAT, "chips", excluded_allergens=["milk"])) == ["a", "e"]
    assert codes(filter_products(CAT, "chips", limit=1)) == ["b"]


def test_group_category_expands():
    assert codes(filter_products(CAT, "snacks", 15, ["gluten_free", "organic"])) == ["a", "e", "c"]
    assert codes(filter_products(CAT, "snacks", 15, ["gluten_free", "organic"], ["sesame"])) == ["a", "e"]


def test_infeasible_is_empty():
    assert acceptable_set(CAT, {"category": "yogurt", "dietary_tags": [],
                                "excluded_allergens": ["milk"], "max_price": None}) == []


def test_params_guard():
    with pytest.raises(ValueError):
        _params("frozen_pizza", None, [], [], 5)
    with pytest.raises(ValueError):
        _params("chips", None, ["keto"], [], 5)
    with pytest.raises(ValueError):
        _params("chips", None, [], ["nuts"], 5)
    p = _params("snacks", 15, ["vegan", "vegan"], [], 5)
    assert p == {"category": "snacks", "max_price": 15.0, "tags": ["vegan"],
                 "allergens": [], "limit": 5}


def test_query_uses_only_parameters():
    for name in ("$category", "$max_price", "$tags", "$allergens", "$limit"):
        assert name in RETRIEVAL_QUERY
    assert "format(" not in RETRIEVAL_QUERY and "%s" not in RETRIEVAL_QUERY


def test_schema_statements_parse():
    stmts = schema_statements()
    assert len(stmts) == 5 and all(s.startswith("CREATE") for s in stmts)


def test_constraint_schema_matches_vocab():
    s = vocab.constraint_json_schema()
    assert set(s["required"]) == {"category", "dietary_tags", "excluded_allergens",
                                  "max_price", "quantity"}
    assert set(s["properties"]["dietary_tags"]["items"]["enum"]) == set(vocab.DIETARY_TAGS)
    assert "snacks" in s["properties"]["category"]["enum"]
    json.dumps(s)


def test_every_category_has_feasible_queries(dump):
    catalog, _ = curate(dump, target=400)
    for c in vocab.CATEGORY_NAMES:
        assert filter_products(catalog, c), c
