"""Tests for the TF-IDF baseline (Config A) and the Config B candidate feed."""
import pytest

from clickless_kg.catalog import load_catalog
from clickless_mining.tfidf import (
    CONFIG_B_DEPTH, TfidfIndex, build_document, config_a, tfidf_candidates,
)


def P(code, name, cat, tags=(), ingredients="water", labels=(), brand="B", price=3.0,
      allergens=()):
    return {"code": code, "name": name, "brand": brand, "category": cat, "price": price,
            "dietary_tags": list(tags), "allergens": list(allergens),
            "ingredients_text": ingredients, "labels_raw": list(labels)}


CATALOG = [
    P("001", "Sea Salt Tortilla Chips", "chips", ["gluten_free"], "corn, oil, salt"),
    P("002", "Dark Chocolate Bar", "chocolate_candy", ["vegan"], "cocoa, sugar"),
    P("003", "Oat Milk Original", "plant_milk", ["vegan"], "water, oats", ["en:no-gmos"]),
    P("004", "Whole Milk", "milk", [], "milk", allergens=["milk"]),
    P("005", "Peanut Butter Cookies", "cookies", [], "peanuts, flour, sugar"),
]


@pytest.fixture(scope="module")
def index():
    return TfidfIndex(CATALOG)


def test_document_uses_name_category_ingredients_and_labels_only():
    doc = build_document(P("9", "Oat Milk", "plant_milk", ["non_gmo"], "oats",
                           ["en:no-gmos"], price=4.99, allergens=["tree_nuts"]))
    assert "Oat Milk" in doc and "oats" in doc
    assert "plant milk" in doc               # leaf id with "_" -> " "
    assert "beverages" in doc            # group name, so "beverages" queries can match
    assert "non gmo" in doc              # underscore removed from tags
    assert "no gmos" in doc              # raw OFF label cleaned
    assert "4.99" not in doc and "tree_nuts" not in doc   # price/allergens not indexed


def test_search_ranks_by_cosine_and_is_deterministic(index):
    ranked = index.search("tortilla chips", k=None)
    assert ranked[0][0]["code"] == "001"
    assert [p["code"] for p, _ in ranked] == [p["code"] for p, _ in index.search("tortilla chips", k=None)]
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 <= scores[-1] <= scores[0] <= 1.0 + 1e-9


def test_ties_broken_by_barcode(index):
    ranked = index.search("zzzz unknown words", k=None)       # every score is 0
    assert [p["code"] for p, _ in ranked] == sorted(p["code"] for p in CATALOG)


def test_group_word_reaches_leaf_products(index):
    assert config_a("something for snacks", index)["category"] in {"chips", "chocolate_candy", "cookies"}


def test_config_a_always_returns_a_product(index):
    # Pure lexical baseline: even an infeasible request yields its top product.
    assert config_a("vegan whole milk under $1", index)["code"] in {p["code"] for p in CATALOG}


def test_negation_is_not_understood(index):
    # Documented baseline failure: "no peanuts" adds the term "peanuts".
    assert config_a("cookies with no peanuts", index)["code"] == "005"


def test_candidates_are_catalog_objects_with_config_b_fields(index):
    cands = tfidf_candidates("vegan milk", index, k=3)
    assert len(cands) == 3
    assert all(any(c is p for p in CATALOG) for c in cands)     # identity preserved
    for c in cands:
        assert {"code", "name", "brand", "price", "category", "dietary_tags",
                "allergens", "ingredients_text"} <= set(c)


def test_rank_of_first(index):
    assert index.rank_of_first("tortilla chips", ["001"]) == 1
    assert index.rank_of_first("tortilla chips", []) is None
    assert index.rank_of_first("tortilla chips", ["404"]) is None


def test_empty_query_rejected(index):
    with pytest.raises(ValueError):
        index.search("   ")


def test_real_catalog_index():
    catalog = load_catalog()
    index = TfidfIndex(catalog)
    assert index.matrix.shape[0] == len(catalog) == 400
    cands = tfidf_candidates("organic gluten free crackers", index)
    assert len(cands) == CONFIG_B_DEPTH
    assert cands[0]["category"] == "crackers"
