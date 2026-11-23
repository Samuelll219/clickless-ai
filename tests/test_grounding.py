"""Tests for the post-extraction grounding guard."""
import pytest

from clickless_llm.grounding import ground_constraints


def C(cat, tags=(), excl=()):
    return {"category": cat, "dietary_tags": list(tags), "excluded_allergens": list(excl),
            "max_price": None, "quantity": 1}


@pytest.mark.parametrize("request_text,given,expected_tags", [
    ("2 oat milks without tree nuts under $6", ["gluten_free", "vegan", "non_gmo"], []),
    ("organic gluten-free crackers", ["organic", "gluten_free"], ["organic", "gluten_free"]),
    ("non-GMO cereal with no sugar added", ["non_gmo", "no_added_sugar"], ["non_gmo", "no_added_sugar"]),
    ("veggie cookies", ["vegetarian"], ["vegetarian"]),
    ("fair trade coffee", ["fair_trade", "organic"], ["fair_trade"]),
])
def test_unstated_tags_removed(request_text, given, expected_tags):
    assert ground_constraints(request_text, C("snacks", given))["dietary_tags"] == expected_tags


@pytest.mark.parametrize("request_text,given,expected", [
    ("organic lactose-free milk", "plant_milk", "milk"),
    ("I need a plant-based milk free from tree nuts", "milk", "plant_milk"),
    ("2 oat milks", "milk", "plant_milk"),
    ("whole milk please", "milk", "milk"),
    ("cheese", "cheese", "cheese"),
])
def test_milk_type(request_text, given, expected):
    assert ground_constraints(request_text, C(given))["category"] == expected


def test_guard_never_adds_constraints():
    out = ground_constraints("vegan organic soup", C("soup", ["vegan"], ["celery"]))
    assert out["dietary_tags"] == ["vegan"] and out["excluded_allergens"] == ["celery"]
