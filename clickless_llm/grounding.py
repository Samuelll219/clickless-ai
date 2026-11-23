"""Deterministic grounding guard applied after LLM extraction.

A 7B model occasionally adds constraints the user never stated (e.g. "vegan",
"non_gmo" for "oat milk") or confuses dairy and plant milk. Prompt rules alone
did not fix this reliably, so the pipeline removes any dietary tag whose
trigger words are absent from the request and fixes the milk/plant-milk split
from the request words. The guard only ever REMOVES unstated tags or picks
between milk and plant_milk; it never adds a constraint.
"""
from __future__ import annotations

import re

TAG_TRIGGERS = {
    "gluten_free": [r"gluten[\s-]*free", r"\bno gluten\b", r"without gluten", r"\bgf\b", r"celiac"],
    "organic": [r"organic"],
    "vegan": [r"vegan"],
    "vegetarian": [r"vegetarian", r"\bveggie\b"],
    "non_gmo": [r"non[\s-]*gmo", r"gmo[\s-]*free", r"no gmos?"],
    "lactose_free": [r"lactose[\s-]*free", r"no lactose", r"without lactose", r"lactose intoleran"],
    "kosher": [r"kosher"],
    "no_added_sugar": [r"no[\s-]*(added[\s-]*)?sugar[\s-]*added", r"no[\s-]*added[\s-]*sugar",
                       r"without added sugar", r"unsweetened", r"sugar[\s-]*free"],
    "fair_trade": [r"fair[\s-]*trade"],
}
PLANT_MILK_WORDS = [r"plant[\s-]*(based\s*)?milk", r"non[\s-]*dairy", r"dairy[\s-]*free milk",
                    r"milk alternative", r"\b(oat|almond|soy|rice|coconut|cashew|hemp|pea)[\s-]*milk"]


def _mentions(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


def ground_constraints(request: str, constraints: dict) -> dict:
    """Return a copy of `constraints` with unstated tags removed and milk type fixed."""
    text = request.lower()
    out = dict(constraints)
    out["dietary_tags"] = [t for t in constraints.get("dietary_tags", [])
                           if t not in TAG_TRIGGERS or _mentions(text, TAG_TRIGGERS[t])]
    cat = constraints.get("category")
    if cat in ("milk", "plant_milk") and re.search(r"milk", text):
        out["category"] = "plant_milk" if _mentions(text, PLANT_MILK_WORDS) else "milk"
    return out
