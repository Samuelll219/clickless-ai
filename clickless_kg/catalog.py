"""Reading, validating and filtering the curated catalog (data/catalog.json).

`filter_products` is a pure-Python implementation of exactly the same
semantics as the Cypher retrieval template. It is used to
  * compute the ground-truth "acceptable product set" for each test prompt
    (Vikas's labels) and to check whether a prompt is infeasible, and
  * cross-check Neo4j results in tests/test_neo4j_integration.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import vocab

REQUIRED_FIELDS = ("code", "name", "brand", "category", "price", "dietary_tags",
                   "allergens", "ingredients_text")


def load_catalog(path: str | Path = "data/catalog.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_catalog(catalog: list[dict]) -> list[str]:
    """Return a list of problems (empty list = catalog is valid)."""
    problems: list[str] = []
    codes = set()
    for p in catalog:
        code = p.get("code", "?")
        for f in REQUIRED_FIELDS:
            if f not in p:
                problems.append(f"{code}: missing field {f}")
        if code in codes:
            problems.append(f"{code}: duplicate code")
        codes.add(code)
        if p.get("category") not in vocab.CATEGORY_BY_NAME:
            problems.append(f"{code}: unknown category {p.get('category')!r}")
        for t in p.get("dietary_tags", []):
            if t not in vocab.DIETARY_TAGS:
                problems.append(f"{code}: unknown dietary tag {t!r}")
        for a in p.get("allergens", []):
            if a not in vocab.ALLERGENS:
                problems.append(f"{code}: unknown allergen {a!r}")
        price = p.get("price")
        if not isinstance(price, (int, float)) or price <= 0:
            problems.append(f"{code}: bad price {price!r}")
    return problems


def filter_products(catalog: list[dict], category: str, max_price: float | None = None,
                    dietary_tags=(), excluded_allergens=(), limit: int | None = 5) -> list[dict]:
    """Same semantics and ordering as retrieval.RETRIEVAL_QUERY."""
    leaves = set(vocab.categories_for_request(category))
    if not leaves:
        raise ValueError(f"unknown category: {category!r}")
    tags, allergens = set(dietary_tags), set(excluded_allergens)
    hits = [p for p in catalog
            if p["category"] in leaves
            and (max_price is None or p["price"] <= max_price)
            and tags <= set(p["dietary_tags"])
            and not (allergens & set(p["allergens"]))]
    hits.sort(key=lambda p: (p["price"], p["code"]))
    return hits if limit is None else hits[:limit]


def acceptable_set(catalog: list[dict], constraints: dict) -> list[str]:
    """All product codes satisfying a constraint JSON (no limit). Empty = infeasible."""
    return [p["code"] for p in filter_products(
        catalog, constraints["category"], constraints.get("max_price"),
        constraints.get("dietary_tags", []), constraints.get("excluded_allergens", []),
        limit=None)]
