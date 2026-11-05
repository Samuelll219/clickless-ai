"""Parameterized Cypher retrieval: the only way the pipeline reads products.

The LLM never writes Cypher. The validated constraint JSON
    {"category": ..., "dietary_tags": [...], "excluded_allergens": [...],
     "max_price": float | null, "quantity": int}
fills the parameters of RETRIEVAL_QUERY. An empty result means "no matching
product" and must be reported to the user as such; constraints are never
relaxed silently.

Usage (pipeline / Config C):
    from clickless_kg.retrieval import KGRetriever
    with KGRetriever.from_env() as kg:
        products = kg.find_products(constraints)       # top 5 by price
        addon = kg.cheapest_in_category("chips", ["gluten_free"], ["milk"])
"""
from __future__ import annotations

import os

from . import vocab

# Same template as the proposal (Section IV-B) with three small, documented
# extensions: a NULL max_price means "no price limit", the requested category
# may be a group ("snacks") that expands to its leaf categories, and ties on
# price are broken by barcode so results are deterministic.
RETRIEVAL_QUERY = """
MATCH (p:Product)-[:BELONGS_TO]->(c:Category)-[:SUBCATEGORY_OF*0..1]->(:Category {name: $category})
WHERE ($max_price IS NULL OR p.price <= $max_price)
  AND ALL(t IN $tags WHERE EXISTS { (p)-[:HAS_TAG]->(:DietaryTag {name: t}) })
  AND NONE(a IN $allergens WHERE EXISTS { (p)-[:CONTAINS]->(:Allergen {name: a}) })
RETURN p.code AS code, p.name AS name, p.brand AS brand, p.price AS price,
       c.name AS category,
       [(p)-[:HAS_TAG]->(dt:DietaryTag) | dt.name] AS dietary_tags,
       [(p)-[:CONTAINS]->(al:Allergen) | al.name] AS allergens
ORDER BY p.price ASC, p.code ASC
LIMIT $limit
"""

VOCAB_QUERY = """
MATCH (c:Category) WITH collect(c) AS cs
RETURN [c IN cs WHERE NOT coalesce(c.is_group, false) | c.name] AS categories,
       [c IN cs WHERE c.is_group | c.name] AS groups,
       COLLECT { MATCH (t:DietaryTag) RETURN t.name } AS tags,
       COLLECT { MATCH (a:Allergen) RETURN a.name } AS allergens
"""


def _params(category: str, max_price, dietary_tags, excluded_allergens, limit: int) -> dict:
    """Check values against the controlled vocabulary before they reach Cypher.
    (The validator should already have done this; this is a second guard.)"""
    if category not in vocab.CATEGORY_BY_NAME and category not in vocab.CATEGORY_GROUPS:
        raise ValueError(f"unknown category: {category!r}")
    bad = [t for t in dietary_tags if t not in vocab.DIETARY_TAGS]
    if bad:
        raise ValueError(f"unknown dietary tags: {bad}")
    bad = [a for a in excluded_allergens if a not in vocab.ALLERGENS]
    if bad:
        raise ValueError(f"unknown allergens: {bad}")
    if max_price is not None and float(max_price) < 0:
        raise ValueError("max_price must be >= 0")
    return {"category": category,
            "max_price": None if max_price is None else float(max_price),
            "tags": list(dict.fromkeys(dietary_tags)),
            "allergens": list(dict.fromkeys(excluded_allergens)),
            "limit": int(limit)}


class KGRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    @classmethod
    def from_env(cls) -> "KGRetriever":
        return cls(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                   os.environ.get("NEO4J_USER", "neo4j"),
                   os.environ.get("NEO4J_PASSWORD", "clickless"),
                   os.environ.get("NEO4J_DATABASE") or None)

    @property
    def driver(self):
        return self._driver

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _run(self, query: str, **params) -> list[dict]:
        records, _, _ = self._driver.execute_query(query, params, database_=self._database)
        return [r.data() for r in records]

    def search(self, category: str, max_price: float | None = None, dietary_tags=(),
               excluded_allergens=(), limit: int = 5) -> list[dict]:
        rows = self._run(RETRIEVAL_QUERY,
                         **_params(category, max_price, dietary_tags, excluded_allergens, limit))
        for r in rows:
            r["dietary_tags"] = sorted(r["dietary_tags"])
            r["allergens"] = sorted(r["allergens"])
        return rows

    def find_products(self, constraints: dict, limit: int = 5) -> list[dict]:
        """Run retrieval directly on the validated constraint JSON."""
        return self.search(constraints["category"], constraints.get("max_price"),
                           constraints.get("dietary_tags") or [],
                           constraints.get("excluded_allergens") or [], limit)

    def cheapest_in_category(self, category: str, dietary_tags=(),
                             excluded_allergens=()) -> dict | None:
        """For the Apriori add-on step: cheapest product in a suggested category
        that still respects the user's dietary constraints."""
        rows = self.search(category, None, dietary_tags, excluded_allergens, limit=1)
        return rows[0] if rows else None

    def vocabulary(self) -> dict:
        """Controlled vocabularies as stored in the graph (for the validator)."""
        return self._run(VOCAB_QUERY)[0]
