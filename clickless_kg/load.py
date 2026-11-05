"""Populate Neo4j from data/catalog.json.

The load is idempotent: re-running it brings the graph exactly in line with
the catalog (products removed from the catalog are deleted, stale tag /
allergen edges are replaced).
"""
from __future__ import annotations

from pathlib import Path

from . import vocab
from .catalog import validate_catalog

SCHEMA_FILE = Path(__file__).with_name("schema.cypher")

UPSERT_VOCAB = """
UNWIND $groups AS g
  MERGE (c:Category {name: g}) SET c.is_group = true, c.display = g
WITH count(*) AS step1
UNWIND $categories AS row
  MERGE (c:Category {name: row.name})
  SET c.display = row.display, c.group = row.group, c.is_group = false,
      c.price_min = row.price_range[0], c.price_max = row.price_range[1]
  WITH c, row
  MATCH (g:Category {name: row.group})
  MERGE (c)-[:SUBCATEGORY_OF]->(g)
WITH count(*) AS step2
UNWIND $tags AS t MERGE (:DietaryTag {name: t})
WITH count(*) AS step3
UNWIND $allergens AS a MERGE (:Allergen {name: a})
"""

DELETE_STALE_PRODUCTS = "MATCH (p:Product) WHERE NOT p.code IN $codes DETACH DELETE p"

CLEAR_PRODUCT_EDGES = """
UNWIND $codes AS code
MATCH (p:Product {code: code})-[r:BELONGS_TO|HAS_TAG|CONTAINS]->()
DELETE r
"""

UPSERT_PRODUCTS = """
UNWIND $rows AS row
MERGE (p:Product {code: row.code})
SET p.name = row.name, p.brand = row.brand, p.price = toFloat(row.price),
    p.ingredients_text = row.ingredients_text, p.traces = row.traces
WITH p, row
MATCH (c:Category {name: row.category})
MERGE (p)-[:BELONGS_TO]->(c)
FOREACH (t IN row.dietary_tags |
  MERGE (d:DietaryTag {name: t}) MERGE (p)-[:HAS_TAG]->(d))
FOREACH (a IN row.allergens |
  MERGE (x:Allergen {name: a}) MERGE (p)-[:CONTAINS]->(x))
"""

STATS = """
MATCH (p:Product)
RETURN count(p) AS products,
       sum(COUNT { (p)-[:BELONGS_TO]->() }) AS belongs_to,
       sum(COUNT { (p)-[:HAS_TAG]->() }) AS has_tag,
       sum(COUNT { (p)-[:CONTAINS]->() }) AS contains
"""


def schema_statements() -> list[str]:
    lines = [l for l in SCHEMA_FILE.read_text(encoding="utf-8").splitlines() if not l.strip().startswith("//")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def load(driver, catalog: list[dict], database: str | None = None, batch_size: int = 200) -> dict:
    problems = validate_catalog(catalog)
    if problems:
        raise ValueError("catalog failed validation:\n  " + "\n  ".join(problems[:20]))
    v = vocab.vocabulary()
    codes = [p["code"] for p in catalog]
    with driver.session(database=database) as s:
        for stmt in schema_statements():
            s.run(stmt).consume()
        s.run(UPSERT_VOCAB, groups=v["category_groups"], categories=v["categories"],
              tags=v["dietary_tags"], allergens=v["allergens"]).consume()
        s.run(DELETE_STALE_PRODUCTS, codes=codes).consume()
        s.run(CLEAR_PRODUCT_EDGES, codes=codes).consume()
        for i in range(0, len(catalog), batch_size):
            rows = [{k: p.get(k) for k in ("code", "name", "brand", "price", "category",
                                            "dietary_tags", "allergens", "traces",
                                            "ingredients_text")}
                    for p in catalog[i:i + batch_size]]
            s.run(UPSERT_PRODUCTS, rows=rows).consume()
        rec = s.run(STATS).single()
    return dict(rec)
