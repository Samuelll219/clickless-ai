"""Neo4j integration test: Cypher results must equal the Python reference filter.

Skipped unless NEO4J_URI is set, e.g.
    docker compose up -d
    NEO4J_URI=bolt://localhost:7687 pytest tests/test_neo4j_integration.py -v

WARNING: loads a synthetic catalog, replacing all Product nodes in the target
database. Point it at a scratch database, or re-run scripts/load_kg.py afterwards.
"""
import os
import random
from pathlib import Path

import pytest

from clickless_kg import vocab
from clickless_kg.catalog import filter_products, load_catalog
from clickless_kg.curate import curate
from conftest import synthetic_dump

pytestmark = pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI not set")


@pytest.fixture(scope="module")
def kg_and_catalog():
    from clickless_kg.load import load
    from clickless_kg.retrieval import KGRetriever
    real = Path("data/catalog.json")
    if os.environ.get("USE_REAL_CATALOG") and real.exists():
        catalog = load_catalog(real)
    else:
        catalog, _ = curate(synthetic_dump(), target=400)
    kg = KGRetriever.from_env()
    stats = load(kg.driver, catalog, database=kg._database)
    assert stats["products"] == len(catalog)
    yield kg, catalog
    kg.close()


def test_load_is_idempotent(kg_and_catalog):
    from clickless_kg.load import load
    kg, catalog = kg_and_catalog
    s1 = load(kg.driver, catalog, database=kg._database)
    s2 = load(kg.driver, catalog, database=kg._database)
    assert s1 == s2


def test_cypher_matches_reference_on_random_constraints(kg_and_catalog):
    kg, catalog = kg_and_catalog
    rng = random.Random(7)
    cats = list(vocab.CATEGORY_NAMES) + list(vocab.CATEGORY_GROUPS)
    for _ in range(300):
        cat = rng.choice(cats)
        tags = rng.sample(vocab.DIETARY_TAG_NAMES, rng.randint(0, 2))
        allergens = rng.sample(vocab.ALLERGEN_NAMES, rng.randint(0, 2))
        max_price = rng.choice([None, 3, 5, 8, 15])
        want = filter_products(catalog, cat, max_price, tags, allergens, limit=5)
        got = kg.search(cat, max_price, tags, allergens, limit=5)
        assert [r["code"] for r in got] == [p["code"] for p in want], (cat, tags, allergens, max_price)
        for r, p in zip(got, want):
            assert r["dietary_tags"] == sorted(p["dietary_tags"])
            assert r["allergens"] == sorted(p["allergens"])


def test_vocabulary_round_trip(kg_and_catalog):
    kg, _ = kg_and_catalog
    v = kg.vocabulary()
    assert set(v["categories"]) == set(vocab.CATEGORY_NAMES)
    assert set(v["groups"]) == set(vocab.CATEGORY_GROUPS)
    assert set(v["tags"]) == set(vocab.DIETARY_TAG_NAMES)
    assert set(v["allergens"]) == set(vocab.ALLERGEN_NAMES)


def test_cheapest_in_category(kg_and_catalog):
    kg, catalog = kg_and_catalog
    best = kg.cheapest_in_category("chips", ["vegan"], ["soy"])
    ref = filter_products(catalog, "chips", None, ["vegan"], ["soy"], limit=1)
    assert (best and best["code"]) == (ref[0]["code"] if ref else None)
