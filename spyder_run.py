# -*- coding: utf-8 -*-
"""
ClickLess AI - Knowledge Graph: run everything from Spyder.

How to use:
  1. In Spyder, open this file (File > Open), it must stay in the clickless-kg folder.
  2. Run ONE cell at a time with Ctrl+Enter (each cell starts with a "# %%" line).
     Shift+Enter runs the cell and jumps to the next one.
  3. Fill in the Neo4j settings in the CONFIG cell first.
"""

# %% CONFIG  -------------------------------------------------------------------
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
os.chdir(PROJECT)                      # all paths (data/...) are relative to the project folder
sys.path.insert(0, str(PROJECT))

# Neo4j connection.
#   Neo4j Desktop (local):  bolt://localhost:7687, user neo4j, the password you chose
#   Neo4j AuraDB Free:      neo4j+s://xxxxxxxx.databases.neo4j.io, user neo4j, password from Aura
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "clickless"
os.environ["NEO4J_DATABASE"] = ""      # leave empty for the default database

print("project folder:", PROJECT)


# %% STEP 0 - install packages (only needed once) -----------------------------
# Installs into the same environment Spyder is running in.
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "neo4j", "requests"])


# %% STEP 1 - download Open Food Facts + build the catalog (~10 min first time) -
# Downloads are cached in data/raw/pages/, so if it stops you can just re-run this cell.
from clickless_kg.fetch_off import fetch_all

raw = PROJECT / "data" / "raw" / "off_raw.jsonl"
raw.parent.mkdir(parents=True, exist_ok=True)
n = fetch_all(raw, pages=2, country="en:united-states")
print(f"fetched {n} unique products -> {raw}")


# %% STEP 2 - curate 400 products, assign prices, write data/ files -------------
sys.argv = ["build_catalog.py"]        # re-curate from data/raw/off_raw.jsonl
exec(open(PROJECT / "scripts" / "build_catalog.py", encoding="utf-8").read(),
     {"__name__": "__main__", "__file__": str(PROJECT / "scripts" / "build_catalog.py")})

import json
report = json.loads((PROJECT / "data" / "curation_report.json").read_text(encoding="utf-8"))
print("\neligible per category:", report["eligible_per_category"])
print("\nrejections:", report["rejections"])
print("tags:", report["dietary_tag_counts"])
print("allergens:", report["allergen_counts"])


# %% STEP 3 - load into Neo4j (Neo4j must be running) ---------------------------
from clickless_kg.catalog import load_catalog
from clickless_kg.load import load
from clickless_kg.retrieval import KGRetriever

catalog = load_catalog(PROJECT / "data" / "catalog.json")
with KGRetriever.from_env() as kg:
    print("loaded:", load(kg.driver, catalog, database=kg._database))
    print("vocab in graph:", {k: len(v) for k, v in kg.vocabulary().items()})


# %% STEP 4 - try some queries --------------------------------------------------
from clickless_kg.catalog import filter_products

def show(rows):
    if not rows:
        print("  no matching product")
    for r in rows:
        print(f"  ${r['price']:>6.2f}  {r['category']:16s} {r['name'][:40]:40s} "
              f"tags={','.join(r['dietary_tags'])}  allergens={','.join(r['allergens']) or '-'}")

queries = [
    dict(category="snacks", max_price=15, dietary_tags=["gluten_free", "organic"], excluded_allergens=[]),
    dict(category="chips", max_price=None, dietary_tags=["vegan"], excluded_allergens=["soy"]),
    dict(category="yogurt", max_price=None, dietary_tags=["vegan"], excluded_allergens=["milk"]),
]
with KGRetriever.from_env() as kg:
    for q in queries:
        print(q)
        neo = kg.find_products(q)
        ref = filter_products(catalog, q["category"], q["max_price"],
                              q["dietary_tags"], q["excluded_allergens"])
        show(neo)
        print("  same as Python reference:", [r["code"] for r in neo] == [p["code"] for p in ref])


# %% STEP 5 - unit tests (no Neo4j needed) --------------------------------------
import pytest
pytest.main(["-q", "-p", "no:cacheprovider", str(PROJECT / "tests"),
             "--ignore", str(PROJECT / "tests" / "test_neo4j_integration.py")])


# %% STEP 6 - Neo4j integration test (Cypher vs Python, 300 random queries) ------
# Uses YOUR real catalog, so the database stays as loaded in STEP 3.
import pytest
os.environ["USE_REAL_CATALOG"] = "1"
pytest.main(["-q", "-p", "no:cacheprovider", str(PROJECT / "tests" / "test_neo4j_integration.py")])
