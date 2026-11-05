#!/usr/bin/env python3
"""Step 2: load data/catalog.json into Neo4j (idempotent).

    docker compose up -d                 # starts Neo4j 5 on bolt://localhost:7687
    python scripts/load_kg.py

Connection settings come from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD /
NEO4J_DATABASE (defaults match docker-compose.yml).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import load_catalog  # noqa: E402
from clickless_kg.load import load  # noqa: E402
from clickless_kg.retrieval import KGRetriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog.json")
    args = ap.parse_args()
    catalog = load_catalog(args.catalog)
    with KGRetriever.from_env() as kg:
        stats = load(kg.driver, catalog, database=kg._database)
        print("loaded:", stats)
        v = kg.vocabulary()
        print(f"vocab in graph: {len(v['categories'])} categories, {len(v['groups'])} groups, "
              f"{len(v['tags'])} dietary tags, {len(v['allergens'])} allergens")


if __name__ == "__main__":
    main()
