#!/usr/bin/env python3
"""Try the retrieval template from the command line.

    python scripts/query_kg.py snacks --tags gluten_free organic --max-price 15
    python scripts/query_kg.py yogurt --exclude milk                 # -> no matching product
    python scripts/query_kg.py chips --tags vegan --offline          # use catalog.json, no Neo4j

--offline runs the pure-Python reference filter on data/catalog.json, which
must return the same products as Neo4j.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import filter_products, load_catalog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("category")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[], help="excluded allergens")
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--catalog", default="data/catalog.json")
    a = ap.parse_args()

    if a.offline:
        rows = filter_products(load_catalog(a.catalog), a.category, a.max_price,
                               a.tags, a.exclude, a.limit)
    else:
        from clickless_kg.retrieval import KGRetriever
        with KGRetriever.from_env() as kg:
            rows = kg.search(a.category, a.max_price, a.tags, a.exclude, a.limit)

    if not rows:
        print("no matching product")
        return
    for r in rows:
        print(f"${r['price']:>6.2f}  {r['category']:16s} {r['name'][:45]:45s} "
              f"tags={','.join(r['dietary_tags'])}  allergens={','.join(r['allergens']) or '-'}")


if __name__ == "__main__":
    main()
