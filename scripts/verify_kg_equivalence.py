"""Check that Neo4j (Cypher template) and OfflineKG return identical products.

For every test prompt: (1) the gold acceptable set (all matches, no limit) and
(2) Config C's top-5 for the constraints the LLM actually extracted (read from
the benchmark cache, so no LLM calls). Needs Neo4j running with the catalog
loaded (docker compose up -d; python scripts/load_kg.py).

    python scripts/verify_kg_equivalence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import load_catalog  # noqa: E402
from clickless_kg.retrieval import KGRetriever  # noqa: E402
from clickless_mining.addons import OfflineKG  # noqa: E402


def main() -> None:
    rows = json.loads(Path("data/evaluation/test_prompts.json").read_text(encoding="utf-8"))
    cache = json.loads(Path("data/evaluation/benchmark_raw.json").read_text(encoding="utf-8"))
    offline = OfflineKG(load_catalog())
    gold_ok = c_ok = c_n = 0
    mismatches = []
    with KGRetriever.from_env() as kg:
        for r in rows:
            neo = [p["code"] for p in kg.find_products(r["expected"], limit=10_000)]
            if neo == r["acceptable"]:
                gold_ok += 1
            else:
                mismatches.append(("gold", r["id"]))
            for key, raw in cache.items():
                if key.endswith("|" + r["prompt"]) and "|C|" in "|" + key and raw.get("constraints"):
                    c_n += 1
                    a = [p["code"] for p in kg.find_products(raw["constraints"], limit=5)]
                    b = [p["code"] for p in offline.find_products(raw["constraints"], limit=5)]
                    if a == b:
                        c_ok += 1
                    else:
                        mismatches.append(("configC", r["id"], key.split("|")[0]))
    print(f"gold acceptable sets identical: {gold_ok}/{len(rows)}")
    print(f"Config C top-5 identical (Neo4j vs OfflineKG): {c_ok}/{c_n}")
    if mismatches:
        print("MISMATCHES:", mismatches)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
