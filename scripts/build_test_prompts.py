"""Build the 80-prompt benchmark test set (proposal Sec. V.A).

Pipeline (each stage is re-runnable; outputs are committed):
  1. SPECS below hold the gold constraint JSON of every prompt, written FIRST,
     with the intended feasibility. `annotate` computes the acceptable set from
     the catalog; the script aborts if a prompt's feasibility differs from the
     intent.
  2. `--paraphrase MODEL` asks a local Ollama model (NOT the model under test,
     qwen2.5:7b) to reword each template prompt. Results are cached in
     data/evaluation/test_prompt_paraphrases.json; existing entries (including
     manual corrections, marked "edited": true) are never overwritten.
  3. The final set is written to data/evaluation/test_prompts.json and
     validated (schema, vocabulary, acceptable sets, 70/10 split, no overlap
     with the 20 development prompts).

    python scripts/build_test_prompts.py --check            # specs only, prints strata
    python scripts/build_test_prompts.py --paraphrase llama3.1:8b
    python scripts/build_test_prompts.py                    # assemble + validate
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import load_catalog  # noqa: E402
from clickless_eval.prompts import annotate, make_expected as E, render_template, validate_test_set  # noqa: E402

OUT = Path("data/evaluation/test_prompts.json")
PARAPHRASES = Path("data/evaluation/test_prompt_paraphrases.json")
DEV = Path("data/evaluation/constraint_prompts.json")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

NUT = ("peanuts", "tree_nuts")

# id, expected JSON, infeasible reason (None = intended feasible)
SPECS: list[tuple[str, dict, str | None]] = [
    # ---- 1 constraint (23 feasible) -------------------------------------------------
    ("T01", E("chips", ["gluten_free"]), None),
    ("T02", E("cookies", ["vegetarian"], quantity=2), None),
    ("T03", E("nuts_seeds", ["kosher"]), None),
    ("T04", E("juice", ["no_added_sugar"]), None),
    ("T05", E("bread", ["organic"], quantity=2), None),
    ("T06", E("yogurt", max_price=3), None),
    ("T07", E("pasta", max_price=3, quantity=3), None),
    ("T08", E("coffee_tea", max_price=7), None),
    ("T09", E("cheese", ["kosher"]), None),
    ("T10", E("snack_bars", excluded=["peanuts"]), None),
    ("T11", E("crackers", excluded=["gluten"]), None),
    ("T12", E("chocolate_candy", excluded=["milk"], quantity=2), None),
    ("T13", E("rice_grains", ["non_gmo"]), None),
    ("T14", E("pasta_sauce", ["vegan"]), None),
    ("T15", E("soup", ["gluten_free"], quantity=4), None),
    ("T16", E("breakfast_cereal", ["non_gmo"]), None),
    ("T17", E("milk", ["lactose_free"]), None),
    ("T18", E("nut_butters", excluded=["peanuts"]), None),
    ("T19", E("plant_milk", excluded=["tree_nuts"]), None),
    ("T20", E("snacks", ["vegan"]), None),
    ("T21", E("beverages", max_price=4, quantity=2), None),
    ("T22", E("dairy", ["organic"]), None),
    ("T23", E("coffee_tea", ["organic"]), None),
    # ---- 2 constraints (23 feasible) ------------------------------------------------
    ("T24", E("chips", ["vegan"], max_price=4), None),
    ("T25", E("cookies", ["vegan"], ["soy"]), None),
    ("T26", E("yogurt", ["lactose_free"], max_price=6, quantity=2), None),
    ("T27", E("cheese", ["gluten_free", "vegetarian"]), None),
    ("T28", E("nut_butters", ["organic"], max_price=8), None),
    ("T29", E("breakfast_cereal", ["organic", "vegan"], quantity=2), None),
    ("T30", E("snack_bars", ["gluten_free"], ["soy"]), None),
    ("T31", E("crackers", ["non_gmo"], max_price=4), None),
    ("T32", E("chocolate_candy", ["fair_trade"], max_price=6), None),
    ("T33", E("nuts_seeds", ["organic"], max_price=10, quantity=2), None),
    ("T34", E("bread", ["non_gmo"], ["sesame"], quantity=3), None),
    ("T35", E("rice_grains", ["gluten_free", "non_gmo"]), None),
    ("T36", E("pasta", ["vegan"], max_price=2.5), None),
    ("T37", E("pasta_sauce", excluded=["milk"], max_price=6), None),
    ("T38", E("soup", ["organic", "vegan"], quantity=2), None),
    ("T39", E("juice", ["vegan"], max_price=5), None),
    ("T40", E("coffee_tea", ["vegan"], max_price=8), None),
    ("T41", E("plant_milk", ["kosher"], ["soy"]), None),
    ("T42", E("milk", ["organic"], max_price=5), None),
    ("T43", E("pantry", ["kosher"], max_price=4), None),
    ("T44", E("snacks", excluded=list(NUT), quantity=3), None),
    ("T45", E("plant_milk", ["no_added_sugar", "vegan"]), None),
    ("T46", E("chips", ["kosher"], ["milk"]), None),
    # ---- 3+ constraints (24 feasible) -----------------------------------------------
    ("T47", E("snack_bars", ["organic", "non_gmo"], max_price=7, quantity=2), None),
    ("T48", E("cookies", ["vegetarian"], list(NUT), quantity=2), None),
    ("T49", E("crackers", ["gluten_free"], ["sesame"], max_price=5.5), None),
    ("T50", E("breakfast_cereal", ["organic", "non_gmo"], max_price=7), None),
    ("T51", E("chips", ["gluten_free", "vegan"], max_price=5), None),
    ("T52", E("plant_milk", ["organic", "vegan"], ["soy"], quantity=2), None),
    ("T53", E("yogurt", ["gluten_free", "vegetarian"], max_price=6), None),
    ("T54", E("cheese", ["kosher", "gluten_free"], max_price=9), None),     # tight
    ("T55", E("nut_butters", ["vegan", "non_gmo"], ["tree_nuts"]), None),
    ("T56", E("nuts_seeds", ["vegan", "kosher"], max_price=12, quantity=2), None),
    ("T57", E("bread", ["organic", "vegan"], max_price=6.5), None),
    ("T58", E("pasta", ["vegan", "non_gmo"], max_price=4, quantity=4), None),
    ("T59", E("pasta_sauce", ["vegan", "gluten_free"], ["milk"]), None),
    ("T60", E("soup", ["gluten_free", "organic"], max_price=5), None),
    ("T61", E("juice", ["vegan", "no_added_sugar"], max_price=6, quantity=3), None),
    ("T62", E("coffee_tea", ["vegan", "organic"], max_price=12), None),
    ("T63", E("rice_grains", ["organic", "vegan", "gluten_free"]), None),
    ("T64", E("chocolate_candy", ["vegan"], ["soy"], max_price=6), None),
    ("T65", E("milk", ["lactose_free", "gluten_free"], max_price=6), None),
    ("T66", E("pasta", ["vegan"], ["gluten"], quantity=2), None),            # tight
    ("T67", E("soup", ["non_gmo", "gluten_free"], ["celery"]), None),        # tight
    ("T68", E("snacks", ["vegan", "kosher"], ["soy"], max_price=8), None),
    ("T69", E("breakfast_cereal", ["no_added_sugar"], ["gluten"], max_price=7), None),
    ("T70", E("snack_bars", ["fair_trade"], ["soy"], max_price=10), None),   # tight
    # ---- 10 deliberately infeasible (in-vocabulary, reach retrieval) ----------------
    ("T71", E("cheese", ["vegan"]), "tag_x_category: no vegan cheese in catalog"),
    ("T72", E("yogurt", excluded=["milk"]), "allergen_x_category: every yogurt contains milk"),
    ("T73", E("bread", ["gluten_free"]), "tag_x_category: no gluten-free bread"),
    ("T74", E("coffee_tea", max_price=4), "price_below_min: cheapest coffee/tea is $4.52"),
    ("T75", E("nuts_seeds", max_price=4, quantity=2), "price_below_min: cheapest nuts/seeds is $4.05"),
    ("T76", E("crackers", ["gluten_free", "organic"]), "tag_conjunction: no organic gluten-free crackers"),
    ("T77", E("milk", ["organic", "lactose_free"]), "tag_conjunction: no organic lactose-free milk"),
    ("T78", E("cookies", ["kosher"], ["eggs"]), "tag_x_allergen: no egg-free kosher cookies"),
    ("T79", E("cookies", ["vegan"], ["soy"], max_price=4), "three_way: no vegan soy-free cookies under $4"),
    ("T80", E("dairy", ["vegan"]), "group_level: no vegan product in the dairy group"),
]

PARAPHRASE_SYSTEM = """You rewrite grocery-shopping requests so they sound like real customers typing into a shopping assistant.
Rules:
- Keep EVERY constraint exactly: the product type, every dietary label, every ingredient or allergen to avoid, the price limit and the quantity.
- Do not add, remove or weaken any constraint. Do not add brands, sizes or new preferences.
- Keep dietary labels recognisable (e.g. gluten-free, organic, vegan, vegetarian, non-GMO, lactose-free, kosher, no added sugar, fair trade).
- Vary sentence structure and tone (casual, polite, terse). One or two sentences.
- Output only the rewritten request, nothing else."""


def paraphrase(model: str, text: str, seed: int) -> str:
    payload = {"model": model, "stream": False,
               "messages": [{"role": "system", "content": PARAPHRASE_SYSTEM},
                            {"role": "user", "content": text}],
               "options": {"temperature": 0.8, "seed": seed}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"].strip().strip('"').strip()


def build_rows(catalog: list[dict]) -> list[dict]:
    rows = []
    for rid, exp, reason in SPECS:
        row = annotate({"id": rid, "expected": exp, "template": render_template(exp),
                        "infeasible_reason": reason}, catalog)
        intended = reason is None
        if row["feasible"] != intended:
            raise SystemExit(f"{rid}: intended feasible={intended} but acceptable set has "
                             f"{row['n_acceptable']} products: {exp}")
        rows.append(row)
    return rows


def print_summary(rows: list[dict]) -> None:
    strata = collections.Counter((r["stratum"], r["feasible"]) for r in rows)
    print("strata (stratum, feasible):", dict(sorted(strata.items())))
    print("leaf coverage:", len({r["expected"]["category"] for r in rows
                                 if r["category_level"] == "leaf"}), "of 19 leaves;",
          sum(r["category_level"] == "group" for r in rows), "group prompts")
    print("quantity>1:", sum(r["expected"]["quantity"] > 1 for r in rows))
    print("tight (1-2 acceptable):", sum(1 <= r["n_acceptable"] <= 2 for r in rows))
    for r in rows:
        print(f"  {r['id']} n={r['n_constraints']} acc={r['n_acceptable']:3d}  {r['template']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", default="data/catalog.json")
    ap.add_argument("--check", action="store_true", help="annotate specs and print strata only")
    ap.add_argument("--paraphrase", metavar="MODEL", help="fill missing paraphrases with this Ollama model")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    rows = build_rows(catalog)
    if args.check:
        print_summary(rows)
        return

    cache = json.loads(PARAPHRASES.read_text(encoding="utf-8")) if PARAPHRASES.exists() else {}
    if args.paraphrase:
        if args.paraphrase.startswith("qwen2.5:7b"):
            raise SystemExit("refusing to paraphrase with the model under test (qwen2.5:7b)")
        for i, r in enumerate(rows):
            if r["id"] in cache:
                continue
            text = paraphrase(args.paraphrase, r["template"], seed=573 + i)
            cache[r["id"]] = {"text": text, "model": args.paraphrase, "edited": False, "note": ""}
            print(f"{r['id']}: {text}")
            PARAPHRASES.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")

    missing = [r["id"] for r in rows if r["id"] not in cache]
    if missing:
        raise SystemExit(f"missing paraphrases for {missing}; run with --paraphrase MODEL")
    for r in rows:
        c = cache[r["id"]]
        r["prompt"] = c["text"]
        r["source"] = "template+paraphrase" + ("+manual_edit" if c.get("edited") else "")
        r["paraphrase_model"] = c["model"]
        r["verified_by"] = c.get("verified_by", [])
        r["notes"] = c.get("note", "")
    dev = json.loads(DEV.read_text(encoding="utf-8")) if DEV.exists() else []
    problems = validate_test_set(rows, catalog, dev_rows=dev)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} prompts to {OUT}")
    print_summary(rows)
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
