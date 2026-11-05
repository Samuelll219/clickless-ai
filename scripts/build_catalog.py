#!/usr/bin/env python3
"""Step 1: fetch Open Food Facts candidates and build the curated catalog.

    python scripts/build_catalog.py --fetch          # download (cached) + curate
    python scripts/build_catalog.py                  # re-curate from data/raw/off_raw.jsonl
    python scripts/build_catalog.py --raw dump.jsonl # curate from any OFF JSONL / CSV dump

Outputs (in --out-dir, default data/):
    catalog.json / catalog.csv   the ~400-product catalog with synthetic prices
    curation_report.json         rejection reasons, per-category / tag / allergen counts,
                                 most common OFF label tags we do not map yet
    vocab.json                   controlled vocabularies (for the validator)
    constraint_schema.json       JSON schema for the Ollama constraint extractor
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg import vocab  # noqa: E402
from clickless_kg.catalog import validate_catalog  # noqa: E402
from clickless_kg.curate import curate  # noqa: E402


def read_raw(path: Path):
    if path.suffix in (".csv", ".tsv") or path.name.endswith(".csv.gz"):
        import pandas as pd
        cols = ["code", "product_name", "brands", "categories_tags", "labels_tags",
                "allergens", "traces_tags", "ingredients_text", "ingredients_analysis_tags",
                "states_tags", "unique_scans_n"]
        for chunk in pd.read_csv(path, sep="\t", usecols=lambda c: c in cols, dtype=str,
                                 chunksize=50_000, on_bad_lines="skip", low_memory=False):
            yield from chunk.to_dict("records")
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download from the OFF API first")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/off_raw.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--seed", type=int, default=573)
    ap.add_argument("--min-tags", type=int, default=1)
    ap.add_argument("--no-analysis-tags", action="store_true",
                    help="use only printed labels for vegan/vegetarian, not OFF ingredient analysis")
    ap.add_argument("--pages", type=int, default=2, help="API pages per OFF category tag")
    ap.add_argument("--country", default="en:united-states", help="'' for all countries")
    args = ap.parse_args()

    if args.fetch:
        from clickless_kg.fetch_off import fetch_all
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        n = fetch_all(args.raw, pages=args.pages, country=args.country or None)
        print(f"fetched {n} unique products -> {args.raw}")

    catalog, report = curate(read_raw(args.raw), target=args.target, seed=args.seed,
                             min_tags=args.min_tags, use_analysis=not args.no_analysis_tags)
    problems = validate_catalog(catalog)
    if problems:
        sys.exit("catalog invalid:\n" + "\n".join(problems[:20]))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "catalog.json").write_text(json.dumps(catalog, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out / "catalog.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "brand", "category", "price", "dietary_tags",
                    "allergens", "allergens_added", "traces", "ingredients_text"])
        for p in catalog:
            w.writerow([p["code"], p["name"], p["brand"], p["category"], f"{p['price']:.2f}",
                        "|".join(p["dietary_tags"]), "|".join(p["allergens"]),
                        "|".join(p["allergens_added"]), "|".join(p["traces"]), p["ingredients_text"]])
    (out / "curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab.vocabulary(), indent=2), encoding="utf-8")
    (out / "constraint_schema.json").write_text(json.dumps(vocab.constraint_json_schema(), indent=2), encoding="utf-8")

    print(f"raw={report['raw_records']} eligible={report['eligible']} "
          f"selected={report['selected']}/{args.target}")
    print("per category:", report["selected_per_category"])
    short = [c for c in vocab.CATEGORY_NAMES
             if report["eligible_per_category"].get(c, 0) < args.target // len(vocab.CATEGORY_NAMES)]
    if short:
        print("WARNING: categories with fewer eligible products than an even share:", short)
    print(f"wrote {out}/catalog.json, catalog.csv, curation_report.json, vocab.json, "
          "constraint_schema.json")


if __name__ == "__main__":
    main()
