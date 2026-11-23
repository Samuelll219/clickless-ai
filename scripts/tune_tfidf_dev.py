"""Compare TF-IDF vectorizer settings on the DEVELOPMENT prompts only.

The 20 prompts in data/evaluation/constraint_prompts.json were used to develop
the extractor and are therefore a development set. The chosen setting is
frozen in clickless_mining.tfidf.DEFAULT_VECTORIZER before the 80 test prompts
are used, so the baseline is never tuned on the test set.

    python scripts/tune_tfidf_dev.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import acceptable_set, load_catalog  # noqa: E402
from clickless_mining.tfidf import DEFAULT_VECTORIZER, TfidfIndex  # noqa: E402

VARIANTS = {
    "sklearn defaults (frozen)": dict(DEFAULT_VECTORIZER),
    "+strip_accents": {**DEFAULT_VECTORIZER, "strip_accents": "unicode"},
    "+sublinear_tf": {**DEFAULT_VECTORIZER, "sublinear_tf": True},
    "+english stop words": {**DEFAULT_VECTORIZER, "stop_words": "english"},
    "+bigrams": {**DEFAULT_VECTORIZER, "ngram_range": (1, 2)},
}


def evaluate(index: TfidfIndex, rows: list[dict], catalog: list[dict]) -> dict:
    top1 = r20 = 0
    rr = 0.0
    n = 0
    for row in rows:
        acc = acceptable_set(catalog, row["expected"])
        if not acc:
            continue
        n += 1
        rank = index.rank_of_first(row["prompt"], acc)
        top1 += rank == 1
        r20 += rank is not None and rank <= 20
        rr += 1.0 / rank if rank else 0.0
    return {"feasible": n, "top1": top1, "recall@20": r20, "mrr": round(rr / n, 3) if n else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="data/evaluation/constraint_prompts.json")
    ap.add_argument("--catalog", default="data/catalog.json")
    args = ap.parse_args()
    catalog = load_catalog(args.catalog)
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    for name, kw in VARIANTS.items():
        print(f"{name:32s} {evaluate(TfidfIndex(catalog, **kw), rows, catalog)}")


if __name__ == "__main__":
    main()
