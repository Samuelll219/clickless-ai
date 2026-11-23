"""Run Config A (TF-IDF top-1, no LLM, no KG) on the test prompts.

Also reports TF-IDF Recall@20 / MRR, i.e. how often an acceptable product is
among the candidates Config B receives (Config B's ceiling).

    python scripts/run_config_a.py
    python scripts/run_config_a.py --dataset data/evaluation/constraint_prompts.json --out /tmp/a_dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import acceptable_set, load_catalog  # noqa: E402
from clickless_eval.metrics import by_stratum, mrr, recall_at_k, summarize, violated  # noqa: E402
from clickless_mining.tfidf import CONFIG_B_DEPTH, DEFAULT_VECTORIZER, TfidfIndex, config_a  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="data/evaluation/test_prompts.json")
    ap.add_argument("--catalog", default="data/catalog.json")
    ap.add_argument("--out", default="data/evaluation/config_a_results.json")
    ap.add_argument("--field", default="prompt", help="prompt text field (e.g. 'template')")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    index = TfidfIndex(catalog)

    results, ranks = [], []
    for r in rows:
        acc = r.get("acceptable")
        if acc is None:
            acc = acceptable_set(catalog, r["expected"])
        feasible = r.get("feasible", bool(acc))
        product = config_a(r[args.field], index)
        rank = index.rank_of_first(r[args.field], acc) if feasible else None
        if feasible:
            ranks.append(rank)
        results.append({
            "id": r.get("id"), "prompt": r[args.field], "expected": r["expected"],
            "feasible": feasible, "acceptable": acc, "stratum": r.get("stratum"),
            "product": product,
            "selected_code": product["code"], "selected_name": product["name"],
            "violated": violated(product, r["expected"]),
            "rank_first_acceptable": rank,
        })

    summary = summarize(results)
    summary["tfidf_recall@1"] = recall_at_k(ranks, 1)
    summary[f"tfidf_recall@{CONFIG_B_DEPTH}"] = recall_at_k(ranks, CONFIG_B_DEPTH)
    summary["tfidf_mrr"] = mrr(ranks)
    violation_counts: dict[str, int] = {}
    for res in results:
        if res["feasible"]:
            for v in res["violated"]:
                kind = v.split(":")[0]
                violation_counts[kind] = violation_counts.get(kind, 0) + 1
    out = {
        "config": "A (TF-IDF top-1)", "dataset": args.dataset, "text_field": args.field,
        "vectorizer": {k: list(v) if isinstance(v, tuple) else v for k, v in DEFAULT_VECTORIZER.items()},
        "summary": summary,
        "by_stratum": by_stratum(results, "stratum") if any(r.get("stratum") for r in rows) else {},
        "violations_on_feasible": violation_counts,
        "results": [{k: v for k, v in res.items() if k != "product"} for res in results],
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s = summary
    print(f"Config A on {s['n_prompts']} prompts ({s['n_feasible']} feasible, {s['n_infeasible']} infeasible)")
    print(f"  CSS {s['css']:.3f}  TSR {s['tsr']:.3f} {tuple(round(x, 3) for x in s['tsr_ci95'])}  "
          f"IRA {s['ira']}")
    print(f"  TF-IDF recall@1 {s['tfidf_recall@1']:.3f}  recall@{CONFIG_B_DEPTH} "
          f"{s[f'tfidf_recall@{CONFIG_B_DEPTH}']:.3f}  MRR {s['tfidf_mrr']:.3f}")
    print(f"  violations on feasible prompts: {violation_counts}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
