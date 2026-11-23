"""Modeled click reduction (proposal Sec. V.C, Eqs. 5-6) for Configs B and C.

Manual task-flow model of the sandbox store: one keyword search (the TF-IDF
ranker over the request text), f filter selections (category always, plus the
price filter when the request has a price cap), then opening product pages in
rank order until the first acceptable product (r page views; dietary labels and
allergens are only visible on the product page), then one add-to-cart:

    Clicks_Manual = 1 + f + r + 1                         (5)
    Clicks_Agent  = 2 + e   (one submission, one approval, e corrections)
    Click reduction = (Clicks_Manual - Clicks_Agent) / Clicks_Manual   (6)

Automated runs have no human corrections, so e = 0 on prompts the agent solved;
we report click reduction over solved feasible prompts together with the
success rate (an unsolved prompt would fall back to the manual flow).

    python scripts/click_reduction.py --benchmark data/evaluation/benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg import vocab  # noqa: E402
from clickless_kg.catalog import load_catalog  # noqa: E402
from clickless_eval.metrics import click_reduction, clicks_manual  # noqa: E402
from clickless_mining.tfidf import TfidfIndex  # noqa: E402


def manual_flow(row: dict, index: TfidfIndex) -> dict:
    exp = row["expected"]
    leaves = set(vocab.categories_for_request(exp["category"]))
    f = 1 + (exp.get("max_price") is not None)
    acceptable = set(row["acceptable"])
    r = None
    filtered = [p for p, _ in index.search(row["prompt"], k=None)
                if p["category"] in leaves and (exp.get("max_price") is None or p["price"] <= exp["max_price"])]
    for rank, p in enumerate(filtered, 1):
        if p["code"] in acceptable:
            r = rank
            break
    return {"f": f, "r": r, "clicks_manual": clicks_manual(f, r) if r else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="data/evaluation/test_prompts.json")
    ap.add_argument("--benchmark", default="data/evaluation/benchmark_results.json")
    ap.add_argument("--out", default="data/evaluation/click_reduction.json")
    args = ap.parse_args()

    rows = {r["id"]: r for r in json.loads(Path(args.dataset).read_text(encoding="utf-8"))}
    bench = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    index = TfidfIndex(load_catalog())
    manual = {rid: manual_flow(r, index) for rid, r in rows.items() if r["feasible"]}
    out = {"benchmark": args.benchmark, "manual": manual, "configs": {}}
    for config in ("B", "C"):
        if config not in bench["configs"]:
            continue
        solved = [r for r in bench["configs"][config]["results"] if r["feasible"] and r["success"]]
        reds = [click_reduction(manual[r["id"]]["f"], manual[r["id"]]["r"], e=0) for r in solved]
        clicks_m = [manual[r["id"]]["clicks_manual"] for r in solved]
        out["configs"][config] = {
            "solved_feasible": len(solved), "feasible": len(manual),
            "mean_clicks_manual": sum(clicks_m) / len(clicks_m) if clicks_m else None,
            "clicks_agent": 2,
            "mean_click_reduction_pct": sum(reds) / len(reds) if reds else None,
            "median_click_reduction_pct": sorted(reds)[len(reds) // 2] if reds else None,
        }
    all_m = [m["clicks_manual"] for m in manual.values()]
    out["mean_clicks_manual_all_feasible"] = sum(all_m) / len(all_m)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"manual clicks over {len(all_m)} feasible prompts: mean {out['mean_clicks_manual_all_feasible']:.2f}")
    for c, s in out["configs"].items():
        print(f"{c}: solved {s['solved_feasible']}/{s['feasible']}, manual {s['mean_clicks_manual']:.2f} vs agent 2 "
              f"-> mean reduction {s['mean_click_reduction_pct']:.1f}% (median {s['median_click_reduction_pct']:.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
