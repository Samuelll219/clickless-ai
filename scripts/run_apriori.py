"""Category-level Apriori add-ons on Instacart: mine, select, evaluate Hit@k.

Protocol (no test leakage):
  1. Build category transactions (product -> aisle -> leaf; unmapped dropped).
  2. Split 80/20 BY USER (seed 573); the 20% test users are touched once.
  3. Inside the 80%, hold out 10% of users as validation and pick min_support
     by validation Hit@3 (leave-one-out).
  4. Re-mine on the full 80% with the chosen min_support -> data/addon_rules.json.
  5. Evaluate on the test users: Hit@1/3/5 + MRR for lift-ranked rules,
     confidence-ranked rules, most-popular and random baselines, under the
     leave-one-out and first-item protocols; rule coverage; exact McNemar
     tests; held-out support / confidence / lift of the top rules.

    python scripts/run_apriori.py --instacart-dir data/instacart
    python scripts/run_apriori.py --split order          # literal "random 20% of orders" robustness run
    python scripts/run_apriori.py --include-optional     # sensitivity: weak aisles mapped too
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_mining import aisles, instacart  # noqa: E402
from clickless_mining.addons import RuleIndex  # noqa: E402
from clickless_mining.apriori_rules import (  # noqa: E402
    format_rule, held_out_quality, mine_rules, popularity, rules_to_records, save_rules,
)
from clickless_mining.hit_eval import (  # noqa: E402
    compare, evaluate, make_queries, popularity_recommender, random_recommender,
    rule_coverage, rule_recommender, strip,
)
from clickless_eval.metrics import holm  # noqa: E402

SWEEP = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instacart-dir", default="data/instacart")
    ap.add_argument("--split", choices=("user", "order"), default="user")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--max-len", type=int, default=3)
    ap.add_argument("--min-support", type=float, help="skip validation selection")
    ap.add_argument("--rank-by", choices=("lift", "confidence"), default="lift", help="with --min-support")
    ap.add_argument("--max-queries", type=int, default=100_000, help="random sample of test queries (seeded); 0 = all")
    ap.add_argument("--seed", type=int, default=573)
    ap.add_argument("--rules-out", default="data/addon_rules.json")
    ap.add_argument("--out", default="data/evaluation/hit_at_3.json")
    args = ap.parse_args()

    d = Path(args.instacart_dir)
    if (d / "aisles.csv").exists():
        problems = aisles.validate_against_aisles_csv(d / "aisles.csv")
        if problems:
            raise SystemExit("aisle map does not match aisles.csv:\n  " + "\n  ".join(problems))
    amap = aisles.mapping(include_optional=args.include_optional)
    cache = d / f"transactions{'_opt' if args.include_optional else ''}.pkl"
    if cache.exists():
        log(f"loading cached transactions {cache}")
        tx = instacart.load(cache)
    else:
        log("building transactions (first run takes a few minutes)")
        tx = instacart.build_transactions(d, amap)
        instacart.save(tx, cache)
    log(f"coverage: {tx.stats}")

    splitter = instacart.split_by_user if args.split == "user" else instacart.split_by_order
    train, test = splitter(tx.frame, 0.2, seed=args.seed)
    train_b, test_b = list(train["categories"]), list(test["categories"])
    log(f"train orders {len(train_b)}, test orders {len(test_b)}")

    # ---- select min_support on a validation split of the training users
    selection = []
    if args.min_support is None:
        inner_tr, val = splitter(train, 0.1, seed=args.seed + 1)
        inner_b, val_b = list(inner_tr["categories"]), list(val["categories"])
        val_q = make_queries(val_b, "leave_one_out", seed=args.seed, max_queries=50_000)
        pop_inner = popularity(inner_b)
        val_pop = evaluate(val_q, popularity_recommender(pop_inner), k=3)["hit@3"]
        log(f"  validation most-popular Hit@3={val_pop:.4f} (reference)")
        for s in SWEEP:
            rules, info = mine_rules(inner_b, s, max_len=args.max_len)
            idx = RuleIndex(rules_to_records(rules), pop_inner)
            res = evaluate(val_q, rule_recommender(idx), k=3)
            res_conf = evaluate(val_q, rule_recommender(RuleIndex(rules_to_records(rules), pop_inner,
                                                                  rank_by="confidence")), k=3)
            selection.append({"min_support": s, "n_rules": info["n_rules"],
                              "val_hit@3": res["hit@3"], "val_hit@3_confidence": res_conf["hit@3"],
                              "val_most_popular_hit@3": val_pop,
                              "val_coverage": rule_coverage(val_q, idx)})
            log(f"  min_support={s}: rules={info['n_rules']} val Hit@3 lift={res['hit@3']:.4f} "
                f"confidence={res_conf['hit@3']:.4f}")
        # Joint choice of ranking and threshold on validation; ties -> fewer rules.
        cands = [(r["val_hit@3"], r["min_support"], "lift") for r in selection] + \
                [(r["val_hit@3_confidence"], r["min_support"], "confidence") for r in selection]
        _, min_support, rank_by = max(cands, key=lambda c: (c[0], c[1]))
        log(f"selected rank_by={rank_by} min_support={min_support}")
    else:
        min_support, rank_by = args.min_support, args.rank_by

    # ---- final rules on the full training split
    rules_df, info = mine_rules(train_b, min_support, max_len=args.max_len)
    records = rules_to_records(rules_df)
    pop = popularity(train_b)
    meta = {"split": args.split, "seed": args.seed, "include_optional_aisles": args.include_optional,
            "aisle_map": {str(k): v for k, v in sorted(amap.items())},
            "coverage": tx.stats, "train_orders": len(train_b), "test_orders": len(test_b),
            "mining": info, "selection": selection, "rank_by": rank_by}
    save_rules(args.rules_out, rules_df, pop, meta)
    log(f"{len(records)} rules (rank_by={rank_by}) -> {args.rules_out}")

    # ---- test evaluation (selected setting + the other ranking at the same support)
    lift_idx = RuleIndex(records, pop, rank_by="lift")
    conf_idx = RuleIndex(records, pop, rank_by="confidence")
    sel_idx = conf_idx if rank_by == "confidence" else lift_idx
    leaves = sorted(pop)
    report = {"meta": {k: v for k, v in meta.items() if k != "aisle_map"},
              "min_support": min_support, "rank_by": rank_by, "protocols": {}}
    for protocol in ("leave_one_out", "first_item"):
        q = make_queries(test_b, protocol, seed=args.seed, max_queries=args.max_queries or None)
        recs = {
            "apriori_selected": rule_recommender(sel_idx),
            "apriori_selected_rules_only": rule_recommender(sel_idx, fallback=False),
            "apriori_lift": rule_recommender(lift_idx),
            "apriori_confidence": rule_recommender(conf_idx),
            "most_popular": popularity_recommender(pop),
            "random": random_recommender(leaves, seed=args.seed),
        }
        block = {"n_queries": len(q), "rule_coverage": rule_coverage(q, sel_idx), "k": {}}
        for k in (1, 3, 5):
            res = {name: evaluate(q, fn, k=k) for name, fn in recs.items()}
            tests = {f"apriori_selected vs {b}": compare(res["apriori_selected"], res[b])
                     for b in ("most_popular", "random")}
            tests["apriori_lift vs apriori_confidence"] = compare(res["apriori_lift"], res["apriori_confidence"])
            adj = holm({n: t["p_value"] for n, t in tests.items()})
            for n in tests:
                tests[n]["p_holm"] = adj[n]
            block["k"][str(k)] = {"results": {n: strip(r) for n, r in res.items()}, "mcnemar": tests}
        report["protocols"][protocol] = block
        h = block["k"]["3"]["results"]
        log(f"{protocol}: n={len(q)} coverage={block['rule_coverage']:.3f} Hit@3 "
            + " ".join(f"{n}={r['hit@3']:.4f}" for n, r in h.items()))

    key = (lambda r: (r["confidence"], r["lift"])) if rank_by == "confidence" else \
        (lambda r: (r["lift"], r["confidence"]))
    top_rules = sorted(records, key=key, reverse=True)[:20]
    report["top_rules_heldout"] = held_out_quality(top_rules, test_b)
    report["top_rules_text"] = [format_rule(r) for r in top_rules]
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
