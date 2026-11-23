"""Benchmark Configs A, B and C on the 80 test prompts (proposal Sec. V.B-V.D).

  A  TF-IDF top-1 (no LLM, no KG)
  B  local LLM picks from the TF-IDF top-20 (no KG)          [Soumya's config_b]
  C  LLM extraction -> validator -> KG retrieval             [Soumya/Samuel's config_c]

B and C use the same Ollama model (default qwen2.5:7b, temperature 0) and
the same catalog, so B vs C isolates structured KG retrieval. By default the
KG is the catalog-backed OfflineKG (same semantics as the Cypher template,
asserted by tests/test_neo4j_integration.py); pass --kg neo4j to query a
running Neo4j instead.

LLM outputs are cached per (config, model, prompt) in
data/evaluation/benchmark_raw.json so interrupted runs resume.

    python scripts/run_benchmark.py                       # A, B, C
    python scripts/run_benchmark.py --configs A C --kg neo4j
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg.catalog import load_catalog  # noqa: E402
from clickless_eval.metrics import (  # noqa: E402
    by_stratum, holm, mcnemar_exact, paired_bootstrap_diff_ci, summarize, task_success, css, violated,
)
from clickless_llm.evaluation import evaluate_predictions  # noqa: E402
from clickless_mining.addons import OfflineKG, RuleIndex, recommend_addons  # noqa: E402
from clickless_mining.tfidf import TfidfIndex, config_a, tfidf_candidates  # noqa: E402

RAW = Path("data/evaluation/benchmark_raw.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_b(prompt, index, model):
    from clickless_llm.config_b import ConfigBError, select_from_candidates
    cands = tfidf_candidates(prompt, index)
    t0 = time.time()
    try:
        chosen, qty = select_from_candidates(prompt, cands, model=model, return_quantity=True)
        err = None
    except ConfigBError as exc:
        chosen, qty, err = None, None, str(exc)
    return {"code": chosen["code"] if chosen else None, "quantity": qty, "error": err,
            "candidate_codes": [c["code"] for c in cands], "seconds": round(time.time() - t0, 2)}


def run_c(prompt, kg, model):
    from clickless_llm.config_c import run_config_c
    from clickless_llm.pipeline import PipelineError
    t0 = time.time()
    try:
        res = run_config_c(prompt, kg=kg, model=model)
        out = {"code": res["products"][0]["code"] if res["products"] else None,
               "constraints": res["constraints"], "quantity": res["quantity"],
               "status": res["status"], "error": None}
    except PipelineError as exc:
        out = {"code": None, "constraints": None, "quantity": None, "status": "error",
               "error": str(exc)}
    out["seconds"] = round(time.time() - t0, 2)
    return out


def failure_stage(config, row, raw, product):
    """Where a failed prompt broke (proposal Sec. V.D)."""
    if row["feasible"]:
        if task_success(product, row["acceptable"]):
            return None
        if config == "B":
            if not set(raw["candidate_codes"]) & set(row["acceptable"]):
                return "retrieval (no acceptable product in TF-IDF top-20)"
            return "selection (no product)" if product is None else "selection (violating product)"
        if config == "C":
            if raw["constraints"] is None:
                return "validation (retries exhausted)"
            return "extraction"
        return "retrieval"
    if product is None:
        return None
    return {"A": "retrieval (always returns a product)", "B": "selection (violating product)",
            "C": "extraction (dropped/changed a constraint)"}[config]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="data/evaluation/test_prompts.json")
    ap.add_argument("--catalog", default="data/catalog.json")
    ap.add_argument("--configs", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--kg", choices=("offline", "neo4j"), default="offline")
    ap.add_argument("--rules", default="data/addon_rules.json")
    ap.add_argument("--out", default="data/evaluation/benchmark_results.json")
    ap.add_argument("--tag", default="v2", help="cache version tag (v1 = frozen pre-error-analysis run)")
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    by_code = {p["code"]: p for p in catalog}
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    index = TfidfIndex(catalog)
    raw_cache = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else {}

    kg = None
    if "C" in args.configs:
        if args.kg == "neo4j":
            from clickless_kg.retrieval import KGRetriever
            kg = KGRetriever.from_env()
        else:
            kg = OfflineKG(catalog)

    per_config: dict[str, list[dict]] = {}
    for config in args.configs:
        results = []
        for i, row in enumerate(rows, 1):
            key = f"{config}|{args.model if config != 'A' else '-'}|{row['prompt']}"
            if args.tag != "v1":
                key = f"{args.tag}|{key}"
            if config == "A":
                raw = {"code": config_a(row["prompt"], index)["code"]}
            elif key in raw_cache:
                raw = raw_cache[key]
            else:
                raw = run_b(row["prompt"], index, args.model) if config == "B" else \
                    run_c(row["prompt"], kg, args.model)
                raw_cache[key] = raw
                RAW.write_text(json.dumps(raw_cache, indent=1) + "\n", encoding="utf-8")
                log(f"{config} {row['id']} ({i}/{len(rows)}): {raw.get('code')} {raw.get('error') or ''}")
            product = by_code.get(raw["code"]) if raw["code"] else None
            results.append({
                "id": row["id"], "feasible": row["feasible"], "expected": row["expected"],
                "acceptable": row["acceptable"], "stratum": row["stratum"],
                "category_level": row["category_level"], "product": product,
                "quantity": raw.get("quantity"), "raw": raw,
                "css": css(product, row["expected"]) if row["feasible"] else None,
                "success": task_success(product, row["acceptable"]) if row["feasible"] else product is None,
                "violated": violated(product, row["expected"]) if product else [],
                "failure_stage": failure_stage(config, row, raw, product),
            })
        per_config[config] = results
    if kg is not None:
        kg.close()

    report = {"dataset": args.dataset, "model": args.model, "kg": args.kg, "tag": args.tag, "configs": {}}
    for config, results in per_config.items():
        s = summarize(results)
        if config in ("B", "C"):
            s["tsr_with_quantity"] = summarize(results, check_quantity=True)["tsr"]
        if config == "C":
            preds = [(r["raw"]["constraints"] or {}, r["expected"]) for r in results]
            cea = evaluate_predictions(preds)
            s["constraint_extraction"] = {"exact_match": cea["exact_match_accuracy"],
                                          "field_accuracy": cea["field_accuracy"]}
        stages: dict[str, int] = {}
        for r in results:
            if r["failure_stage"]:
                stages[r["failure_stage"]] = stages.get(r["failure_stage"], 0) + 1
        secs = [r["raw"].get("seconds") for r in results if r["raw"].get("seconds") is not None]
        report["configs"][config] = {
            "summary": s, "by_stratum": by_stratum(results, "stratum"),
            "by_category_level": by_stratum(results, "category_level"),
            "failure_stages": stages,
            "median_seconds": sorted(secs)[len(secs) // 2] if secs else None,
            "results": [{k: (v["code"] if k == "product" and v else v) for k, v in r.items()}
                        for r in results],
        }

    # Paired comparisons on the same prompts (feasible: TSR & CSS; infeasible: IRA).
    feas_idx = [i for i, r in enumerate(rows) if r["feasible"]]
    comps, pvals = {}, {}
    names = list(per_config)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            ra, rb = per_config[a], per_config[b]
            tsr_test = mcnemar_exact([ra[i]["success"] for i in feas_idx], [rb[i]["success"] for i in feas_idx])
            css_diff = paired_bootstrap_diff_ci([ra[i]["css"] for i in feas_idx], [rb[i]["css"] for i in feas_idx])
            comps[f"{a} vs {b}"] = {"tsr_mcnemar": tsr_test, "css_paired_bootstrap": css_diff}
            pvals[f"{a} vs {b} TSR"] = tsr_test["p_value"]
            pvals[f"{a} vs {b} CSS"] = css_diff["p_value"]
    adj = holm(pvals) if pvals else {}
    for name, c in comps.items():
        c["tsr_mcnemar"]["p_holm"] = adj[f"{name} TSR"]
        c["css_paired_bootstrap"]["p_holm"] = adj[f"{name} CSS"]
    report["comparisons"] = comps

    # Add-ons on Config C matches (offline rules, if mined).
    if "C" in per_config and Path(args.rules).exists():
        from clickless_mining.apriori_rules import load_rules
        rules = load_rules(args.rules)
        ridx = RuleIndex(rules["rules"], rules["popularity"], rank_by=rules["meta"].get("rank_by", "lift"))
        akg = OfflineKG(catalog)
        n_matched = n_full = n_comp = n_addons = 0
        for r in per_config["C"]:
            if r["product"] is None:
                continue
            n_matched += 1
            cons = r["raw"]["constraints"]
            addons = recommend_addons([r["product"]["category"]], cons, ridx, akg, k=3)
            n_full += len(addons) == 3
            for a in addons:
                n_addons += 1
                n_comp += not violated(a["product"], {**cons, "category": a["category"], "max_price": None})
        report["addons_on_config_c"] = {
            "matched_prompts": n_matched, "prompts_with_3_addons": n_full,
            "addon_constraint_compliance": n_comp / n_addons if n_addons else None}

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{'cfg':4s} {'CSS':>6s} {'TSR':>6s} {'IRA':>6s} {'infF1':>6s} {'overref':>8s}")
    for config, block in report["configs"].items():
        s = block["summary"]
        print(f"{config:4s} {s['css']:6.3f} {s['tsr']:6.3f} {s['ira']:6.3f} "
              f"{s['infeasible_f1']:6.3f} {s['over_refusal_rate']:8.3f}")
    for name, c in comps.items():
        print(f"{name}: TSR McNemar p={c['tsr_mcnemar']['p_value']:.4f} (Holm {c['tsr_mcnemar']['p_holm']:.4f}); "
              f"CSS diff {c['css_paired_bootstrap']['mean_diff']:+.3f} CI {tuple(round(x, 3) for x in c['css_paired_bootstrap']['ci95'])}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
