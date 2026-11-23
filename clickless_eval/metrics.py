"""Benchmark metrics (proposal Sec. V.C) and the statistics used to report them.

Definitions (fixed here so all three configurations are scored identically):

* A *constraint* is one of: the category, each required dietary tag, each
  excluded allergen, and the price cap. Quantity is not a constraint for CSS;
  it is checked by TSR for Configs B and C only.
* CSS (Constraint Satisfaction Score), per feasible prompt: satisfied
  constraints / total constraints for the selected product. A system that
  returns no product on a feasible prompt scores 0.
* TSR (Task Success Rate), per feasible prompt: the selected product is in the
  annotated acceptable set (and, when `check_quantity`, the quantity matches).
* IRA (Infeasible-Request Accuracy), per infeasible prompt: the system returns
  no product. Because only 10 of 80 prompts are infeasible, an "always
  feasible" system already scores 87.5% plain accuracy, so we also report the
  infeasible-class precision / recall / F1 and the over-refusal rate
  (no product returned on a feasible prompt).
* Click reduction: proposal Eqs. (5)-(6).
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from clickless_kg import vocab


# --------------------------------------------------------------------------- per prompt

def constraint_checks(product: dict | None, expected: dict) -> list[tuple[str, bool]]:
    """(constraint name, satisfied?) for every annotated constraint."""
    leaves = set(vocab.categories_for_request(expected["category"]))
    checks = [("category", product is not None and product.get("category") in leaves)]
    tags = set(product.get("dietary_tags", [])) if product else set()
    allergens = set(product.get("allergens", [])) if product else set()
    for t in expected.get("dietary_tags", []):
        checks.append((f"tag:{t}", product is not None and t in tags))
    for a in expected.get("excluded_allergens", []):
        checks.append((f"allergen:{a}", product is not None and a not in allergens))
    if expected.get("max_price") is not None:
        checks.append(("price", product is not None and product["price"] <= expected["max_price"]))
    return checks


def css(product: dict | None, expected: dict) -> float:
    checks = constraint_checks(product, expected)
    return sum(ok for _, ok in checks) / len(checks)


def task_success(product: dict | None, acceptable: Iterable[str], *, quantity: int | None = None,
                 expected_quantity: int | None = None, check_quantity: bool = False) -> bool:
    if product is None or product["code"] not in set(acceptable):
        return False
    return not check_quantity or quantity == expected_quantity


def violated(product: dict | None, expected: dict) -> list[str]:
    """Names of the constraints the product violates (for error analysis)."""
    return [name for name, ok in constraint_checks(product, expected) if not ok]


# --------------------------------------------------------------------------- aggregate

def summarize(results: Sequence[dict], *, check_quantity: bool = False) -> dict:
    """Aggregate per-prompt results.

    Each result needs: `feasible` (bool), `expected` (constraint JSON),
    `acceptable` (codes), `product` (dict or None) and, if `check_quantity`,
    `quantity`.
    """
    feas = [r for r in results if r["feasible"]]
    infeas = [r for r in results if not r["feasible"]]
    css_vals = [css(r["product"], r["expected"]) for r in feas]
    tsr_vals = [task_success(r["product"], r["acceptable"], quantity=r.get("quantity"),
                             expected_quantity=r["expected"].get("quantity"),
                             check_quantity=check_quantity) for r in feas]
    # Infeasible class = positive class ("system says no product").
    tp = sum(r["product"] is None for r in infeas)
    fn = len(infeas) - tp
    fp = sum(r["product"] is None for r in feas)
    tn = len(feas) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {
        "n_prompts": len(results), "n_feasible": len(feas), "n_infeasible": len(infeas),
        "css": _mean(css_vals), "tsr": _mean(tsr_vals),
        "tsr_successes": sum(tsr_vals),
        "ira": tp / len(infeas) if infeas else None,
        "infeasible_confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "infeasible_precision": prec, "infeasible_recall": rec, "infeasible_f1": f1,
        "over_refusal_rate": fp / len(feas) if feas else None,
        "tsr_ci95": wilson_ci(sum(tsr_vals), len(tsr_vals)),
        "ira_ci95": wilson_ci(tp, len(infeas)) if infeas else None,
        "css_ci95": bootstrap_mean_ci(css_vals) if css_vals else None,
    }
    return out


def by_stratum(results: Sequence[dict], key: str, **kw) -> dict:
    """`summarize` separately for each value of results[i][key]."""
    groups: dict = {}
    for r in results:
        groups.setdefault(r.get(key), []).append(r)
    return {str(k): summarize(v, **kw) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def recall_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Share of queries whose first relevant item has rank <= k (None = never)."""
    ranks = list(ranks)
    return _mean([r is not None and r <= k for r in ranks])


def mrr(ranks: Iterable[int | None]) -> float:
    return _mean([1.0 / r if r else 0.0 for r in ranks])


# --------------------------------------------------------------------------- click model

def clicks_manual(f: int, r: int) -> int:
    """Eq. (5): one search + f filter selections + r product-page views + one add-to-cart."""
    return 1 + f + r + 1


def clicks_agent(e: int = 0) -> int:
    """One submission + one approval + e corrections."""
    return 2 + e


def click_reduction(f: int, r: int, e: int = 0) -> float:
    """Eq. (6), in percent."""
    m = clicks_manual(f, r)
    return (m - clicks_agent(e)) / m * 100.0


# --------------------------------------------------------------------------- statistics

def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson score 95% interval for a proportion (good for small n, e.g. IRA with n = 10)."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_success: Sequence[bool], b_success: Sequence[bool]) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes.

    b = A succeeds & B fails, c = A fails & B succeeds; two-sided p-value.
    """
    if len(a_success) != len(b_success):
        raise ValueError("paired outcomes must have equal length")
    b = sum(x and not y for x, y in zip(a_success, b_success))
    c = sum(y and not x for x, y in zip(a_success, b_success))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_value": min(1.0, 2 * tail)}


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10_000, seed: int = 573,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean."""
    rng = random.Random(seed)
    vals = list(values)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1])


def paired_bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                             seed: int = 573, alpha: float = 0.05) -> dict:
    """CI for mean(b - a) over paired items, plus a two-sided bootstrap p-value."""
    if len(a) != len(b):
        raise ValueError("paired values must have equal length")
    diffs = [y - x for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    obs = sum(diffs) / n
    p = 2 * min(sum(d <= 0 for d in boots), sum(d >= 0 for d in boots)) / n_boot
    return {"mean_diff": obs, "ci95": (boots[int(alpha / 2 * n_boot)],
                                       boots[int((1 - alpha / 2) * n_boot) - 1]),
            "p_value": min(1.0, p)}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
