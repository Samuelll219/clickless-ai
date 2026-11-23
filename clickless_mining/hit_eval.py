"""Hit@k evaluation of add-on category recommendation (proposal Sec. V.C).

Protocols on held-out orders with >= 2 mapped categories:
  leave_one_out (primary; Karypis, CIKM 2001): hide one category chosen at
      random (fixed seed); the rest is the cart; hit if the hidden category is
      among the top-k recommendations.
  first_item: the cart is the category of the first item added to the order
      (mirrors ClickLess's single-item requests); hit if ANY other category of
      the order is among the top-k.

Recommenders never return categories already in the cart. Baselines: the
most-popular categories (training support) and uniformly random categories.
Hit@k = hits / #queries (hit-rate, Deshpande & Karypis, TOIS 2004).
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

from clickless_eval.metrics import mcnemar_exact, wilson_ci

Query = tuple[tuple[str, ...], frozenset]          # (cart, held-out targets)


def make_queries(baskets: Sequence[Sequence[str]], protocol: str = "leave_one_out",
                 seed: int = 573, max_queries: int | None = None) -> list[Query]:
    rng = random.Random(seed)
    queries: list[Query] = []
    for b in baskets:
        cats = list(dict.fromkeys(b))               # keep first-added order, dedupe
        if len(cats) < 2:
            continue
        if protocol == "leave_one_out":
            hidden = rng.choice(cats)
            queries.append((tuple(c for c in cats if c != hidden), frozenset([hidden])))
        elif protocol == "first_item":
            queries.append(((cats[0],), frozenset(cats[1:])))
        else:
            raise ValueError(f"unknown protocol {protocol!r}")
    if max_queries is not None and len(queries) > max_queries:
        queries = random.Random(seed + 1).sample(queries, max_queries)
    return queries


def popularity_recommender(popularity: dict[str, float]) -> Callable:
    order = list(popularity)

    def rec(cart, k):
        cs = set(cart)
        return [c for c in order if c not in cs][:k]
    return rec


def random_recommender(items: Sequence[str], seed: int = 573) -> Callable:
    rng = random.Random(seed)
    items = sorted(items)

    def rec(cart, k):
        pool = [c for c in items if c not in set(cart)]
        return rng.sample(pool, min(k, len(pool)))
    return rec


def rule_recommender(index, fallback: bool = True) -> Callable:
    def rec(cart, k):
        return [c for c, _, _ in index.recommend(cart, k=k, fallback=fallback)]
    return rec


def evaluate(queries: Sequence[Query], recommender: Callable, k: int = 3) -> dict:
    hits, rr = [], 0.0
    for cart, targets in queries:
        recs = recommender(cart, k)
        hit_rank = next((i for i, c in enumerate(recs, 1) if c in targets), None)
        hits.append(hit_rank is not None)
        rr += 1.0 / hit_rank if hit_rank else 0.0
    n = len(queries)
    return {"n_queries": n, f"hit@{k}": sum(hits) / n if n else 0.0,
            "hits": sum(hits), "ci95": wilson_ci(sum(hits), n), f"mrr@{k}": rr / n if n else 0.0,
            "_per_query": hits}


def rule_coverage(queries: Sequence[Query], index) -> float:
    """Share of queries where at least one rule fires (no popularity fallback)."""
    if not queries:
        return 0.0
    return sum(bool(index.fired(cart)) for cart, _ in queries) / len(queries)


def compare(a: dict, b: dict) -> dict:
    """Exact McNemar test between two recommenders scored on the same queries."""
    return mcnemar_exact(a["_per_query"], b["_per_query"])


def strip(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}
