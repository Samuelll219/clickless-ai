"""Add-on recommendation from category-level rules (proposal Sec. IV.C).

Given the leaf categories already in the cart, every mined rule whose
antecedent is contained in the cart votes for its consequent (if that
category is not already in the cart). Candidates are ranked by the best
rule's lift (ties: confidence, support, name); `rank_by="confidence"` is the
ablation. If fewer than k categories are suggested, the most popular
remaining categories fill the list (reported separately as rule coverage).
For each suggested category the cheapest product that satisfies the user's
dietary tags and excluded allergens is looked up in the KG.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

from clickless_kg.catalog import filter_products


class RuleIndex:
    """Rules indexed by antecedent for fast lookup with many carts."""

    def __init__(self, rules: list[dict], popularity: dict[str, float] | None = None,
                 rank_by: str = "lift"):
        if rank_by not in ("lift", "confidence"):
            raise ValueError("rank_by must be 'lift' or 'confidence'")
        self.rank_by = rank_by
        self.popularity = dict(popularity or {})
        self.by_antecedent: dict[frozenset, list[dict]] = {}
        self.max_antecedent = 0
        for r in rules:
            key = frozenset(r["antecedents"])
            self.by_antecedent.setdefault(key, []).append(r)
            self.max_antecedent = max(self.max_antecedent, len(key))
        self._cache: dict[tuple, tuple] = {}

    def _key(self, rule: dict) -> tuple:
        if self.rank_by == "lift":
            return (rule["lift"], rule["confidence"], rule["support"])
        return (rule["confidence"], rule["lift"], rule["support"])

    def fired(self, cart: Iterable[str]) -> dict[str, dict]:
        """consequent -> best firing rule, for consequents not in the cart."""
        cart = frozenset(cart)
        best: dict[str, dict] = {}
        for size in range(1, min(len(cart), self.max_antecedent) + 1):
            for sub in combinations(sorted(cart), size):
                for r in self.by_antecedent.get(frozenset(sub), ()):
                    c = r["consequent"]
                    if c in cart:
                        continue
                    if c not in best or self._key(r) > self._key(best[c]):
                        best[c] = r
        return best

    def recommend(self, cart: Iterable[str], k: int = 3, fallback: bool = True) -> tuple:
        """Tuple of (category, source, rule|None); source is 'rule' or 'popularity'."""
        cart = frozenset(cart)
        ck = (cart, k, fallback)
        if ck in self._cache:
            return self._cache[ck]
        best = self.fired(cart)
        ranked = sorted(best.items(), key=lambda kv: tuple(-x for x in self._key(kv[1])) + (kv[0],))
        out = [(c, "rule", r) for c, r in ranked[:k]]
        if fallback and len(out) < k:
            chosen = {c for c, _, _ in out} | cart
            for c in self.popularity:
                if len(out) >= k:
                    break
                if c not in chosen:
                    out.append((c, "popularity", None))
                    chosen.add(c)
        result = tuple(out)
        self._cache[ck] = result
        return result


def recommend_categories(cart: Iterable[str], rules: list[dict], k: int = 3,
                         popularity: dict[str, float] | None = None,
                         rank_by: str = "lift") -> list[str]:
    return [c for c, _, _ in RuleIndex(rules, popularity, rank_by).recommend(cart, k)]


class OfflineKG:
    """Catalog-backed stand-in for KGRetriever (no Neo4j needed).

    `filter_products` has the same semantics and ordering as the Cypher
    template; tests/test_neo4j_integration.py asserts the two agree.
    """

    def __init__(self, catalog: list[dict]):
        self.catalog = catalog

    def find_products(self, constraints: dict, limit: int = 5) -> list[dict]:
        return filter_products(self.catalog, constraints["category"], constraints.get("max_price"),
                               constraints.get("dietary_tags", []),
                               constraints.get("excluded_allergens", []), limit=limit)

    def cheapest_in_category(self, category, dietary_tags=(), excluded_allergens=()):
        hits = filter_products(self.catalog, category, None, dietary_tags, excluded_allergens, limit=1)
        return hits[0] if hits else None

    def close(self) -> None:
        pass


def recommend_addons(cart_categories: Iterable[str], constraints: dict, index: RuleIndex, kg,
                     k: int = 3, max_price_applies: bool = False) -> list[dict]:
    """Up to k add-on products: cheapest compliant product per suggested category.

    `kg` is a KGRetriever (Neo4j) or an OfflineKG. Categories with no compliant
    product are skipped and the next suggestion is tried.
    """
    tags = list(constraints.get("dietary_tags", []))
    allergens = list(constraints.get("excluded_allergens", []))
    out: list[dict] = []
    for cat, source, rule in index.recommend(cart_categories, k=len(index.popularity) or 19):
        if len(out) >= k:
            break
        product = kg.cheapest_in_category(cat, tags, allergens)
        if product is None:
            continue
        if max_price_applies and constraints.get("max_price") is not None \
                and product["price"] > constraints["max_price"]:
            continue
        out.append({"category": cat, "source": source, "rule": rule, "product": product})
    return out
