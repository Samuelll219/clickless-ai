"""Category-level association rules (proposal Sec. IV.C, Eqs. 1-3).

For a transaction database T of n category sets and category sets X, Y:

    sup(X -> Y)  = (X u Y).count / n                         (1)
    conf(X -> Y) = sup(X -> Y) / sup(X)                       (2)
    lift(X -> Y) = conf(X -> Y) / sup(Y)                      (3)

Frequent itemsets are mined level-wise with Apriori (Agrawal & Srikant, VLDB
1994) via mlxtend; FP-growth must give identical itemsets (used as a check).
Rule template (constrained rule mining): a single consequent category,
lift > 1 (positive association), antecedent of up to max_len - 1 categories.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

RULE_COLUMNS = ["antecedents", "consequents", "antecedent support", "consequent support",
                "support", "confidence", "lift"]


def one_hot(baskets: Sequence[Iterable[str]], items: Sequence[str] | None = None) -> pd.DataFrame:
    """Boolean transaction matrix (rows = baskets, columns = sorted items)."""
    baskets = [frozenset(b) for b in baskets]
    items = sorted(set().union(*baskets)) if items is None else sorted(items)
    col = {it: j for j, it in enumerate(items)}
    mat = np.zeros((len(baskets), len(items)), dtype=bool)
    for i, b in enumerate(baskets):
        for it in b:
            if it in col:
                mat[i, col[it]] = True
    return pd.DataFrame(mat, columns=items)


def frequent_itemsets(df: pd.DataFrame, min_support: float, max_len: int = 3,
                      algorithm: str = "apriori") -> pd.DataFrame:
    fn = {"apriori": apriori, "fpgrowth": fpgrowth}[algorithm]
    freq = fn(df, min_support=min_support, use_colnames=True, max_len=max_len)
    freq["length"] = freq["itemsets"].map(len)
    return freq


def level_stats(freq: pd.DataFrame) -> dict[int, int]:
    """|F_k| for each level k (downward-closure / level-wise search table)."""
    return {int(k): int(v) for k, v in freq["length"].value_counts().sort_index().items()}


def mine_rules(baskets: Sequence[Iterable[str]], min_support: float, *, max_len: int = 3,
               min_lift: float = 1.0, min_confidence: float = 0.0,
               algorithm: str = "apriori") -> tuple[pd.DataFrame, dict]:
    """Single-consequent rules with lift > min_lift; returns (rules, info)."""
    df = one_hot(baskets)
    freq = frequent_itemsets(df, min_support, max_len=max_len, algorithm=algorithm)
    info = {"n_transactions": int(len(df)), "n_items": int(df.shape[1]),
            "min_support": min_support, "max_len": max_len, "min_lift": min_lift,
            "min_confidence": min_confidence, "algorithm": algorithm,
            "frequent_itemsets_per_level": level_stats(freq)}
    if freq.empty or freq["length"].max() < 2:
        info.update(n_rules_all=0, n_rules=0)
        return pd.DataFrame(columns=RULE_COLUMNS), info
    rules = association_rules(freq, num_itemsets=len(df), metric="lift", min_threshold=0.0)
    info["n_rules_all"] = int(len(rules))
    rules = rules[(rules["consequents"].map(len) == 1)
                  & (rules["lift"] > min_lift)
                  & (rules["confidence"] >= min_confidence)]
    rules = rules[RULE_COLUMNS].sort_values(["lift", "confidence", "support"], ascending=False)
    info["n_rules"] = int(len(rules))
    return rules.reset_index(drop=True), info


def held_out_quality(rules: list[dict], baskets: Sequence[Iterable[str]]) -> list[dict]:
    """Re-measure rules on held-out baskets: support, confidence (= rule precision), lift.

    The course standard (classifier lecture p7): "Keep concise rules with
    held-out support, precision, coverage, and lift. Frequency alone is
    insufficient."
    """
    sets = [frozenset(b) for b in baskets]
    n = len(sets)
    item_count = Counter(it for s in sets for it in s)
    out = []
    for r in rules:
        lhs = frozenset(r["antecedents"])
        c = r["consequent"]
        n_lhs = sum(lhs <= s for s in sets)
        n_both = sum(lhs <= s and c in s for s in sets)
        sup_c = item_count[c] / n if n else 0.0
        conf = n_both / n_lhs if n_lhs else 0.0
        out.append({**r, "heldout_support": n_both / n if n else 0.0,
                    "heldout_confidence": conf,
                    "heldout_lift": conf / sup_c if sup_c else 0.0,
                    "heldout_antecedent_count": n_lhs})
    return out


def popularity(baskets: Sequence[Iterable[str]]) -> dict[str, float]:
    """Support of every single category (the most-popular baseline)."""
    counts = Counter(it for b in baskets for it in set(b))
    n = len(baskets)
    return {it: c / n for it, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def rules_to_records(rules: pd.DataFrame) -> list[dict]:
    return [{"antecedents": sorted(r["antecedents"]), "consequent": next(iter(r["consequents"])),
             "antecedent_support": float(r["antecedent support"]),
             "consequent_support": float(r["consequent support"]),
             "support": float(r["support"]), "confidence": float(r["confidence"]),
             "lift": float(r["lift"])} for _, r in rules.iterrows()]


def save_rules(path: str | Path, rules: pd.DataFrame, popularity_: dict[str, float],
               meta: dict) -> None:
    payload = {"meta": meta, "popularity": popularity_, "rules": rules_to_records(rules)}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_rules(path: str | Path) -> dict:
    """{"meta", "popularity", "rules": [ {antecedents, consequent, support, confidence, lift, ...} ]}"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_rule(rule: dict) -> str:
    """Course notation: X -> Y [sup = .., conf = .., lift = ..]."""
    lhs = ", ".join(rule["antecedents"])
    return (f"{lhs} -> {rule['consequent']} [sup = {rule['support']:.3f}, "
            f"conf = {rule['confidence']:.3f}, lift = {rule['lift']:.2f}]")
