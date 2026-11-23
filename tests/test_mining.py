"""Tests for the aisle map, Instacart transactions, Apriori rules, add-ons and Hit@k.

Uses a hand-computable 10-basket database (no Instacart download needed):

    milk 6/10, breakfast_cereal 5/10, bread 4/10, coffee_tea 4/10
    {milk, cereal} 4/10 -> conf(milk->cereal) = 4/6, lift = (4/6)/(5/10) = 4/3
                          conf(cereal->milk) = 4/5, lift = (4/5)/(6/10) = 4/3
    {bread, coffee} 2/10 -> conf = 2/4 both ways, lift = 0.5/0.4 = 1.25
    {milk, bread} 2/10  -> lift(milk->bread) = (2/6)/0.4 = 0.83 < 1 (dropped)
"""
from unittest.mock import Mock

import pandas as pd
import pytest

from clickless_kg import vocab
from clickless_mining import aisles, instacart
from clickless_mining.addons import OfflineKG, RuleIndex, recommend_addons, recommend_categories
from clickless_mining.apriori_rules import (
    format_rule, frequent_itemsets, held_out_quality, load_rules, mine_rules, one_hot, popularity,
    rules_to_records, save_rules,
)
from clickless_mining.hit_eval import (
    compare, evaluate, make_queries, popularity_recommender, random_recommender,
    rule_coverage, rule_recommender,
)

M, C, B, K = "milk", "breakfast_cereal", "bread", "coffee_tea"
BASKETS = [(M, C), (M, C), (M, C, B), (M, B), (C,), (B, K), (B, K), (K,), (M,), (M, C, K)]


@pytest.fixture(scope="module")
def mined():
    rules, info = mine_rules(BASKETS, min_support=0.2, max_len=2)
    return rules_to_records(rules), info


def _rule(records, lhs, rhs):
    return next(r for r in records if r["antecedents"] == [lhs] and r["consequent"] == rhs)


# ------------------------------------------------------------------ aisle mapping

def test_aisle_mapping_is_valid_and_covers_all_leaves():
    assert aisles.validate_mapping() == []
    assert set(aisles.mapping().values()) == set(vocab.CATEGORY_NAMES)
    assert set(aisles.mapping().values()) <= set(vocab.CATEGORY_NAMES)    # leaves only
    assert not set(aisles.mapping().values()) & set(vocab.CATEGORY_GROUPS)
    assert aisles.mapping()[84] == "milk" and aisles.mapping()[107] == "chips"
    assert 24 not in aisles.mapping()                     # fresh fruits: unmapped
    assert 48 not in aisles.mapping() and 48 in aisles.mapping(include_optional=True)


def test_validate_against_aisles_csv(tmp_path):
    p = tmp_path / "aisles.csv"
    rows = ["aisle_id,aisle"] + [f"{i},aisle {i}" for i in range(1, 135)]
    p.write_text("\n".join(rows))
    problems = aisles.validate_against_aisles_csv(p)
    assert any("aisle 84" in x for x in problems)          # names differ in the fake file


# ------------------------------------------------------------------ rules

def test_support_confidence_lift_match_hand_values(mined):
    records, info = mined
    r = _rule(records, M, C)
    assert r["support"] == pytest.approx(0.4)
    assert r["confidence"] == pytest.approx(4 / 6)
    assert r["lift"] == pytest.approx(4 / 3)
    r = _rule(records, C, M)
    assert r["confidence"] == pytest.approx(0.8) and r["lift"] == pytest.approx(4 / 3)
    r = _rule(records, B, K)
    assert r["confidence"] == pytest.approx(0.5) and r["lift"] == pytest.approx(1.25)
    assert not any(r["antecedents"] == [M] and r["consequent"] == B for r in records)  # lift < 1
    assert info["frequent_itemsets_per_level"] == {1: 4, 2: 3}   # {M,C},{B,K},{M,B}


def test_apriori_and_fpgrowth_agree():
    df = one_hot(BASKETS)
    a = frequent_itemsets(df, 0.2, algorithm="apriori")
    f = frequent_itemsets(df, 0.2, algorithm="fpgrowth")
    as_set = lambda fr: {(frozenset(i), round(s, 9)) for i, s in zip(fr["itemsets"], fr["support"])}
    assert as_set(a) == as_set(f)


def test_popularity_and_rule_roundtrip(tmp_path, mined):
    pop = popularity(BASKETS)
    assert list(pop) == [M, C, B, K] and pop[M] == pytest.approx(0.6)
    records, _ = mined
    rules_df, _ = mine_rules(BASKETS, min_support=0.2, max_len=2)
    save_rules(tmp_path / "r.json", rules_df, pop, {"note": "test"})
    loaded = load_rules(tmp_path / "r.json")
    assert loaded["rules"] == records and loaded["popularity"] == pop
    assert format_rule(_rule(records, M, C)).startswith("milk -> breakfast_cereal [sup = 0.400")


def test_held_out_quality(mined):
    records, _ = mined
    held = [((M, C)), ((M,)), ((C,)), ((M, C))]
    q = held_out_quality([_rule(records, M, C)], held)[0]
    assert q["heldout_support"] == pytest.approx(0.5)          # 2 of 4 baskets contain both
    assert q["heldout_confidence"] == pytest.approx(2 / 3)     # 2 of 3 milk baskets
    assert q["heldout_lift"] == pytest.approx((2 / 3) / 0.75)


def test_no_rules_when_support_too_high():
    rules, info = mine_rules(BASKETS, min_support=0.9)
    assert rules.empty and info["n_rules"] == 0


# ------------------------------------------------------------------ recommender

def test_recommend_excludes_cart_and_falls_back_to_popularity(mined):
    records, _ = mined
    index = RuleIndex(records, popularity(BASKETS))
    rec = index.recommend([M], k=3)
    assert [c for c, _, _ in rec] == [C, B, K]
    assert [s for _, s, _ in rec] == ["rule", "popularity", "popularity"]
    assert [c for c, _, _ in index.recommend([B], k=3)] == [K, M, C]
    assert index.recommend([M], k=3, fallback=False) == ((C, "rule", _rule(records, M, C)),)
    assert recommend_categories([M, C], records, k=2, popularity=popularity(BASKETS)) == [B, K]


def test_rank_by_confidence_changes_order():
    rules = [{"antecedents": ["a"], "consequent": "x", "support": .1, "confidence": .9, "lift": 1.1},
             {"antecedents": ["a"], "consequent": "y", "support": .1, "confidence": .2, "lift": 3.0}]
    assert recommend_categories(["a"], rules, k=2, rank_by="lift") == ["y", "x"]
    assert recommend_categories(["a"], rules, k=2, rank_by="confidence") == ["x", "y"]


def test_recommend_addons_forwards_constraints_and_skips_missing(mined):
    records, _ = mined
    index = RuleIndex(records, popularity(BASKETS))
    kg = Mock()
    kg.cheapest_in_category.side_effect = lambda cat, tags, allergens: (
        None if cat == C else {"code": cat, "price": 3.0, "category": cat})
    out = recommend_addons([M], {"dietary_tags": ["vegan"], "excluded_allergens": ["soy"]},
                           index, kg, k=2)
    assert [o["category"] for o in out] == [B, K]          # cereal skipped (no product)
    kg.cheapest_in_category.assert_any_call(C, ["vegan"], ["soy"])


def test_offline_kg_matches_filter_semantics():
    catalog = [{"code": "1", "category": "bread", "price": 3.0, "dietary_tags": ["vegan"], "allergens": []},
               {"code": "2", "category": "bread", "price": 2.0, "dietary_tags": [], "allergens": []}]
    kg = OfflineKG(catalog)
    assert kg.cheapest_in_category("bread")["code"] == "2"
    assert kg.cheapest_in_category("bread", ["vegan"])["code"] == "1"
    assert kg.cheapest_in_category("bread", ["organic"]) is None


# ------------------------------------------------------------------ Hit@k

def test_queries_protocols():
    q = make_queries([(M, C, B), (K,)], protocol="first_item")
    assert q == [((M,), frozenset({C, B}))]
    q = make_queries([(M, C, B), (K,)], protocol="leave_one_out", seed=1)
    assert len(q) == 1 and len(q[0][0]) == 2 and len(q[0][1]) == 1
    assert set(q[0][0]) | q[0][1] == {M, C, B}


def test_hit_at_k_and_baselines(mined):
    records, _ = mined
    pop = popularity(BASKETS)
    queries = [((M,), frozenset({C})), ((B,), frozenset({K})), ((K,), frozenset({M}))]
    rules = evaluate(queries, rule_recommender(RuleIndex(records, pop), fallback=False), k=1)
    assert rules["hit@1"] == pytest.approx(2 / 3)           # M->C, B->K hit; K->B misses M
    popr = evaluate(queries, popularity_recommender(pop), k=1)
    assert popr["hit@1"] == pytest.approx(2 / 3)            # C, M, M
    assert compare(rules, popr)["b"] == 1 and compare(rules, popr)["c"] == 1
    rnd = evaluate(queries, random_recommender([M, C, B, K]), k=3)
    assert 0 <= rnd["hit@3"] <= 1
    assert rule_coverage(queries, RuleIndex(records, pop)) == 1.0


# ------------------------------------------------------------------ Instacart loading

def _write_fake_instacart(d):
    pd.DataFrame({"product_id": [1, 2, 3, 4], "aisle_id": [84, 121, 24, 112]}).to_csv(d / "products.csv", index=False)
    pd.DataFrame({"order_id": [10, 11, 12, 13], "user_id": [1, 1, 2, 3],
                  "eval_set": ["prior", "train", "prior", "test"],
                  "order_number": [1, 2, 1, 1]}).to_csv(d / "orders.csv", index=False)
    pd.DataFrame({"order_id": [10, 10, 10, 11, 12], "product_id": [3, 2, 1, 4, 3],
                  "add_to_cart_order": [1, 2, 3, 1, 1], "reordered": 0}).to_csv(d / "order_products__prior.csv", index=False)
    pd.DataFrame({"order_id": [11], "product_id": [1], "add_to_cart_order": [2],
                  "reordered": 0}).to_csv(d / "order_products__train.csv", index=False)


def test_build_transactions_and_splits(tmp_path):
    _write_fake_instacart(tmp_path)
    tx = instacart.build_transactions(tmp_path, aisles.mapping())
    by_order = dict(zip(tx.frame["order_id"], tx.frame["categories"]))
    assert by_order == {10: (C, M), 11: (B, M)}             # first-added order kept; fruit dropped
    assert tx.stats["order_lines"] == 6 and tx.stats["mapped_order_lines"] == 4
    assert tx.stats["orders_with_ge2_categories"] == 2
    instacart.save(tx, tmp_path / "tx.pkl")
    assert instacart.load(tmp_path / "tx.pkl").stats == tx.stats

    frame = pd.DataFrame({"user_id": [u for u in range(10) for _ in range(3)],
                          "order_id": range(30), "categories": [(M, C)] * 30})
    train, test = instacart.split_by_user(frame, 0.2, seed=573)
    assert set(train["user_id"]).isdisjoint(test["user_id"])
    assert test["user_id"].nunique() == 2 and len(train) + len(test) == 30
