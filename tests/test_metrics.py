"""Tests for benchmark metrics and statistics."""
import math

import pytest

from clickless_eval.metrics import (
    bootstrap_mean_ci, by_stratum, click_reduction, clicks_manual, constraint_checks, css,
    holm, mcnemar_exact, mrr, paired_bootstrap_diff_ci, recall_at_k, summarize,
    task_success, violated, wilson_ci,
)

PROD = {"code": "1", "category": "chips", "price": 4.0,
        "dietary_tags": ["vegan", "gluten_free"], "allergens": ["soy"]}
EXP = {"category": "snacks", "dietary_tags": ["vegan", "organic"],
       "excluded_allergens": ["soy", "milk"], "max_price": 5.0, "quantity": 2}


def test_constraint_checks_and_css():
    checks = dict(constraint_checks(PROD, EXP))
    assert checks == {"category": True, "tag:vegan": True, "tag:organic": False,
                      "allergen:soy": False, "allergen:milk": True, "price": True}
    assert css(PROD, EXP) == pytest.approx(4 / 6)
    assert violated(PROD, EXP) == ["tag:organic", "allergen:soy"]


def test_no_product_scores_zero_css():
    assert css(None, EXP) == 0.0


def test_category_only_prompt_has_one_constraint():
    assert css(PROD, {"category": "chips", "dietary_tags": [], "excluded_allergens": [],
                      "max_price": None, "quantity": 1}) == 1.0


def test_task_success_with_and_without_quantity():
    assert task_success(PROD, ["1", "2"])
    assert not task_success(PROD, ["2"])
    assert not task_success(None, ["1"])
    assert task_success(PROD, ["1"], quantity=2, expected_quantity=2, check_quantity=True)
    assert not task_success(PROD, ["1"], quantity=1, expected_quantity=2, check_quantity=True)


def _r(feasible, product, acceptable=("1",)):
    return {"feasible": feasible, "product": product, "acceptable": list(acceptable),
            "expected": {**EXP, "dietary_tags": [], "excluded_allergens": []}}


def test_summarize_infeasible_confusion():
    results = [_r(True, PROD), _r(True, None), _r(False, None, ()), _r(False, PROD, ())]
    s = summarize(results)
    assert s["n_feasible"] == 2 and s["n_infeasible"] == 2
    assert s["tsr"] == 0.5 and s["ira"] == 0.5
    assert s["infeasible_confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert s["infeasible_precision"] == 0.5 and s["infeasible_recall"] == 0.5
    assert s["over_refusal_rate"] == 0.5
    assert s["css"] == pytest.approx((1.0 + 0.0) / 2)


def test_by_stratum():
    results = [dict(_r(True, PROD), n=1), dict(_r(True, None), n=2)]
    st = by_stratum(results, "n")
    assert st["1"]["tsr"] == 1.0 and st["2"]["tsr"] == 0.0


def test_recall_and_mrr():
    ranks = [1, 3, None, 25]
    assert recall_at_k(ranks, 20) == 0.5
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 25) / 4)


def test_click_model():
    assert clicks_manual(f=2, r=3) == 7
    assert click_reduction(f=2, r=3, e=0) == pytest.approx((7 - 2) / 7 * 100)


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_ci(0, 0) is None
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
    assert wilson_ci(10, 10)[0] == pytest.approx(0.7225, abs=1e-3)


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 2
    b = [False] * 10 + [True] * 2          # b=10, c=2
    res = mcnemar_exact(a, b)
    expected = 2 * sum(math.comb(12, i) for i in range(3)) / 2 ** 12
    assert res["b"] == 10 and res["c"] == 2
    assert res["p_value"] == pytest.approx(expected)
    assert mcnemar_exact([True], [True])["p_value"] == 1.0


def test_bootstrap_cis_contain_mean():
    vals = [0.0, 0.5, 1.0, 1.0, 0.5, 0.25]
    lo, hi = bootstrap_mean_ci(vals, n_boot=2000)
    assert lo <= sum(vals) / len(vals) <= hi
    d = paired_bootstrap_diff_ci([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], n_boot=2000)
    assert d["mean_diff"] == pytest.approx(5 / 6) and d["ci95"][0] > 0 and d["p_value"] < 0.05


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}
