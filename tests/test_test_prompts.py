"""Checks on the committed 80-prompt test set and the prompt helpers."""
import json
from pathlib import Path

import pytest

from clickless_kg.catalog import load_catalog
from clickless_eval.prompts import (
    count_constraints, make_expected, render_template, stratum, validate_test_set,
)
from clickless_llm.evaluation import CONSTRAINT_FIELDS

TEST_SET = Path("data/evaluation/test_prompts.json")
DEV_SET = Path("data/evaluation/constraint_prompts.json")


@pytest.fixture(scope="module")
def rows():
    return json.loads(TEST_SET.read_text(encoding="utf-8"))


def test_test_set_is_valid_against_current_catalog(rows):
    dev = json.loads(DEV_SET.read_text(encoding="utf-8"))
    assert validate_test_set(rows, load_catalog(), dev_rows=dev) == []


def test_composition(rows):
    assert len(rows) == 80
    feasible = [r for r in rows if r["feasible"]]
    assert len(feasible) == 70
    by_stratum = {s: sum(r["stratum"] == s for r in feasible) for s in ("1", "2", "3+")}
    assert all(n >= 20 for n in by_stratum.values()), by_stratum
    leaves = {r["expected"]["category"] for r in rows if r["category_level"] == "leaf"}
    assert len(leaves) == 19
    assert sum(r["expected"]["quantity"] > 1 for r in rows) >= 15


def test_rows_feed_existing_extraction_evaluator(rows):
    for r in rows:
        assert set(r["expected"]) == set(CONSTRAINT_FIELDS)
        assert r["prompt"].strip()


def test_infeasible_prompts_have_reasons(rows):
    for r in rows:
        assert (r["infeasible_reason"] is None) == r["feasible"]


def test_helpers():
    exp = make_expected("snacks", ["vegan"], ["peanuts", "tree_nuts"], 8, quantity=2)
    assert count_constraints(exp) == 4 and stratum(exp) == "3+"
    assert render_template(exp) == "Add 2 packs of vegan snacks with no nuts under $8 to my cart."
    assert render_template(make_expected("pasta", max_price=2.5)) == "Add pasta under $2.50 to my cart."


def test_validator_catches_problems():
    catalog = load_catalog()
    bad = [{"id": "X", "prompt": "p", "expected": make_expected("cheese", ["vegan"]),
            "feasible": True, "acceptable": ["123"], "infeasible_reason": None}]
    problems = validate_test_set(bad, catalog, expected_total=None, expected_infeasible=None)
    assert any("acceptable set out of date" in p for p in problems)
    assert any("feasible flag wrong" in p for p in problems)
