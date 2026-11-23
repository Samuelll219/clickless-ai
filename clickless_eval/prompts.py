"""Test-prompt set: template rendering, annotation and validation (proposal Sec. V.A).

Every prompt carries its ground-truth constraint JSON (`expected`) and the set
of acceptable products computed from the catalog with exactly the Cypher
semantics (`clickless_kg.catalog.acceptable_set`). A prompt is infeasible iff
that set is empty.

Counting convention (used for the 1 / 2 / 3+ strata): each required dietary
tag, each excluded allergen and the price cap is one constraint; the category
is always present and is not counted; quantity is tracked separately.
"""
from __future__ import annotations

from clickless_kg import vocab
from clickless_kg.catalog import acceptable_set

TAG_PHRASES = {
    "gluten_free": "gluten-free", "organic": "organic", "vegan": "vegan",
    "vegetarian": "vegetarian", "non_gmo": "non-GMO", "lactose_free": "lactose-free",
    "kosher": "kosher", "no_added_sugar": "no-added-sugar", "fair_trade": "fair-trade",
}
ALLERGEN_PHRASES = {
    "gluten": "gluten", "milk": "milk", "eggs": "eggs", "peanuts": "peanuts",
    "tree_nuts": "tree nuts", "soy": "soy", "fish": "fish", "shellfish": "shellfish",
    "sesame": "sesame", "mustard": "mustard", "celery": "celery", "lupin": "lupin",
    "sulphites": "sulphites",
}
CATEGORY_PHRASES = {
    "plant_milk": "plant-based milk", "milk": "milk", "yogurt": "yogurt", "cheese": "cheese",
    "nut_butters": "nut butter", "breakfast_cereal": "breakfast cereal",
    "snack_bars": "snack bars", "chips": "chips", "crackers": "crackers",
    "cookies": "cookies", "chocolate_candy": "chocolate", "nuts_seeds": "nuts or seeds",
    "bread": "bread", "rice_grains": "rice or grains", "pasta": "pasta",
    "pasta_sauce": "pasta sauce", "soup": "soup", "juice": "juice",
    "coffee_tea": "coffee or tea",
    # groups
    "snacks": "snacks", "dairy": "dairy products", "beverages": "drinks",
    "pantry": "pantry staples", "breakfast": "breakfast food", "bakery": "bakery items",
}

ROW_KEYS = ("id", "prompt", "template", "expected", "n_constraints", "constraint_types",
            "stratum", "feasible", "infeasible_reason", "acceptable", "n_acceptable",
            "category_level", "source", "paraphrase_model", "verified_by", "notes")


def count_constraints(expected: dict) -> int:
    return (len(expected.get("dietary_tags", [])) + len(expected.get("excluded_allergens", []))
            + (expected.get("max_price") is not None))


def constraint_types(expected: dict) -> list[str]:
    types = []
    if expected.get("dietary_tags"):
        types.append("tag")
    if expected.get("excluded_allergens"):
        types.append("allergen")
    if expected.get("max_price") is not None:
        types.append("price")
    return types


def stratum(expected: dict) -> str:
    n = count_constraints(expected)
    return "3+" if n >= 3 else str(n)


def _allergen_clause(allergens: list[str]) -> str:
    if set(allergens) >= {"peanuts", "tree_nuts"}:
        rest = [a for a in allergens if a not in ("peanuts", "tree_nuts")]
        parts = ["nuts"] + [ALLERGEN_PHRASES[a] for a in rest]
    else:
        parts = [ALLERGEN_PHRASES[a] for a in allergens]
    return "with no " + " or ".join(parts) if len(parts) <= 2 else \
        "with no " + ", ".join(parts[:-1]) + " or " + parts[-1]


def render_template(expected: dict) -> str:
    """Canonical (template) wording of a constraint JSON."""
    qty = expected.get("quantity", 1)
    words = ["Add"]
    if qty > 1:
        words.append(f"{qty} packs of")
    words += [TAG_PHRASES[t] for t in expected.get("dietary_tags", [])]
    words.append(CATEGORY_PHRASES[expected["category"]])
    if expected.get("excluded_allergens"):
        words.append(_allergen_clause(expected["excluded_allergens"]))
    if expected.get("max_price") is not None:
        p = expected["max_price"]
        words.append(f"under ${p:g}" if float(p).is_integer() else f"under ${p:.2f}")
    words.append("to my cart.")
    return " ".join(words)


def make_expected(category: str, tags=(), excluded=(), max_price=None, quantity: int = 1) -> dict:
    return {"category": category, "dietary_tags": list(tags),
            "excluded_allergens": list(excluded), "max_price": max_price,
            "quantity": quantity}


def annotate(row: dict, catalog: list[dict]) -> dict:
    """Fill the derived fields of a test-prompt row from its `expected` JSON."""
    exp = row["expected"]
    acc = acceptable_set(catalog, exp)
    out = dict(row)
    out.update({
        "n_constraints": count_constraints(exp),
        "constraint_types": constraint_types(exp),
        "stratum": stratum(exp),
        "feasible": bool(acc),
        "acceptable": acc,
        "n_acceptable": len(acc),
        "category_level": "group" if exp["category"] in vocab.CATEGORY_GROUPS else "leaf",
    })
    out.setdefault("template", render_template(exp))
    return {k: out.get(k) for k in ROW_KEYS if k in out} | \
        {k: v for k, v in out.items() if k not in ROW_KEYS}


def validate_test_set(rows: list[dict], catalog: list[dict], *, dev_rows: list[dict] = (),
                      expected_total: int | None = 80,
                      expected_infeasible: int | None = 10) -> list[str]:
    """Return a list of problems (empty = valid)."""
    from clickless_llm.validator import ValidationError, validate_constraints

    problems: list[str] = []
    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    prompts = [r.get("prompt", "").strip().lower() for r in rows]
    if len(set(prompts)) != len(prompts):
        problems.append("duplicate prompt texts")
    dev_texts = {d["prompt"].strip().lower() for d in dev_rows}
    dev_expected = {_key(d["expected"]) for d in dev_rows}
    for r in rows:
        rid = r.get("id")
        if not r.get("prompt", "").strip():
            problems.append(f"{rid}: empty prompt")
        try:
            validate_constraints(dict(r["expected"]))
        except ValidationError as exc:
            problems.append(f"{rid}: invalid expected JSON: {exc}")
            continue
        acc = acceptable_set(catalog, r["expected"])
        if acc != r.get("acceptable"):
            problems.append(f"{rid}: acceptable set out of date with catalog")
        if bool(acc) != r.get("feasible"):
            problems.append(f"{rid}: feasible flag wrong")
        if not r.get("feasible") and not r.get("infeasible_reason"):
            problems.append(f"{rid}: infeasible prompt without infeasible_reason")
        if r["prompt"].strip().lower() in dev_texts:
            problems.append(f"{rid}: prompt text duplicates a development prompt")
        if _key(r["expected"]) in dev_expected:
            problems.append(f"{rid}: expected JSON duplicates a development prompt")
    if expected_total is not None and len(rows) != expected_total:
        problems.append(f"expected {expected_total} prompts, found {len(rows)}")
    n_inf = sum(not r.get("feasible") for r in rows)
    if expected_infeasible is not None and n_inf != expected_infeasible:
        problems.append(f"expected {expected_infeasible} infeasible prompts, found {n_inf}")
    return problems


def _key(exp: dict) -> tuple:
    return (exp["category"], tuple(sorted(exp.get("dietary_tags", []))),
            tuple(sorted(exp.get("excluded_allergens", []))), exp.get("max_price"),
            exp.get("quantity", 1))
