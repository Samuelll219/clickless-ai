from collections import Counter

from clickless_kg import vocab
from clickless_kg.catalog import validate_catalog
from clickless_kg.curate import assign_category, curate, evaluate, normalize_record
from clickless_kg.pricing import assign_price
from conftest import off_record


def ev(rec):
    return evaluate(normalize_record(rec))


def test_category_priority_and_exclusions():
    assert assign_category(["en:foods", "en:snacks", "en:crackers"]) == "crackers"
    assert assign_category(["en:milks", "en:plant-based-milk-alternatives"]) == "plant_milk"
    assert assign_category(["en:nuts", "en:nut-butters"]) == "nut_butters"
    assert assign_category(["en:frozen-foods"]) is None
    # OFF puts cereal grains under en:seeds; they must not end up in nuts_seeds
    assert assign_category(["en:seeds", "en:cereals-and-their-products", "en:cereal-grains",
                            "en:rices"]) == "rice_grains"
    assert assign_category(["en:seeds", "en:cereals-and-their-products", "en:oat-flakes"]) is None
    assert assign_category(["en:seeds", "en:chia-seeds"]) == "nuts_seeds"


def test_most_specific_tag_wins_and_global_excludes():
    # Lay's chips mis-tagged as cheese by OFF: the later, more specific tag wins
    assert assign_category(["en:dairies", "en:cheeses", "en:salty-snacks", "en:crisps",
                            "en:potato-crisps"]) == "chips"
    assert assign_category(["en:biscuits", "en:crackers-appetizers"]) == "crackers"
    assert assign_category(["en:bars", "en:chocolates", "en:milk-chocolates"]) == "chocolate_candy"
    assert assign_category(["en:pastas", "en:meats", "en:pate"]) is None
    assert assign_category(["en:seeds", "en:legume-seeds", "en:pulses", "en:black-beans"]) is None
    assert assign_category(["en:sauces", "en:tomato-sauces", "en:ketchup"]) is None
    assert assign_category(["en:condiments", "en:sauces", "en:pasta-sauces"]) == "pasta_sauce"


def test_biscuit_name_overrides():
    p, _ = ev(off_record(30, "Reduced Fat Triscuit Crackers", ["en:biscuits"], labels=["en:kosher"]))
    assert p["category"] == "crackers"
    p, _ = ev(off_record(31, "MacroBar", ["en:biscuits"], labels=["en:kosher"]))
    assert p["category"] == "snack_bars"
    p, _ = ev(off_record(32, "Oreo", ["en:biscuits"], labels=["en:kosher"]))
    assert p["category"] == "cookies"


def test_tag_and_allergen_normalisation():
    p, reason = ev(off_record(1, "Oat bar", ["en:cereal-bars"],
                              labels=["en:no-gluten", "en:usda-organic", "en:some-unknown-label"],
                              allergens=["en:nuts", "en:soybeans", "en:not-an-allergen"]))
    assert reason == "ok"
    assert p["category"] == "snack_bars"
    assert p["dietary_tags"] == ["gluten_free", "organic"]
    assert p["allergens"] == ["soy", "tree_nuts"]


def test_vegan_implies_vegetarian_and_analysis_tags():
    p, _ = ev(off_record(2, "Chips", ["en:crisps"], labels=["en:vegan"]))
    assert p["dietary_tags"] == ["vegan", "vegetarian"]
    rec = normalize_record(off_record(3, "Chips 2", ["en:crisps"], labels=["en:kosher"],
                                      analysis=["en:vegan", "en:maybe-palm-oil"]))
    assert evaluate(rec, use_analysis=True)[0]["dietary_tags"] == ["kosher", "vegan", "vegetarian"]
    assert evaluate(rec, use_analysis=False)[0]["dietary_tags"] == ["kosher"]


def test_maybe_vegan_is_not_vegan():
    p, _ = ev(off_record(4, "Chips", ["en:crisps"], labels=["en:kosher"], analysis=["en:maybe-vegan"]))
    assert "vegan" not in p["dietary_tags"]


def test_rejections():
    assert ev(off_record(5, "", ["en:crisps"], labels=["en:vegan"]))[1] == "missing_code_or_name"
    assert ev(off_record(6, "X", ["en:crisps"], labels=["en:vegan"], ingredients=""))[1] == "missing_ingredients"
    assert ev(off_record(7, "X", ["en:crisps"], labels=["en:vegan"], ingredients_n=0))[1] == "ingredients_not_analysed"
    assert ev(off_record(8, "X", ["en:frozen-foods"], labels=["en:vegan"]))[1] == "no_matching_category"
    assert ev(off_record(9, "X", ["en:crisps"]))[1] == "too_few_dietary_tags"
    assert ev(off_record(10, "X", ["en:breads"], labels=["en:no-gluten"],
                         allergens=["en:gluten"]))[1] == "conflict_gluten_free"
    assert ev(off_record(11, "X", ["en:cheeses"], labels=["en:vegan"],
                         allergens=["en:milk"]))[1] == "conflict_vegan"


def test_csv_style_record():
    raw = {"code": "123", "product_name": "Pasta", "brands": "A,B",
           "categories_tags": "en:foods,en:pastas", "labels_tags": "en:organic",
           "allergens": "en:gluten", "ingredients_text": "durum wheat semolina",
           "ingredients_analysis_tags": float("nan"), "states_tags": "en:ingredients-completed"}
    p, reason = evaluate(normalize_record(raw))
    assert reason == "ok" and p["brand"] == "A" and p["allergens"] == ["gluten"]


def test_prices_reproducible_and_in_range():
    for cat in vocab.CATEGORY_NAMES:
        lo, hi = vocab.CATEGORY_BY_NAME[cat].price_range
        for code in ("1", "2", "0001234567890"):
            p = assign_price(code, cat, 573)
            assert lo <= p <= hi
            assert p == assign_price(code, cat, 573)
    assert assign_price("42", "chips", 1) != assign_price("42", "chips", 2)


def test_curate_hits_target_balanced_and_deterministic(dump):
    cat1, rep = curate(dump, target=400, seed=573)
    cat2, _ = curate(list(reversed(dump)), target=400, seed=573)
    assert len(cat1) == 400
    assert cat1 == cat2  # input order does not matter
    counts = Counter(p["category"] for p in cat1)
    assert set(counts) == set(vocab.CATEGORY_NAMES)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert validate_catalog(cat1) == []
    assert rep["selected"] == 400


def test_short_category_quota_redistributed(dump):
    dump = [r for r in dump if "en:soups" not in r["categories_tags"]][:] + \
        [r for r in dump if "en:soups" in r["categories_tags"]][:5]
    cat, rep = curate(dump, target=400)
    assert len(cat) == 400
    assert rep["selected_per_category"]["soup"] == 5


def test_duplicates_removed():
    a = off_record(20, "Same Name", ["en:crisps"], labels=["en:vegan"], brand="B")
    b = off_record(21, "same  name", ["en:crisps"], labels=["en:vegan"], brand="b")
    c = dict(a)
    cat, rep = curate([a, b, c], target=10)
    assert len(cat) == 1
    assert rep["rejections"]["duplicate_code"] == 1
    assert rep["rejections"]["duplicate_name_brand"] == 1
