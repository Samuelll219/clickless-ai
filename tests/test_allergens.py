from clickless_kg.allergens import infer_allergens
from clickless_kg.curate import evaluate, normalize_record
from conftest import off_record


def test_finds_missing_allergens():
    assert infer_allergens("dry roasted organic peanuts and sea salt.") == ["peanuts"]
    assert infer_allergens("cultured milk, enzymes, salt") == ["milk"]
    assert infer_allergens("Chocolate con leche y almendras") == ["milk", "tree_nuts"]
    assert infer_allergens("rice flour, sugar, malt extract") == ["gluten"]
    assert infer_allergens("whole wheat flour, sesame seeds") == ["gluten", "sesame"]


def test_avoids_common_false_positives():
    assert infer_allergens("corn, maltodextrin, salt") == []                 # maltodextrin is not malt
    assert infer_allergens("almondmilk (water, almonds), coconut milk") == ["tree_nuts"]
    assert infer_allergens("cacao, cane sugar, cocoa-butter") == []          # cocoa butter is not dairy
    assert infer_allergens("peanuts. peanut butter") == ["peanuts"]
    assert infer_allergens("buckwheat flour, water") == []                   # buckwheat is not wheat
    assert infer_allergens("oats, salt. gluten-free. dairy free") == []
    assert infer_allergens("corn, salt. May contain milk, wheat and soy.") == []
    assert infer_allergens("blé dur. Traces d'œufs.") == ["gluten"]
    assert infer_allergens("avoine, beurre de cacahuètes") == ["peanuts"]


def test_name_used_except_for_milk_and_eggs():
    assert infer_allergens("almonds", name="Vegan Cream Cheese Style Spread") == ["tree_nuts"]
    assert infer_allergens("water, oats", name="Peanut Oat Bar") == ["peanuts"]


def test_vegan_label_blocks_animal_inference():
    assert infer_allergens("cashews, cultures", name="", vegan=True) == ["tree_nuts"]
    assert infer_allergens("milk", vegan=True) == []


def test_catalog_uses_union_and_records_additions():
    p, reason = evaluate(normalize_record(off_record(
        40, "Organic Creamy Peanut Butter", ["en:peanut-butters"], labels=["en:organic"],
        ingredients="dry roasted organic peanuts and sea salt.", allergens=[])))
    assert reason == "ok" and p["allergens"] == ["peanuts"] and p["allergens_added"] == ["peanuts"]


def test_dairy_categories_always_contain_milk():
    p, _ = evaluate(normalize_record(off_record(
        41, "Whole Milk", ["en:milks"], labels=["en:organic"], ingredients="grade a milk, vitamin d3")))
    assert "milk" in p["allergens"]
    p, _ = evaluate(normalize_record(off_record(
        42, "Fairlife", ["en:milks"], labels=["en:organic"], ingredients="ultra filtered nonfat liquid")))
    assert "milk" in p["allergens"]
